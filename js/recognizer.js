/**
 * recognizer.js
 * Runs the PaddleOCR CRNN text recognition model (rec.onnx).
 *
 * Pipeline per detected box:
 *   1. Crop the region from the preprocessed BGR image
 *   2. Resize to height=48, proportional width (min 10 px)
 *   3. Normalise → CHW float32 tensor  (mean=0.5, std=0.5)
 *   4. Run rec.onnx
 *   5. Greedy CTC decode using dict.txt  (blank token at index 0)
 */

const Recognizer = (() => {

  // ── Normalisation ──────────────────────────────────────────────────────────
  const REC_HEIGHT  = 48;
  const REC_MEAN    = 0.5;
  const REC_STD     = 0.5;
  const MIN_WIDTH   = 10;

  let session  = null;
  let charset  = null;   // ['[blank]', '0', '1', ..., 'Z', 'a', ...]

  // ── Session factory — WebGL first, WASM fallback ──────────────────────────
  async function _createSession(modelSource) {
    // Try WebGPU first (Chrome on Windows supports it, fastest)
    try {
      const s = await ort.InferenceSession.create(modelSource, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      console.log('[Recognizer] EP: webgpu ✅');
      return s;
    } catch (e) {
      console.warn('[Recognizer] WebGPU failed, trying WebGL:', e.message);
    }
    // Try WebGL
    try {
      const s = await ort.InferenceSession.create(modelSource, {
        executionProviders: ['webgl'],
        graphOptimizationLevel: 'all',
      });
      console.log('[Recognizer] EP: webgl ✅');
      return s;
    } catch (e) {
      console.warn('[Recognizer] WebGL failed, falling back to WASM:', e.message);
    }
    // WASM fallback
    console.log('[Recognizer] EP: wasm');
    return ort.InferenceSession.create(modelSource, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Load the recognition model and parse dict.txt.
   * @param {string} modelPath  e.g. './models/rec.onnx'
   * @param {string} dictPath   e.g. './models/dict.txt'
   * @param {function} onProgress  callback(fraction 0-1)
   */
  async function load(modelPath, dictPath, onProgress) {
    const dictResp = await fetch(dictPath);
    if (!dictResp.ok) throw new Error(`Cannot load dict: ${dictPath}`);
    const dictText = await dictResp.text();
    const chars    = dictText.split(/\r?\n/).filter(c => c.length > 0);
    charset        = ['[blank]', ...chars, ' '];
    console.log(`[Recognizer] Dict loaded: ${chars.length} chars + blank + space = ${charset.length} classes`);
    if (onProgress) onProgress(0.1);
    session = await _createSession(modelPath);
    if (onProgress) onProgress(1.0);
    console.log('[Recognizer] Loaded from URL  inputs:', session.inputNames, ' outputs:', session.outputNames);
  }

  /**
   * Load the recognition model from an ArrayBuffer (from IndexedDB cache).
   * dict.txt is still fetched over HTTP (it's tiny, 8 KB — not worth caching).
   * @param {ArrayBuffer} arrayBuffer
   * @param {string}      dictPath   e.g. './models/dict.txt'
   */
  async function loadFromBytes(arrayBuffer, dictPath) {
    // Load charset
    const dictResp = await fetch(dictPath);
    if (!dictResp.ok) throw new Error(`Cannot load dict: ${dictPath}`);
    const dictText = await dictResp.text();
    const chars    = dictText.split(/\r?\n/).filter(c => c.length > 0);
    charset        = ['[blank]', ...chars, ' '];   // 438 total
    console.log(`[Recognizer] Dict: ${chars.length} chars + blank + space = ${charset.length} classes`);

    // Create session from bytes
    session = await _createSession(arrayBuffer);
    console.log('[Recognizer] Loaded from bytes  inputs:', session.inputNames, ' outputs:', session.outputNames);
  }

  /**
   * Recognise text within one detected bounding box.
   *
   * @param {cv.Mat}  bgrMat   Full preprocessed BGR image
   * @param {Array}   box      [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  TL→TR→BR→BL
   * @returns {Promise<string>}  Decoded text string (empty string if failed)
   */
  async function recognize(bgrMat, box) {
    // ── 1. Perspective-warp crop (handles rotated / angled boxes) ──────────
    // box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  TL→TR→BR→BL
    const tl = box[0], tr = box[1], br = box[2], bl = box[3];

    // Compute output dimensions from the box geometry
    const cropW = Math.round(
      Math.max(
        Math.hypot(tr[0] - tl[0], tr[1] - tl[1]),  // top edge length
        Math.hypot(br[0] - bl[0], br[1] - bl[1])   // bottom edge length
      )
    );
    const cropH = Math.round(
      Math.max(
        Math.hypot(bl[0] - tl[0], bl[1] - tl[1]),  // left edge length
        Math.hypot(br[0] - tr[0], br[1] - tr[1])   // right edge length
      )
    );

    if (cropW < 2 || cropH < 2) return '';

    // Source quadrilateral (4 corners of the detected box)
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      tl[0], tl[1],
      tr[0], tr[1],
      br[0], br[1],
      bl[0], bl[1],
    ]);

    // Destination: flat upright rectangle
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0,     0,
      cropW, 0,
      cropW, cropH,
      0,     cropH,
    ]);

    const M       = cv.getPerspectiveTransform(srcPts, dstPts);
    const cropMat = new cv.Mat();
    cv.warpPerspective(bgrMat, cropMat, M, new cv.Size(cropW, cropH));
    srcPts.delete(); dstPts.delete(); M.delete();

    // ── 2. Resize to fixed height ──────────────────────────────────────────
    let   warpH  = cropMat.rows;
    let   warpW  = cropMat.cols;

    // If the warped region is taller than wide (e.g. 90° text column), rotate 90° CW
    let workMat = cropMat;
    if (warpH > warpW * 1.5) {
      const rotated = new cv.Mat();
      cv.rotate(cropMat, rotated, cv.ROTATE_90_CLOCKWISE);
      cropMat.delete();
      workMat = rotated;
      // swap dims after rotation
      [warpW, warpH] = [workMat.cols, workMat.rows];
    }

    const aspect  = warpW / warpH;
    const recW    = Math.max(MIN_WIDTH, Math.round(REC_HEIGHT * aspect));
    const resized = new cv.Mat();
    cv.resize(workMat, resized, new cv.Size(recW, REC_HEIGHT));
    workMat.delete();

    // ── 3. Build tensor ────────────────────────────────────────────────────
    const tensor = _bgrToTensor(resized, REC_HEIGHT, recW);
    resized.delete();

    // ── 4 & 5. Inference + decode ──────────────────────────────────────────
    return await _runAndDecode(tensor);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /** Run inference and CTC-decode the output. */
  async function _runAndDecode(tensor) {
    const inputName  = 'x';
    const outputName = 'fetch_name_0';
    const results    = await session.run({ [inputName]: tensor });
    const output     = results[outputName];

    // rec.onnx output: [N, T, 438]  where N=1 (batch), T=seq len, 438=classes
    const seqLen     = output.dims[1];
    const numClasses = output.dims[2];   // always 438

    const data = output.data;   // Float32Array

    // Greedy CTC decode
    let text     = '';
    let prevIdx  = -1;

    for (let t = 0; t < seqLen; t++) {
      // Argmax over class dimension
      let maxVal = -Infinity, maxIdx = 0;
      const offset = t * numClasses;
      for (let c = 0; c < numClasses; c++) {
        if (data[offset + c] > maxVal) {
          maxVal = data[offset + c];
          maxIdx = c;
        }
      }

      // Remove blanks (index 0) and consecutive duplicates
      if (maxIdx !== 0 && maxIdx !== prevIdx) {
        text += charset[maxIdx];   // charset[0]=blank, charset[1..436]=dict, charset[437]=space
      }
      prevIdx = maxIdx;
    }

    return text;
  }

  /**
   * Convert a BGR cv.Mat to a float32 ONNX tensor [1, 3, H, W].
   * Normalisation: (pixel/255 − 0.5) / 0.5
   * OpenCV BGR → model RGB.
   */
  function _bgrToTensor(bgrMat, h, w) {
    const data    = bgrMat.data;   // Uint8Array BGRBGR…
    const float32 = new Float32Array(3 * h * w);
    const rOff    = 0;
    const gOff    = h * w;
    const bOff    = 2 * h * w;

    for (let i = 0; i < h * w; i++) {
      const b = data[i * 3 + 0] / 255.0;
      const g = data[i * 3 + 1] / 255.0;
      const r = data[i * 3 + 2] / 255.0;
      float32[rOff + i] = (r - REC_MEAN) / REC_STD;
      float32[gOff + i] = (g - REC_MEAN) / REC_STD;
      float32[bOff + i] = (b - REC_MEAN) / REC_STD;
    }
    return new ort.Tensor('float32', float32, [1, 3, h, w]);
  }

  /**
   * Fetch a binary resource, reporting download progress.
   * Returns an ArrayBuffer.
   */
  async function _fetchWithProgress(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

    const total  = parseInt(response.headers.get('content-length') || '0', 10);
    const reader = response.body.getReader();
    const chunks = [];
    let loaded   = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (total > 0 && onProgress) onProgress(loaded / total);
    }

    const buffer = new Uint8Array(loaded);
    let offset   = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    return buffer.buffer;
  }

  return { load, loadFromBytes, recognize };
})();
