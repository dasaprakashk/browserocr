/**
 * detector.js
 * Runs the PaddleOCR DB (Differentiable Binarization) text detection model (det_v5.onnx).
 *
 * Pipeline:
 *   1. Resize input to multiple-of-32 dims, longest side ≤ DET_LIMIT
 *   2. Normalise with ImageNet mean/std → CHW float32 tensor
 *   3. Run det_v5.onnx
 *   4. Threshold probability map → binary mask
 *   5. Find contours → compute score → unclip → scale back to original coords
 */

const Detector = (() => {

  // ── Hyper-parameters (matches PaddleOCR defaults) ──────────────────────────
  const DET_LIMIT       = 960;   // max side length fed to the model
  const DB_THRESH       = 0.3;   // probability map binarisation threshold
  const DB_BOX_THRESH   = 0.5;   // mean-score threshold to keep a box
  const DB_UNCLIP_RATIO = 1.6;   // expansion ratio (higher = larger boxes)
  const MIN_BOX_PX      = 8;     // minimum side length (in map pixels)

  // ImageNet normalisation
  const MEAN = [0.485, 0.456, 0.406];
  const STD  = [0.229, 0.224, 0.225];

  let session = null;


  async function _createSession(modelSource) {
    // WebGPU inference fails for this model due to MaxPool kernel shape computation
    // limitations in onnxruntime-web — WASM is the only reliable EP.
    console.log('[Detector] EP: wasm');
    return ort.InferenceSession.create(modelSource, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Load the detection ONNX model from an ArrayBuffer (from IndexedDB cache).
   * This is the preferred path — no network request on repeated loads.
   * @param {ArrayBuffer} arrayBuffer
   */
  async function loadFromBytes(arrayBuffer) {
    session = await _createSession(arrayBuffer);
    console.log('[Detector] Loaded from bytes  inputs:', session.inputNames, ' outputs:', session.outputNames);
  }

  /**
   * Detect text regions in a BGR cv.Mat (the preprocessed image).
   * Returns an array of axis-aligned boxes in original image coordinates:
   *   [{ box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], score: number }, ...]
   *   Points are ordered: TL → TR → BR → BL
   *
   * @param {cv.Mat} bgrMat  Preprocessed BGR image
   * @returns {Promise<Array>}
   */
  async function detect(bgrMat) {
    const origH = bgrMat.rows;
    const origW = bgrMat.cols;

    // ── 1. Resize to model input size ────────────────────────────────────────
    const scale  = Math.min(DET_LIMIT / Math.max(origH, origW), 1.0);
    const detH   = Math.max(Math.round(origH * scale / 32) * 32, 32);
    const detW   = Math.max(Math.round(origW * scale / 32) * 32, 32);
    const scaleH = origH / detH;
    const scaleW = origW / detW;

    const resized = new cv.Mat();
    cv.resize(bgrMat, resized, new cv.Size(detW, detH));

    // ── 2. BGR → float32 CHW tensor (ImageNet normalised) ───────────────────
    const tensor = _bgrToTensor(resized, detH, detW, MEAN, STD);
    resized.delete();

    // ── 3. Inference ─────────────────────────────────────────────────────────
    const inputName  = 'x';
    const outputName = 'fetch_name_0';
    const results    = await session.run({ [inputName]: tensor });
    const output     = results[outputName];

    // output.dims: [1, 1, mapH, mapW]
    const mapH    = output.dims[2];
    const mapW    = output.dims[3];
    const probMap = output.data;   // Float32Array, values in [0, 1]

    console.log(
      `[Detector] prob map ${mapH}×${mapW}`,
      'min:', Math.min(...probMap.slice(0, 100)).toFixed(3),
      'max:', Math.max(...probMap.slice(0, 100)).toFixed(3)
    );

    // ── 4 & 5. Post-process ──────────────────────────────────────────────────
    const boxes = _postProcess(probMap, mapH, mapW, scaleH, scaleW, origH, origW);
    return boxes;
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /**
   * Convert a BGR cv.Mat to a normalised float32 ONNX tensor [1,3,H,W].
   * OpenCV stores pixels as BGR; the model expects RGB.
   */
  function _bgrToTensor(bgrMat, h, w, mean, std) {
    const data    = bgrMat.data;   // Uint8Array, layout BGRBGR…
    const float32 = new Float32Array(3 * h * w);
    const rOff    = 0;
    const gOff    = h * w;
    const bOff    = 2 * h * w;

    for (let i = 0; i < h * w; i++) {
      const b = data[i * 3 + 0] / 255.0;
      const g = data[i * 3 + 1] / 255.0;
      const r = data[i * 3 + 2] / 255.0;
      float32[rOff + i] = (r - mean[0]) / std[0];
      float32[gOff + i] = (g - mean[1]) / std[1];
      float32[bOff + i] = (b - mean[2]) / std[2];
    }
    return new ort.Tensor('float32', float32, [1, 3, h, w]);
  }

  /**
   * Threshold the probability map, find contours, unclip boxes, filter by score.
   */
  function _postProcess(probMap, mapH, mapW, scaleH, scaleW, origH, origW) {
    // Build binary mask via OpenCV
    const probMat = cv.matFromArray(mapH, mapW, cv.CV_32F, probMap);
    const binMat  = new cv.Mat();
    cv.threshold(probMat, binMat, DB_THRESH, 255, cv.THRESH_BINARY);
    probMat.delete();

    const binU8 = new cv.Mat();
    binMat.convertTo(binU8, cv.CV_8U);
    binMat.delete();

    // Find contours
    const contours  = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(binU8, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    hierarchy.delete();
    binU8.delete();

    const boxes = [];

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area    = cv.contourArea(contour);

      // Skip tiny noise
      if (area < MIN_BOX_PX * MIN_BOX_PX) {
        contour.delete();
        continue;
      }

      // Axis-aligned bounding rect in map space
      const br = cv.boundingRect(contour);
      contour.delete();

      // Compute box score = mean probability within the box
      const score = _meanScore(
        probMap, mapW,
        br.x, br.y, br.x + br.width, br.y + br.height
      );
      if (score < DB_BOX_THRESH) continue;

      // Unclip (expand) the box
      const rectArea = br.width * br.height;
      const perim    = 2 * (br.width + br.height);
      const dist     = Math.round((rectArea * DB_UNCLIP_RATIO) / perim);

      const mx1 = Math.max(0,    br.x - dist);
      const my1 = Math.max(0,    br.y - dist);
      const mx2 = Math.min(mapW, br.x + br.width  + dist);
      const my2 = Math.min(mapH, br.y + br.height + dist);

      // Skip if too small after clip check
      if ((mx2 - mx1) < MIN_BOX_PX || (my2 - my1) < MIN_BOX_PX) continue;

      // Scale back to original image coordinates
      const ox1 = Math.max(0,      Math.round(mx1 * scaleW));
      const oy1 = Math.max(0,      Math.round(my1 * scaleH));
      const ox2 = Math.min(origW,  Math.round(mx2 * scaleW));
      const oy2 = Math.min(origH,  Math.round(my2 * scaleH));

      boxes.push({
        box: [
          [ox1, oy1],  // TL
          [ox2, oy1],  // TR
          [ox2, oy2],  // BR
          [ox1, oy2],  // BL
        ],
        score,
      });
    }

    contours.delete();
    console.log(`[Detector] Found ${boxes.length} box(es)`);
    return boxes;
  }

  /** Mean of probMap values in the rectangle [x1,y1,x2,y2] (clamped). */
  function _meanScore(probMap, mapW, x1, y1, x2, y2) {
    let sum = 0, n = 0;
    for (let r = y1; r < y2; r++) {
      for (let c = x1; c < x2; c++) {
        sum += probMap[r * mapW + c];
        n++;
      }
    }
    return n > 0 ? sum / n : 0;
  }

  return { loadFromBytes, detect };
})();
