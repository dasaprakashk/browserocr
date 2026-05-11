/**
 * app.js
 * Main orchestrator: wires together the UI, model loading, and inference pipeline.
 */

// ── IndexedDB model cache ─────────────────────────────────────────────────────
const DB_NAME    = 'ocr-models';
const DB_VERSION = 1;
const DB_STORE   = 'models';

/** Open (or create) the IndexedDB database. Returns a Promise<IDBDatabase>. */
function _openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        db.createObjectStore(DB_STORE);
        console.log('[IDB] Object store created');
      }
    };
    req.onsuccess = e => resolve(e.target.result);
    req.onerror   = e => reject(e.target.error);
  });
}

/** Read an ArrayBuffer from IndexedDB. Returns null if key not found. */
async function _idbGet(key) {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(DB_STORE, 'readonly');
    const req = tx.objectStore(DB_STORE).get(key);
    req.onsuccess = e => resolve(e.target.result ?? null);
    req.onerror   = e => reject(e.target.error);
  });
}

/** Write an ArrayBuffer to IndexedDB under the given key. */
async function _idbPut(key, buffer) {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).put(buffer, key);
    tx.oncomplete = resolve;
    tx.onerror    = e => reject(e.target.error);
  });
}

/**
 * Fetch a model file with progress reporting, save to IndexedDB, and return
 * its ArrayBuffer. On subsequent calls, returns straight from IndexedDB
 * (no network request, instant).
 *
 * @param {string}   url        e.g. './models/det.onnx'
 * @param {string}   cacheKey   IDB key, e.g. 'det.onnx'
 * @param {function} onProgress callback(fraction 0-1)
 * @returns {Promise<ArrayBuffer>}
 */
async function _loadModelBytes(url, cacheKey, onProgress) {
  // ── 1. Try IndexedDB cache first ────────────────────────────────────────
  const cached = await _idbGet(cacheKey);
  if (cached) {
    console.log(`[IDB] Cache hit: ${cacheKey} (${(cached.byteLength / 1e6).toFixed(1)} MB)`);
    if (onProgress) onProgress(1);
    return cached;
  }

  // ── 2. Not cached — stream from local server with progress ────────────────
  console.log(`[IDB] Cache miss: ${cacheKey} — fetching from server...`);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

  const total   = parseInt(response.headers.get('content-length') || '0', 10);
  const reader  = response.body.getReader();
  const chunks  = [];
  let   loaded  = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total > 0 && onProgress) onProgress(loaded / total);
  }

  // Concatenate chunks into one ArrayBuffer
  const buffer = new Uint8Array(loaded);
  let offset   = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }
  const arrayBuffer = buffer.buffer;

  // ── 3. Persist to IndexedDB for next page load ────────────────────────────
  await _idbPut(cacheKey, arrayBuffer);
  console.log(`[IDB] Saved to cache: ${cacheKey}`);

  return arrayBuffer;
}

// ── Globals ───────────────────────────────────────────────────────────────────
let _modelsLoaded = false;

/**
 * Pipeline: det_v5.onnx (4.7 MB, WASM) + rec.onnx (7.8 MB, WebGPU) + dict.txt
 * Single model set used for all flows — live camera, upload, and manual crop.
 */

// ── Pipeline config ──────────────────────────────────────────────────────────
const APPLY_PREPROCESSING = true;   // set false to skip CLAHE+sharpen (raw input to det+rec)

// ── Camera state ─────────────────────────────────────────────────────────────────────
let _cameraStream  = null;   // MediaStream from getUserMedia
let _liveDetector  = null;   // LiveDetectorDet instance

// ── DOM refs (populated after DOMContentLoaded) ───────────────────────────────
let dropZone, fileInput, uploadBtn;
let statusText, progressBar, progressFill;
let resultsContainer, outputCanvas, dotResultEl, allTextsEl;
let preprocessCanvas, originalCanvas, logBody, logCount;

/**
 * Append a timestamped entry to the on-screen inference log panel.
 * @param {'info'|'ok'|'warn'|'error'|'model'|'preproc'|'detect'|'recog'|'dot'|'idb'} tag
 * @param {string} msg
 */
function _log(tag, msg) {
  // Always mirror to browser console too
  const method = tag === 'error' ? 'error' : tag === 'warn' ? 'warn' : 'log';
  console[method](`[${tag.toUpperCase()}] ${msg}`);

  if (!logBody) return;   // panel not mounted yet

  const now    = new Date();
  const ts     = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}.${String(now.getMilliseconds()).padStart(3,'0')}`;

  const row    = document.createElement('div');
  row.className = 'log-entry';
  row.innerHTML = `
    <span class="log-time">${ts}</span>
    <span class="log-tag tag-${tag}">[${tag.toUpperCase()}]</span>
    <span class="log-msg">${_esc(String(msg))}</span>`;
  logBody.appendChild(row);
  logBody.scrollTop = logBody.scrollHeight;   // auto-scroll to latest

  // Update badge count
  const n = logBody.querySelectorAll('.log-entry').length;
  if (logCount) logCount.textContent = n;
}

// ── Entry point ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Grab DOM elements
  dropZone         = document.getElementById('drop-zone');
  fileInput        = document.getElementById('file-input');
  uploadBtn        = document.getElementById('upload-btn');
  statusText       = document.getElementById('status-text');
  progressBar      = document.getElementById('progress-bar');
  progressFill     = document.getElementById('progress-fill');
  resultsContainer = document.getElementById('results-container');
  outputCanvas     = document.getElementById('output-canvas');
  dotResultEl      = document.getElementById('dot-result');
  allTextsEl       = document.getElementById('all-texts');
  preprocessCanvas = document.getElementById('preprocess-canvas');
  originalCanvas   = document.getElementById('original-canvas');
  logBody          = document.getElementById('log-body');
  logCount         = document.getElementById('log-count');

  // Wire up file input
  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  // Wire up Take Photo button
  const takePhotoBtn = document.getElementById('take-photo-btn');
  if (takePhotoBtn) takePhotoBtn.addEventListener('click', _openCamera);

  // Drag & drop
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  // WASM paths for onnxruntime-web 1.22.0 (must be set before creating sessions)
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  ort.env.wasm.numThreads = 1;   // Python http.server lacks COOP/COEP headers needed for SharedArrayBuffer

  _maybeInit();
});

// ── Initialise models (when both DOM + OpenCV are ready) ──────────────────────
async function _maybeInit() {
  if (!document.getElementById('status-text')) return;  // DOM not ready yet
  if (!window._cvReady) return;                         // OpenCV WASM not ready yet
  if (_modelsLoaded) return;

  try {
    _log('model', 'Loading det_v5.onnx + rec.onnx + dict.txt…');

    // ── 1. det_v5.onnx — used for all detection (live + final OCR) ─────────────
    _setStatus('Loading det_v5.onnx — checking cache…', 0.02);
    _log('idb', 'Checking IndexedDB for det_v5…');
    const detBytes = await _loadModelBytes(
      './models/det_v5.onnx', 'det_v5',
      frac => {
        _setStatus(
          frac === 1
            ? 'det_v5.onnx ✅ loaded from cache'
            : `Downloading det_v5.onnx… ${Math.round(frac * 100)}%`,
          0.02 + frac * 0.28
        );
        if (frac === 1) _log('idb', 'det_v5: cache hit — loaded instantly');
        else if (frac < 0.02) _log('idb', 'det_v5: cache miss — downloading (4.7 MB)…');
      }
    );
    _log('model', `det_v5.onnx ready: ${(detBytes.byteLength / 1e6).toFixed(1)} MB`);
    _setStatus('Initialising detection model…', 0.31);
    const t0det = performance.now();
    await Detector.loadFromBytes(detBytes);
    _log('ok', `det_v5.onnx session ready in ${(performance.now() - t0det).toFixed(0)} ms`);

    // ── 2. rec.onnx — WebGPU accelerated recognition ───────────────────────
    _setStatus('Loading rec.onnx — checking cache…', 0.35);
    _log('idb', 'Checking IndexedDB for rec_default…');
    const recBytes = await _loadModelBytes(
      './models/rec.onnx', 'rec_default',
      frac => {
        _setStatus(
          frac === 1
            ? 'rec.onnx ✅ loaded from cache'
            : `Downloading rec.onnx… ${Math.round(frac * 100)}%`,
          0.35 + frac * 0.63
        );
        if (frac === 1) _log('idb', 'rec_default: cache hit — loaded from IndexedDB instantly');
        else if (frac < 0.02) _log('idb', 'rec_default: cache miss — downloading (7.8 MB)…');
      }
    );
    _log('model', `rec.onnx ready: ${(recBytes.byteLength / 1e6).toFixed(1)} MB`);
    _setStatus('Initialising recognition model…', 0.99);
    _log('model', 'Creating ort.InferenceSession for rec.onnx (WebGPU → WASM fallback) + dict.txt…');
    const t0rec = performance.now();
    await Recognizer.loadFromBytes(recBytes, './models/dict.txt');
    _log('ok', `rec.onnx session ready in ${(performance.now() - t0rec).toFixed(0)} ms`);

    _modelsLoaded = true;
    _setStatus('✅ Models ready — upload an image to start', 1);
    progressBar.style.display = 'none';
    dropZone.classList.add('ready');

  } catch (err) {
    _setStatus(`❌ Model load failed: ${err.message}`, 0);
    console.error('[App] Model load error:', err);
  }
}

// ── Camera workflow ──────────────────────────────────────────────────────────

/**
 * Open the camera modal and start live detection using LiveDetectorDet (det.onnx).
 */
async function _openCamera() {
  const modal         = document.getElementById('camera-modal');
  const videoEl       = document.getElementById('camera-video');
  const overlayCanvas = document.getElementById('camera-overlay');
  const statusEl      = document.getElementById('camera-status');

  modal.classList.add('active');
  statusEl.textContent = 'Requesting camera access…';

  if (!_modelsLoaded) {
    statusEl.textContent = '⚠️ Models still loading — please wait for models to finish before using the camera.';
    _log('warn', 'Camera opened before models finished loading');
    return;
  }

  try {
    _cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode:  { ideal: 'environment' },
        width:       { ideal: 1280 },
        height:      { ideal: 720 },
        aspectRatio: { ideal: 16/9 },   // widest available on device
      },
      audio: false,
    });

    videoEl.srcObject = _cameraStream;
    await videoEl.play();

    videoEl.addEventListener('loadedmetadata', () => {
      // Size canvas to wrapper display dimensions (CSS strip), not full video dims
      const wrap = overlayCanvas.parentElement;
      overlayCanvas.width  = wrap.offsetWidth  || 400;
      overlayCanvas.height = wrap.offsetHeight || 125;
      _log('info', `Camera stream: ${videoEl.videoWidth}\u00d7${videoEl.videoHeight} — display strip: ${overlayCanvas.width}\u00d7${overlayCanvas.height}`);
    }, { once: true });

    _liveDetector = LiveDetectorDet;
    _liveDetector.start(videoEl, overlayCanvas, statusEl, _onLiveCapture);
    _log('info', 'Camera started — guide-box mode');

  } catch (err) {
    statusEl.textContent = `❌ Camera error: ${err.message}`;
    _log('error', `Camera access failed: ${err.message}`);
    console.error('[Camera]', err);
  }
}

/** Stop live detection, release camera, close modal. */
function _closeCamera() {
  if (_liveDetector) {
    _liveDetector.stop();
    _liveDetector = null;
  }
  if (_cameraStream) {
    _cameraStream.getTracks().forEach(t => t.stop());
    _cameraStream = null;
  }
  const videoEl = document.getElementById('camera-video');
  if (videoEl) videoEl.srcObject = null;
  document.getElementById('camera-modal').classList.remove('active');
  _log('info', 'Camera closed');
}

/** Manual capture button — grab current frame and pass to crop tool. */
function _manualCapture() {
  if (_liveDetector) {
    _liveDetector.captureNow();
  }
}

/**
 * Called when a stable frame is captured (auto or manual).
 * Stops camera, closes modal, opens Cropper pre-positioned to detected box.
 * @param {HTMLCanvasElement} canvas    Full-resolution captured frame.
 * @param {{x,y,width,height}|null} cropRect  Detected box in full-res coords, or null.
 */
function _onLiveCapture(canvas, cropRect) {
  // Tear down camera immediately
  if (_liveDetector) { _liveDetector.stop(); _liveDetector = null; }
  if (_cameraStream) {
    _cameraStream.getTracks().forEach(t => t.stop());
    _cameraStream = null;
  }
  const videoEl = document.getElementById('camera-video');
  if (videoEl) videoEl.srcObject = null;
  document.getElementById('camera-modal').classList.remove('active');

  // Always use the detected cropRect for pre-positioning, regardless of auto-capture mode
  const effectiveCropRect = cropRect ?? null;

  _log('info', `Live capture: ${canvas.width}×${canvas.height} px — opening crop tool${
    effectiveCropRect ? '' : ' (no detection box — full-frame 90% crop)'
  }`);

  // Feed captured frame into Cropper, passing detected box for pre-positioning
  const img  = new Image();
  img.onload = () => _showCropModal(img, effectiveCropRect);
  img.src    = canvas.toDataURL('image/jpeg', 0.92);
}

// ── Crop modal state (Cropper.js) ───────────────────────────────────────────
let _cropper = null;
let _cropState = {
  img: null,
};

function _showCropModal(img, cropRect) {
  const modal = document.getElementById('crop-modal');
  const cropImg = document.getElementById('crop-img');

  // Set image source
  cropImg.src = img.src;
  _cropState.img = img;

  // Initialize Cropper.js if not already done
  if (_cropper) {
    _cropper.destroy();
  }

  _cropper = new Cropper(cropImg, {
    guides: true,
    background: true,
    responsive: true,
    restore: true,
    center: true,
    highlight: true,
    cropBoxMovable: true,
    cropBoxResizable: true,
    toggleDragModeOnDblclick: true,
    autoCropArea: cropRect ? 1 : 0.8,   // full canvas first; we'll set box manually below
    ready() {
      if (!cropRect) return;
      // Map full-res image coords → Cropper canvas coords
      const canvasData = _cropper.getCanvasData();
      const scaleX = canvasData.width  / img.naturalWidth;
      const scaleY = canvasData.height / img.naturalHeight;
      _cropper.setCropBoxData({
        left:   canvasData.left + cropRect.x      * scaleX,
        top:    canvasData.top  + cropRect.y      * scaleY,
        width:  cropRect.width  * scaleX,
        height: cropRect.height * scaleY,
      });
      _log('info', `Crop box pre-set to detected region: ${Math.round(cropRect.x)},${Math.round(cropRect.y)} ${Math.round(cropRect.width)}×${Math.round(cropRect.height)} px`);
    },
  });

  modal.classList.add('active');
  _log('info', 'Crop tool ready — adjust region and click "Crop & Process"');
}

function _hideCropModal() {
  const modal = document.getElementById('crop-modal');
  modal.classList.remove('active');
  if (_cropper) {
    _cropper.destroy();
    _cropper = null;
  }
  _cropState.img = null;
}

// ── Handle image file ─────────────────────────────────────────────────────────
async function handleFile(file) {
  if (!_modelsLoaded) {
    alert('Models are still loading — please wait.');
    return;
  }
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file.');
    return;
  }

  // Decode image
  const img = await _loadImage(file);
  _log('info', `Image decoded: ${img.naturalWidth} × ${img.naturalHeight} px — showing crop tool`);

  // Show crop modal instead of immediately processing
  _showCropModal(img);
}

// Process the cropped region through OCR pipeline
async function _cropAndProcess() {
  if (!_cropper) {
    alert('Cropper not initialized.');
    return;
  }

  try {
    // Get canvas from Cropper.js
    const canvas = _cropper.getCroppedCanvas();
    if (!canvas) {
      alert('Failed to extract cropped image.');
      return;
    }

    _hideCropModal();
    _log('info', `Cropped region: ${canvas.width}×${canvas.height} px`);

    // Now run the normal pipeline on cropped image
    await _processCroppedImage(canvas, 'crop');
  } catch (err) {
    _log('error', `Crop extraction failed: ${err.message}`);
    console.error('[Crop]', err);
  }
}

function _cancelCrop() {
  _hideCropModal();
  _log('warn', 'Crop cancelled by user');
}

async function _processCroppedImage(srcCanvas, fileName) {
  _setStatus('Processing image…', 0);
  progressBar.style.display = '';
  resultsContainer.hidden = false;
  dotResultEl.innerHTML   = '';
  allTextsEl.innerHTML    = '';
  if (logBody)  { logBody.innerHTML = ''; }
  if (logCount) { logCount.textContent = '0'; }
  const ocrAreaEl = document.getElementById('ocr-text-area');
  if (ocrAreaEl) ocrAreaEl.innerHTML = '';

  try {
    _log('info', `Processing cropped image: size ${srcCanvas.width}×${srcCanvas.height} px`);

    // Draw to left panel (original cropped)
    originalCanvas.width  = srcCanvas.width;
    originalCanvas.height = srcCanvas.height;
    originalCanvas.getContext('2d').drawImage(srcCanvas, 0, 0);

    // ── Preprocess ────────────────────────────────────────────────────────
    _setStatus('Preprocessing…', 0.05);
    _log('preproc', `Sending ${srcCanvas.width}×${srcCanvas.height} px image to preprocessor…`);
    const t0pre = performance.now();
    let original, preprocessed;
    if (APPLY_PREPROCESSING) {
      ({ original, preprocessed } = Preprocessor.process(srcCanvas));
      _log('preproc', `Resize → CLAHE (clip=2.5, 8×8 tiles) → unsharp mask (σ=1.0) → output: ${preprocessed.cols}×${preprocessed.rows} px`);
    } else {
      preprocessed = cv.imread(srcCanvas);
      const tmp = new cv.Mat();
      cv.cvtColor(preprocessed, tmp, cv.COLOR_RGBA2BGR);
      preprocessed.delete();
      preprocessed = tmp;
      original = preprocessed.clone();
      _log('preproc', `[PREPROCESSING DISABLED] Raw image passed directly — ${preprocessed.cols}×${preprocessed.rows} px`);
    }
    _log('ok', `Preprocessing done in ${(performance.now() - t0pre).toFixed(0)} ms`);
    Preprocessor.renderToCanvas(preprocessed, preprocessCanvas);

    // Update the preprocess panel label
    const preprocLabel = document.getElementById('preprocess-label');
    if (preprocLabel) {
      preprocLabel.textContent = APPLY_PREPROCESSING
        ? 'Preprocessed (CLAHE + sharpen)'
        : 'Preprocessed (raw image)';
    }

    // ── Detection ─────────────────────────────────────────────────────────
    _setStatus('Detecting text regions…', 0.15);
    _log('detect', `Preprocessed mat → det_v5.onnx (resize ≤960px, ×32, ImageNet norm, CHW float32)…`);
    const t0det = performance.now();
    let detections = await Detector.detect(preprocessed);
    detections = _sortDetectionsReadingOrder(detections);
    const tDet = (performance.now() - t0det).toFixed(0);
    _log(detections.length > 0 ? 'ok' : 'warn',
      `Detection done in ${tDet} ms — ${detections.length} region(s) found`);
    _log('detect', 'Detections reordered to reading order (top→bottom, left→right)');
    detections.forEach((d, i) => {
      const b = d.box;
      _log('detect', `  [${i+1}] TL(${b[0][0]},${b[0][1]}) BR(${b[2][0]},${b[2][1]})  score=${d.score.toFixed(3)}`);
    });
    _setStatus(`Found ${detections.length} region(s) — recognising…`, 0.4);

    // ── Recognition ───────────────────────────────────────────────────────
    _log('recog', `Starting recognition for ${detections.length} region(s)…`);
    const results = [];   // { text, score, box }
    for (let i = 0; i < detections.length; i++) {
      _setStatus(`Recognising ${i+1} / ${detections.length}…`, 0.4 + (i / detections.length) * 0.5);
      const b     = detections[i].box;
      const cropW = b[2][0] - b[0][0];
      const cropH = b[2][1] - b[0][1];
      _log('recog', `[${i+1}] crop ${cropW}×${cropH} px → h=48 resize → rec.onnx → greedy CTC…`);
      const t0r  = performance.now();
      const text = await Recognizer.recognize(preprocessed, b);
      _log(text.trim() ? 'recog' : 'warn',
        `[${i+1}] "${text.trim() || '(empty)'}"  conf=${detections[i].score.toFixed(2)}  ${(performance.now()-t0r).toFixed(0)} ms`);
      results.push({ text, score: detections[i].score, box: b });
    }

    // ── Render ────────────────────────────────────────────────────────────
    _renderResults(results, preprocessed);

    // ── Cleanup ───────────────────────────────────────────────────────────
    original.delete();
    preprocessed.delete();

    const nText = results.filter(r => r.text.trim()).length;
    _setStatus('✅ Done!', 1);
    _log('ok', `Pipeline complete — ${nText} text region(s) extracted out of ${results.length}`);
    setTimeout(() => { progressBar.style.display = 'none'; }, 1500);

  } catch (err) {
    _setStatus(`❌ Error: ${err.message}`, 0);
    _log('error', `Pipeline failed: ${err.message}`);
    console.error('[App] Pipeline error:', err);
  }
}


// ── Render bboxes + OCR text output ──────────────────────────────────────────
function _renderResults(results, preprocessedMat) {
  // ── Detection canvas ─────────────────────────────────────────────────────
  const canvas = outputCanvas;
  canvas.width  = preprocessedMat.cols;
  canvas.height = preprocessedMat.rows;
  const ctx = canvas.getContext('2d');

  const rgbaMat = new cv.Mat();
  cv.cvtColor(preprocessedMat, rgbaMat, cv.COLOR_BGR2RGBA);
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(rgbaMat.data), preprocessedMat.cols, preprocessedMat.rows),
    0, 0
  );
  rgbaMat.delete();

  results.forEach(({ text, score, box }, i) => {
    const hasText = text.trim().length > 0;
    ctx.strokeStyle = hasText ? '#00E5FF' : '#444';
    ctx.lineWidth   = hasText ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(box[0][0], box[0][1]);
    for (let k = 1; k < 4; k++) ctx.lineTo(box[k][0], box[k][1]);
    ctx.closePath();
    ctx.stroke();

    if (hasText) {
      const label = text.length > 35 ? text.slice(0, 33) + '…' : text;
      ctx.font      = 'bold 12px monospace';
      ctx.fillStyle = '#00E5FF';
      ctx.fillText(`${i+1}: ${label}`, box[0][0] + 2, Math.max(box[0][1] - 5, 13));
    }
  });

  // ── OCR output card ───────────────────────────────────────────────────────
  const nonEmpty = results.filter(r => r.text.trim());
  const ocrMeta  = document.getElementById('ocr-meta');
  const ocrArea  = document.getElementById('ocr-text-area');

  if (ocrMeta) ocrMeta.textContent = `${nonEmpty.length} of ${results.length} regions with text`;

  if (ocrArea) {
    if (nonEmpty.length > 0) {
      ocrArea.innerHTML = nonEmpty.map((r, i) =>
        `<div class="ocr-line">
          <span class="ocr-idx">${i + 1}</span>
          <span class="ocr-txt">${_esc(r.text)}</span>
          <span class="ocr-conf">${Math.round(r.score * 100)}%</span>
        </div>`
      ).join('');
    } else {
      ocrArea.innerHTML = '<span class="ocr-txt is-empty">No readable text detected — try a clearer image or better lighting.</span>';
    }
  }

  // Store plain text for copy button
  window._lastOcrText = nonEmpty.map(r => r.text).join('\n');

  resultsContainer.hidden = false;
  resultsContainer.scrollIntoView({ behavior: 'smooth' });
}

/** Copy all OCR text to clipboard — called by the Copy button. */
function _copyOCR() {
  if (!window._lastOcrText) return;
  navigator.clipboard.writeText(window._lastOcrText).then(() => {
    const btn = document.getElementById('copy-btn');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy all'; }, 2000); }
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function _boxMetrics(box) {
  const x1 = Math.min(box[0][0], box[1][0], box[2][0], box[3][0]);
  const y1 = Math.min(box[0][1], box[1][1], box[2][1], box[3][1]);
  const x2 = Math.max(box[0][0], box[1][0], box[2][0], box[3][0]);
  const y2 = Math.max(box[0][1], box[1][1], box[2][1], box[3][1]);
  return {
    x1,
    y1,
    x2,
    y2,
    cx: (x1 + x2) * 0.5,
    cy: (y1 + y2) * 0.5,
    h: Math.max(1, y2 - y1),
  };
}

function _sortDetectionsReadingOrder(detections) {
  if (!detections || detections.length <= 1) return detections || [];

  const items = detections.map(det => ({ det, m: _boxMetrics(det.box) }));
  const heights = items.map(item => item.m.h).sort((a, b) => a - b);
  const medianH = heights[Math.floor(heights.length / 2)] || 20;
  const rowTolerance = Math.max(10, medianH * 0.6);

  items.sort((a, b) => {
    if (a.m.cy !== b.m.cy) return a.m.cy - b.m.cy;
    return a.m.cx - b.m.cx;
  });

  const rows = [];
  for (const item of items) {
    let bestRow = null;
    let bestDelta = Infinity;

    for (const row of rows) {
      const delta = Math.abs(row.cy - item.m.cy);
      if (delta <= rowTolerance && delta < bestDelta) {
        bestDelta = delta;
        bestRow = row;
      }
    }

    if (!bestRow) {
      rows.push({ cy: item.m.cy, items: [item] });
      continue;
    }

    bestRow.items.push(item);
    bestRow.cy = bestRow.items.reduce((sum, current) => sum + current.m.cy, 0) / bestRow.items.length;
  }

  rows.sort((a, b) => a.cy - b.cy);
  for (const row of rows) {
    row.items.sort((a, b) => {
      if (a.m.cx !== b.m.cx) return a.m.cx - b.m.cx;
      return a.m.cy - b.m.cy;
    });
  }

  return rows.flatMap(row => row.items.map(item => item.det));
}

function _setStatus(msg, frac) {
  if (statusText)   statusText.textContent    = msg;
  if (progressFill) progressFill.style.width  = `${Math.round(frac * 100)}%`;
  if (progressBar)  progressBar.style.display = '';
}

function _loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image'));
    img.src     = URL.createObjectURL(file);
  });
}

function _esc(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Global exports (called from HTML event handlers) ─────────────────────────
window.handleFile           = handleFile;
window._copyOCR             = _copyOCR;
window._cropAndProcess      = _cropAndProcess;
window._cancelCrop          = _cancelCrop;
window._openCamera          = _openCamera;
window._closeCamera         = _closeCamera;
window._manualCapture       = _manualCapture;
