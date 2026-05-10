/**
 * live-detector.js
 *
 * LiveDetectorDet — camera hover / auto-capture using det.onnx.
 *
 * Runs the PaddleOCR DB detector on a downscaled video frame every 500 ms.
 * When the same text region stays stable for 3 consecutive hits (~1.5 s),
 * a full-resolution frame is captured and passed to onCapture().
 *
 * Interface:
 *   LiveDetectorDet.start(videoEl, overlayCanvas, statusEl, onCapture)
 *   LiveDetectorDet.stop()
 *   LiveDetectorDet.captureNow()
 */

const LiveDetectorDet = (() => {

  // ── Tuning ────────────────────────────────────────────────────────────────
  const SAMPLE_INTERVAL_MS  = 100;   // ms between det.onnx runs
  const DOWNSCALE_MAX       = 640;   // longest side fed to model (keep light)
  const STABLE_HITS_NEEDED  = 3;     // N consecutive STABLE frames required → capture (~1 s)
  const STABILITY_PX        = 20;    // max union-box centre drift (px in downscaled space) to count as stable

  // ── State ─────────────────────────────────────────────────────────────────
  let _running         = false;
  let _timer           = null;
  let _stableHits      = 0;      // frames in a row where box centre barely moved
  let _prevUnionSmall  = null;   // union rect in downscaled space from previous frame {cx,cy,x,y,w,h}
  let _videoEl         = null;
  let _overlayCanvas   = null;
  let _statusEl        = null;
  let _onCapture       = null;
  let _busy            = false;  // prevent overlapping async ticks
  let _lastDetBox      = null;   // union rect of all boxes in full-res coords {x,y,width,height}
  let _errorCount      = 0;      // consecutive tick errors — bail out after threshold
  let _autoCapture     = true;   // mirrors the UI toggle; when false, only manual capture fires

  // ── Helpers ───────────────────────────────────────────────────────────────

  /** Compute union bounding rect of all detections in downscaled model space. */
  function _unionRectSmall(detections) {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    detections.forEach(({ box: b }) => {
      [b[0][0], b[1][0], b[2][0], b[3][0]].forEach(x => { x1 = Math.min(x1, x); x2 = Math.max(x2, x); });
      [b[0][1], b[1][1], b[2][1], b[3][1]].forEach(y => { y1 = Math.min(y1, y); y2 = Math.max(y2, y); });
    });
    return { x: x1, y: y1, w: x2 - x1, h: y2 - y1, cx: (x1 + x2) / 2, cy: (y1 + y2) / 2 };
  }

  /** Scale a small-space union rect to full-res video coordinates. */
  function _unionRect(detections, sw, sh) {
    const vw  = _videoEl.videoWidth;
    const vh  = _videoEl.videoHeight;
    const upx = vw / sw;
    const upy = vh / sh;
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    detections.forEach(({ box: b }) => {
      [b[0][0], b[1][0], b[2][0], b[3][0]].forEach(x => { x1 = Math.min(x1, x * upx); x2 = Math.max(x2, x * upx); });
      [b[0][1], b[1][1], b[2][1], b[3][1]].forEach(y => { y1 = Math.min(y1, y * upy); y2 = Math.max(y2, y * upy); });
    });
    return {
      x:      Math.max(0,  x1),
      y:      Math.max(0,  y1),
      width:  Math.min(vw, x2) - Math.max(0, x1),
      height: Math.min(vh, y2) - Math.max(0, y1),
    };
  }

  function _grabFrame(maxSide) {
    const vw    = _videoEl.videoWidth;
    const vh    = _videoEl.videoHeight;
    if (!vw || !vh) return null;
    const scale = Math.min(1, maxSide / Math.max(vw, vh));
    const sw    = Math.round(vw * scale);
    const sh    = Math.round(vh * scale);
    const cnv   = document.createElement('canvas');
    cnv.width   = sw;
    cnv.height  = sh;
    cnv.getContext('2d').drawImage(_videoEl, 0, 0, sw, sh);
    return { canvas: cnv, scale, vw, vh };
  }

  function _autoCaptureFrame() {
    if (!_onCapture || !_videoEl) return;
    const cap   = document.createElement('canvas');
    cap.width   = _videoEl.videoWidth;
    cap.height  = _videoEl.videoHeight;
    cap.getContext('2d').drawImage(_videoEl, 0, 0);
    _onCapture(cap, _lastDetBox);   // pass detected box rect for crop pre-positioning
  }

  async function _tick() {
    if (!_running || _busy) return;
    _busy = true;

    const frame = _grabFrame(DOWNSCALE_MAX);
    if (!frame) { _busy = false; return; }

    const { canvas: tmpCanvas } = frame;
    const sw = tmpCanvas.width;
    const sh = tmpCanvas.height;

    let originalMat     = null;
    let preprocessedMat = null;

    try {
      const { original, preprocessed } = Preprocessor.process(tmpCanvas);
      originalMat     = original;
      preprocessedMat = preprocessed;

      const detections = await Detector.detect(preprocessedMat);

      const ctx   = _overlayCanvas.getContext('2d');
      const dispW = _overlayCanvas.width;
      const dispH = _overlayCanvas.height;
      ctx.clearRect(0, 0, dispW, dispH);

      if (detections.length === 0) {
        // Lost sight of text — full reset
        _stableHits     = 0;
        _prevUnionSmall = null;
        if (_statusEl) _statusEl.textContent = 'Scanning… point camera at text';
        _busy = false;
        return;
      }

      // ── Stability check ────────────────────────────────────────────────
      const smallRect = _unionRectSmall(detections);
      let isStable = false;
      if (_prevUnionSmall) {
        const drift = Math.sqrt(
          Math.pow(smallRect.cx - _prevUnionSmall.cx, 2) +
          Math.pow(smallRect.cy - _prevUnionSmall.cy, 2)
        );
        isStable = drift <= STABILITY_PX;
      }
      _prevUnionSmall = smallRect;

      if (isStable) {
        _stableHits++;
      } else {
        _stableHits = 1;   // reset but stay at 1 so next frame only needs 1 more stable hit to progress
      }

      // ── Draw boxes — colour shifts red→yellow→green with stability ─────
      const sx = dispW / sw;
      const sy = dispH / sh;
      const prog = Math.min(_stableHits / STABLE_HITS_NEEDED, 1);
      ctx.lineWidth = 2.5;
      detections.forEach((det, i) => {
        ctx.strokeStyle = i === 0
          ? `hsl(${Math.round(prog * 120)}, 100%, 55%)`   // red → green
          : `rgba(0,229,255,0.45)`;
        ctx.beginPath();
        const b = det.box;
        ctx.moveTo(b[0][0] * sx, b[0][1] * sy);
        for (let k = 1; k < 4; k++) ctx.lineTo(b[k][0] * sx, b[k][1] * sy);
        ctx.closePath();
        ctx.stroke();
      });

      // ── Always track latest det box (used by manual Capture Now) ────────
      _lastDetBox = _unionRect(detections, sw, sh);

      // ── Status text ───────────────────────────────────────────────────
      const stableBar = '█'.repeat(_stableHits) + '░'.repeat(Math.max(0, STABLE_HITS_NEEDED - _stableHits));
      if (_statusEl) {
        _statusEl.textContent = isStable
          ? `Hold still… ${stableBar} (${_stableHits}/${STABLE_HITS_NEEDED})`
          : `Text found — hold steady… ${stableBar}`;
      }

      // ── Auto-capture when stable long enough ──────────────────────────
      if (_autoCapture && _stableHits >= STABLE_HITS_NEEDED && _running) {
        _stableHits     = 0;
        _prevUnionSmall = null;
        _autoCaptureFrame();
      }

    } catch (err) {
      _errorCount++;
      console.error(`[LiveDetectorDet] tick error (${_errorCount}):`, err);
      if (_errorCount >= 3) {
        console.error('[LiveDetectorDet] Too many consecutive errors — stopping.');
        if (_statusEl) _statusEl.textContent = `❌ Detection error: ${err.message.slice(0, 80)} — try closing and reopening camera.`;
        _running = false;
        if (_timer) { clearInterval(_timer); _timer = null; }
      }
    } finally {
      if (originalMat)     originalMat.delete();
      if (preprocessedMat) preprocessedMat.delete();
      _busy = false;
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    start(videoEl, overlayCanvas, statusEl, onCapture, autoCapture) {
      _videoEl       = videoEl;
      _overlayCanvas = overlayCanvas;
      _statusEl      = statusEl;
      _onCapture     = onCapture;
      _running         = true;
      _busy            = false;
      _autoCapture     = (autoCapture !== false);   // default true unless explicitly false
      _stableHits      = 0;
      _prevUnionSmall  = null;
      _lastDetBox      = null;
      _errorCount      = 0;
      if (_statusEl) _statusEl.textContent = 'Scanning… point camera at text';
      _timer = setInterval(_tick, SAMPLE_INTERVAL_MS);
      console.log('[LiveDetectorDet] started — interval', SAMPLE_INTERVAL_MS, 'ms');
    },

    stop() {
      _running = false;
      if (_timer) { clearInterval(_timer); _timer = null; }
      if (_overlayCanvas) {
        _overlayCanvas.getContext('2d').clearRect(0, 0, _overlayCanvas.width, _overlayCanvas.height);
      }
      console.log('[LiveDetectorDet] stopped');
    },

    captureNow() {
      _autoCaptureFrame();   // _lastDetBox included if a box was seen recently
    },

    setAutoCapture(enabled) {
      _autoCapture = enabled;
      // Reset stability so box doesn’t immediately fire if toggled on mid-stream
      if (enabled) { _stableHits = 0; _prevUnionSmall = null; }
    }
  };

})();
