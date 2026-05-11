/**
 * live-detector.js
 *
 * Not live detection anymore, rectangle box shows up
 * user needs to align the box over DOT number manually.
 */

const LiveDetectorDet = (() => {

  // ── Guide box proportions (fraction of video dimensions) ─────────────────
  // Landscape strip — matches typical DOT code aspect ratio
  const GUIDE_W_FRAC = 0.82;   // 82% of video width
  const GUIDE_H_FRAC = 0.28;   // 28% of video height

  // ── State ─────────────────────────────────────────────────────────────────
  let _videoEl       = null;
  let _overlayCanvas = null;
  let _statusEl      = null;
  let _onCapture     = null;
  let _running       = false;
  let _rafId         = null;

  // ── Guide rect helper ─────────────────────────────────────────────────────
  function _getGuideRect(w, h) {
    const gw = Math.round(w * GUIDE_W_FRAC);
    const gh = Math.round(h * GUIDE_H_FRAC);
    const gx = Math.round((w - gw) / 2);
    const gy = Math.round((h - gh) / 2);
    return { x: gx, y: gy, w: gw, h: gh };
  }

  // ── Draw guide overlay on every animation frame ───────────────────────────
  function _drawGuide() {
    if (!_running) return;

    const canvas = _overlayCanvas;
    const ctx    = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const { x, y, w, h } = _getGuideRect(W, H);

    ctx.clearRect(0, 0, W, H);

    // Dim everything outside the guide box
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, W, H);
    ctx.clearRect(x, y, w, h);

    // Guide box border
    ctx.strokeStyle = '#00E5FF';
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(x, y, w, h);

    // Corner accent marks
    const C = 22;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth   = 3.5;
    [
      [x,     y,      C, 0,   0,  C],
      [x+w,   y,     -C, 0,   0,  C],
      [x,     y+h,    C, 0,   0, -C],
      [x+w,   y+h,   -C, 0,   0, -C],
    ].forEach(([ox, oy, dx1, dy1, dx2, dy2]) => {
      ctx.beginPath();
      ctx.moveTo(ox + dx1, oy + dy1);
      ctx.lineTo(ox, oy);
      ctx.lineTo(ox + dx2, oy + dy2);
      ctx.stroke();
    });

    // Label above box
    ctx.font      = 'bold 13px system-ui, sans-serif';
    ctx.fillStyle = '#00E5FF';
    ctx.textAlign = 'center';
    ctx.fillText('Align DOT code inside the box', W / 2, y - 10);
    ctx.textAlign = 'left';

    _rafId = requestAnimationFrame(_drawGuide);
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return {
    start(videoEl, overlayCanvas, statusEl, onCapture) {
      _videoEl       = videoEl;
      _overlayCanvas = overlayCanvas;
      _statusEl      = statusEl;
      _onCapture     = onCapture;
      _running       = true;

      if (videoEl.videoWidth) {
        overlayCanvas.width  = videoEl.videoWidth;
        overlayCanvas.height = videoEl.videoHeight;
      }

      if (_statusEl) _statusEl.textContent = 'Align the DOT code inside the box, then press Capture';
      _rafId = requestAnimationFrame(_drawGuide);
      console.log('[LiveDetectorDet] started — guide-box mode');
    },

    stop() {
      _running = false;
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      if (_overlayCanvas) {
        _overlayCanvas.getContext('2d').clearRect(0, 0, _overlayCanvas.width, _overlayCanvas.height);
      }
      console.log('[LiveDetectorDet] stopped');
    },

    captureNow() {
      if (!_videoEl || !_videoEl.videoWidth) return;
      const vw = _videoEl.videoWidth;
      const vh = _videoEl.videoHeight;
      const cap = document.createElement('canvas');
      cap.width  = vw;
      cap.height = vh;
      cap.getContext('2d').drawImage(_videoEl, 0, 0);
      const { x, y, w, h } = _getGuideRect(vw, vh);
      if (_onCapture) _onCapture(cap, { x, y, width: w, height: h });
    },
  };

})();

