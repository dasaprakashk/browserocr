/**
 * preprocessor.js
 * Handles image preprocessing optimised for low-contrast, curved tyre sidewall
 * images taken on phone cameras (embossed text, uneven lighting, dirt/mud).
 *
 * Pipeline:
 *   RGBA → BGR → resize (cap long side ≤ 1280)
 *              → grayscale
 *              → custom CLAHE (clip=2.5, 8×8 tiles) — local contrast boost
 *              → unsharp mask (sigma=1.0, α=1.5) — fine edge sharpening, no halos
 *              → back to 3-channel BGR for the detection model
 *
 * NOTE: Bilateral filter was intentionally removed — it destroys fine stamped
 * text edges on tyres despite denoising natural images well.
 */

const Preprocessor = (() => {

  // ── Tuning parameters ────────────────────────────────────────────────
  const MAX_SIDE    = 1280;  // resize input so longest side ≤ this
  const CLAHE_CLIP  = 2.5;   // lower clip = less aggressive, fewer halos on clean images
  const CLAHE_GRID  = 8;     // NxN tile grid
  // Unsharp mask: tight sigma preserves fine stamped edges, wider sigma creates halos
  const SHARP_ALPHA = 1.5;   // blend weight of original
  const SHARP_BETA  = -0.5;  // blend weight of blurred (negative)
  const SHARP_SIGMA = 1.0;   // small sigma = only sharpen very fine details, no halo

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Full preprocessing pipeline.
   * @param {HTMLCanvasElement|HTMLImageElement} source
   * @returns {{ original: cv.Mat, preprocessed: cv.Mat }}  both BGR, caller must .delete()
   */
  function process(source) {
    // ── 1. Read RGBA → BGR ──────────────────────────────────────────────────
    const rgba = cv.imread(source);
    const bgr  = new cv.Mat();
    cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
    rgba.delete();

    // ── 2. Resize so longest side ≤ MAX_SIDE ───────────────────────────────
    const original = _maybeResize(bgr);
    bgr.delete();

    // ── 3. Grayscale ──────────────────────────────────────────────────
    const gray = new cv.Mat();
    cv.cvtColor(original, gray, cv.COLOR_BGR2GRAY);

    // ── 4. Custom CLAHE ───────────────────────────────────────────────
    const clahed = _customCLAHE(gray, CLAHE_CLIP, CLAHE_GRID);
    gray.delete();

    // ── 5. Back to 3-channel BGR ──────────────────────────────────────────────
    const bgr3 = new cv.Mat();
    cv.cvtColor(clahed, bgr3, cv.COLOR_GRAY2BGR);
    clahed.delete();

    // ── 6. Unsharp mask ─────────────────────────────────────────────────
    const blurred     = new cv.Mat();
    const preprocessed = new cv.Mat();
    cv.GaussianBlur(bgr3, blurred, new cv.Size(0, 0), SHARP_SIGMA);
    cv.addWeighted(bgr3, SHARP_ALPHA, blurred, SHARP_BETA, 0, preprocessed);
    bgr3.delete();
    blurred.delete();

    return { original, preprocessed };
  }

  /**
   * Render a BGR cv.Mat onto a canvas for debug display.
   * @param {cv.Mat} bgrMat
   * @param {HTMLCanvasElement} canvas
   */
  function renderToCanvas(bgrMat, canvas) {
    const rgba = new cv.Mat();
    cv.cvtColor(bgrMat, rgba, cv.COLOR_BGR2RGBA);
    cv.imshow(canvas, rgba);
    rgba.delete();
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  /**
   * Resize a BGR mat so its longest side is ≤ MAX_SIDE.
   * Returns a new Mat (or clone of original if already small enough).
   */
  function _maybeResize(bgrMat) {
    const h = bgrMat.rows, w = bgrMat.cols;
    const scale = Math.min(MAX_SIDE / Math.max(h, w), 1.0);
    if (scale === 1.0) return bgrMat.clone();
    const out = new cv.Mat();
    cv.resize(bgrMat, out, new cv.Size(Math.round(w * scale), Math.round(h * scale)));
    return out;
  }

  /**
   * Custom CLAHE (Contrast Limited Adaptive Histogram Equalization).
   *
   * cv.createCLAHE is a contrib function not compiled into the standard
   * opencv.js WASM build, so we implement it from scratch here.
   *
   * Algorithm:
   *   1. Divide image into tileGrid × tileGrid tiles
   *   2. For each tile:
   *        a. Build 256-bin histogram of pixel values
   *        b. Clip any bin exceeding clipLimit * (tileArea / 256)
   *        c. Redistribute clipped counts uniformly across all bins
   *        d. Compute CDF → build a 256-entry LUT that maps old → new intensity
   *   3. For each output pixel, bilinearly interpolate among the four nearest
   *      tile LUTs (based on distance to tile centres) to avoid block artefacts
   *
   * @param {cv.Mat}  src        Grayscale CV_8U input
   * @param {number}  clipLimit  Contrast clip limit (e.g. 3.0)
   * @param {number}  tileGrid   Number of tiles per axis (e.g. 8)
   * @returns {cv.Mat}  New grayscale CV_8U mat, caller must .delete()
   */
  function _customCLAHE(src, clipLimit, tileGrid) {
    const rows    = src.rows;
    const cols    = src.cols;
    const srcData = src.data;          // Uint8ClampedArray or Uint8Array
    const tileH   = Math.ceil(rows / tileGrid);
    const tileW   = Math.ceil(cols / tileGrid);

    // ── Step 1 & 2: Build one LUT per tile ────────────────────────────────
    // luts[gy][gx] = Uint8Array(256)
    const luts = [];

    for (let gy = 0; gy < tileGrid; gy++) {
      luts[gy] = [];
      for (let gx = 0; gx < tileGrid; gx++) {
        const r0 = gy * tileH;
        const c0 = gx * tileW;
        const r1 = Math.min(r0 + tileH, rows);
        const c1 = Math.min(c0 + tileW, cols);
        const tileArea = (r1 - r0) * (c1 - c0);

        // Build histogram
        const hist = new Int32Array(256);
        for (let r = r0; r < r1; r++) {
          const rowOff = r * cols;
          for (let c = c0; c < c1; c++) {
            hist[srcData[rowOff + c]]++;
          }
        }

        // Clip & redistribute
        const clipThresh = Math.max(1, Math.round(clipLimit * tileArea / 256));
        let excess = 0;
        for (let i = 0; i < 256; i++) {
          if (hist[i] > clipThresh) {
            excess  += hist[i] - clipThresh;
            hist[i]  = clipThresh;
          }
        }
        const perBin   = Math.floor(excess / 256);
        let   leftover = excess % 256;
        for (let i = 0; i < 256; i++) {
          hist[i] += perBin;
          if (leftover > 0) { hist[i]++; leftover--; }
        }

        // CDF → LUT
        const lut = new Uint8Array(256);
        let cdf = 0;
        for (let i = 0; i < 256; i++) {
          cdf    += hist[i];
          lut[i]  = Math.min(255, Math.round((cdf * 255) / tileArea));
        }
        luts[gy][gx] = lut;
      }
    }

    // ── Step 3: Bilinear interpolation of LUT values for each pixel ────────
    const dst     = new cv.Mat(rows, cols, cv.CV_8U);
    const dstData = dst.data;

    for (let r = 0; r < rows; r++) {
      const rowOff = r * cols;

      // Fractional tile coordinate for this row (measured from tile centres)
      const gy_f = (r - tileH * 0.5) / tileH;
      const gy0  = Math.max(0, Math.min(tileGrid - 2, Math.floor(gy_f)));
      const gy1  = gy0 + 1 < tileGrid ? gy0 + 1 : gy0;
      const ty   = Math.max(0, Math.min(1, gy_f - gy0));

      for (let c = 0; c < cols; c++) {
        const pixel = srcData[rowOff + c];

        const gx_f = (c - tileW * 0.5) / tileW;
        const gx0  = Math.max(0, Math.min(tileGrid - 2, Math.floor(gx_f)));
        const gx1  = gx0 + 1 < tileGrid ? gx0 + 1 : gx0;
        const tx   = Math.max(0, Math.min(1, gx_f - gx0));

        // Bilinear blend of the 4 surrounding tile LUTs
        const v00 = luts[gy0][gx0][pixel];
        const v01 = luts[gy0][gx1][pixel];
        const v10 = luts[gy1][gx0][pixel];
        const v11 = luts[gy1][gx1][pixel];

        dstData[rowOff + c] = Math.round(
          (1 - ty) * ((1 - tx) * v00 + tx * v01) +
               ty  * ((1 - tx) * v10 + tx * v11)
        );
      }
    }

    return dst;
  }

  return { process, renderToCanvas };
})();

