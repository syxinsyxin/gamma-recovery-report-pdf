# -*- coding: utf-8 -*-
"""
Batch recovery and gamma re-analysis of myQA verification reports
single-file script, v7

For each case the script pairs XXX.pdf (myQA verification report) with
XXX.dcm (Eclipse RT Dose export) by filename, rebuilds both dose maps from
the colour images embedded in the PDF, registers the DCM onto the display
grid, downsamples the measured map to the 32x32 MatriXX grid, and recomputes
gamma with pymedphys in several configurations:

  gamma_plan / gamma_plan_det   measured vs recovered TPS map (fine / detector grid)
  gamma_DCM / gamma_DCM_det     measured vs DCM (fine / detector grid)
  gamma_plan_sm / gamma_DCM_sm  aperture-smoothed controls on the fine grid
  gamma_plan_dirT               reversed reference convention (direction control)
  gamma_Self                    recovered TPS map vs DCM (self-consistency)

Each pathway is computed with the criteria declared in the report and with
3%/3 mm. Dose outside the covered region is set to 0 on every evaluation map
and removed by the low-dose cutoff, so it cannot lower the pass rate.

Usage
  pip install numpy scipy pillow pymupdf pydicom pymedphys
  python batch_gamma_recovery.py "F:/Work/Script/verification_2026" --out D:/bg_out
  python batch_gamma_recovery.py --limit 10        # trial run
  python batch_gamma_recovery.py --shard 0 4       # parallel shards
  python batch_gamma_recovery.py --no-ab           # skip the reversed control

Outputs: results.csv (opens in Excel), skipped.csv, summary.txt and QA
figures for flagged cases. Completed cases are skipped on a re-run, so a
long batch can be interrupted and resumed.
"""
import argparse
import csv
import inspect
import logging
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# keep console output UTF-8 (mainly for the Windows terminal)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# silence a few known library warnings; genuine errors are still caught and logged per case
warnings.filterwarnings("ignore", category=FutureWarning, message=".*fitz.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*quiet.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

# PyMuPDF: try the new module name first, fall back to the legacy fitz
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pymupdf as fitz          # new name
    except ImportError:
        try:
            import fitz                  # legacy name (fallback only)
        except ImportError:
            sys.exit("PyMuPDF not found: run  pip install pymupdf")
try:
    import pydicom
except ImportError:
    sys.exit("pydicom not found: run  pip install pydicom")
try:
    from pymedphys import gamma as pm_gamma
except ImportError:
    try:
        from pymedphys._gamma.implementation.shell import gamma_shell as pm_gamma
    except ImportError:
        sys.exit("pymedphys not found: run  pip install pymedphys")

# probe which keywords this pymedphys.gamma accepts, so unsupported ones can be dropped at call time
try:
    _PM_GAMMA_PARAMS = frozenset(inspect.signature(pm_gamma).parameters)
except (TypeError, ValueError):
    _PM_GAMMA_PARAMS = frozenset()   # probe failed: keep the full call, the TypeError retry below handles it
# a signature without the three core parameters (e.g. a bare *args wrapper) is
# unusable for filtering: dropping every kwarg would silently revert pymedphys
# to its default criteria (10%/10 mm), which the reports never used
_CORE_GAMMA_KW = frozenset({"dose_percent_threshold", "distance_mm_threshold",
                            "lower_percent_dose_cutoff"})
if _PM_GAMMA_PARAMS and not _CORE_GAMMA_KW.issubset(_PM_GAMMA_PARAMS):
    _PM_GAMMA_PARAMS = frozenset()

# several gamma calls per case; print each compatibility warning only once
_WARNED_GAMMA = set()

# pymedphys logs through logging; the old quiet= argument is deprecated, so raise the log level instead
for _name in ("pymedphys", "pymedphys.gamma", "pymedphys._gamma"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# ============================== Configuration ==============================
DEFAULT_ROOT = r"F:\verification_2026"

DIST_THRESH = 45.0        # max RGB distance to the colour curve for a pixel to count as valid
TICK_X_VALS = list(np.arange(-150, 151, 50))   # X axis tick values in mm (7 ticks)
TICK_Y_VALS = list(np.arange(-100, 101, 50))   # Y axis tick values in mm (5 ticks)
CALIB_FALLBACK = {"cx": (0.8739, -284.46), "cy": (0.8726, -153.14)}  # axis fit from prototype reports, used only if tick detection fails
EXPECTED_IMG_W, EXPECTED_IMG_H = 550, 360

# --- MatriXX detector grid ---
DET_N = 32                    # array side length (MatriXX 32x32)
DET_NOMINAL_PITCH_MM = 7.62   # nominal pitch (self-check only)
DET_PITCH_TOL = 0.12          # tolerance of +/-12%
DET_FILL_MIN_FLAG = 0.6       # flag low_det_fill below this worst-block valid-pixel fraction

# --- Gates ---
GATE_NCC = 0.98               # registration NCC gate
AXIS_MISMATCH_TOL_MM = 0.5    # axis end-point tolerance, about half a pixel
GATE_COVERAGE = 0.95          # DCM coverage
GATE_PP_DIFF_STRICT = 6.0     # |gamma_plan - reported| gate in pp (strict criteria, <=3%)
GATE_PP_DIFF_LOOSE = 3.0      # same gate under looser criteria (>3%)
GATE_CURSOR_PCT = 5.0         # gate on the POI cursor deviation (% of Dmax)

COARSE_RANGE = 8.0            # coarse search radius in mm around the DCM grid centre
COARSE_STEP = 2.0
REFINE_RANGE = 2.0
REFINE_STEP = 0.5

WRITE_EVERY = 10              # flush the CSV every N cases
# ======================================================================


# ---------------------------------------------------------------- text parsing
def parse_report_text(raw_text):
    """Parse report fields from the PDF text layer. Whitespace is stripped so
    the label patterns match regardless of the PDF extractor."""
    t = re.sub(r"\s+", "", raw_text)
    f = {}

    def grab(pat, cast=str):
        m = re.search(pat, t)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except Exception:
            return None

    f["patient"] = grab(r"Patient([A-Za-z\u4e00-\u9fff]+?)(?=PatientID|PatientGender)")
    f["machine"] = grab(r"Machine([A-Za-z0-9_\-]+)")
    f["energy"] = grab(r"EnergyValue([\d.]+MV)")
    f["gantry"] = grab(r"GantryAngle([\d.]+)", float)
    f["dmax_report"] = grab(r"DoseRatio\(100%\)([\d.]+)cGy", float)
    f["dp_rep"] = grab(r"DeltaDoseRatio([\d.]+)%", float)
    f["dd_rep"] = grab(r"DeltaDistance([\d.]+)mm", float)
    f["mode"] = grab(r"DoseErrorMode([A-Za-z]+)")
    f["search_mm"] = grab(r"SearchDistance([\d.]+)mm", float)
    f["thresh_pct"] = grab(r"Threshold([\d.]+)%", float)
    f["rep_pass"] = grab(r"PassingValues([\d.]+)%", float)
    f["rep_avg_gamma"] = grab(r"AverageValue([\d.]+)", float)
    # POI table: two-column layout "ReferenceValue<max dose>cGy<cursor value>cGy"
    m = re.search(r"ReferenceValue(-?[\d.]+)cGy(-?[\d.]+)cGy", t)
    f["poi_ref_max"], f["poi_ref_cursor"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"CompareValue(-?[\d.]+)cGy(-?[\d.]+)cGy", t)
    f["poi_cmp_max"], f["poi_cmp_cursor"] = (float(m.group(1)), float(m.group(2))) if m else (None, None)

    # colour-bar range "2.261 - 274.208 cGy": among all min-max candidates keep the smallest plausible minimum
    cands = [(float(a), float(b)) for a, b in re.findall(r"([\d.]+)[-–—]([\d.]+)cGy", t)]
    cands = [(a, b) for a, b in cands if 0 <= a < b and 50 < b < 5000 and a < 0.2 * b]
    f["dmin_fallback"] = False
    if cands:
        f["dmin_cgy"], f["dmax_cgy"] = min(cands, key=lambda c: c[0])
    else:
        f["dmin_cgy"], f["dmax_cgy"] = 0.0, f["dmax_report"]
        f["dmin_fallback"] = True
    return f


# ---------------------------------------------------------------- PDF images
def extract_pdf_images(pdf_path, tmp_dir=None):
    """Locate the Reference/Compare dose maps from their layout labels.
    The same words also appear as left-panel headings, so the label must sit
    close to the image itself. In this template Reference is the measured dose
    and Compare the planned dose. Returns (ref_img, cmp_img), (H,W,3) uint8"""
    ref_img = cmp_img = None
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            rects = {}
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in rects:
                    continue
                rs = page.get_image_rects(xref)
                if rs:
                    rects[xref] = rs[0]
            occ = {"reference": [], "compare": []}
            for w in page.get_text("words"):
                key = w[4].rstrip(":").lower()
                if key in occ:
                    occ[key].append((w[0], w[1]))
            found = {}
            for xref, r in rects.items():
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.colorspace is not None and pix.colorspace.n > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.n < 3 or pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8)
                    a = arr.reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
                except Exception:
                    continue
                best = None
                for key, pts in occ.items():
                    for lx, ly in pts:
                        if ly <= r.y0 + 15 and lx > r.x0 - 100:
                            d = abs(lx - r.x0) + (r.y0 - ly) * 0.2
                            if best is None or d < best[0]:
                                best = (d, key)
                if best:
                    found.setdefault(best[1], a)
            if "reference" in found and "compare" in found:
                ref_img, cmp_img = found["reference"], found["compare"]
                break
    if ref_img is None or cmp_img is None:
        raise RuntimeError("could not locate the Reference/Compare dose maps from the layout labels")
    if tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ref_img).save(tmp_dir / "img_ref.png")
        Image.fromarray(cmp_img).save(tmp_dir / "img_cmp.png")
    return ref_img, cmp_img


# ---------------------------------------------------------------- inversion core
def detect_frame(a):
    """Find the plot frame of a dose map; returns (xl, xr, yt, yb)."""
    g = a.mean(axis=2)
    bm = g < 130
    H, W = bm.shape
    vc = np.where(bm.sum(axis=0) > 0.70 * H)[0]
    if len(vc) < 2:
        raise RuntimeError("vertical frame-line detection failed")
    xl, xr = vc.min(), vc.max()
    if xr - xl < 10:
        raise RuntimeError("vertical frame-line spacing too small")
    inner = bm[:, xl + 2:xr - 1]
    hc = np.where(inner.sum(axis=1) > 0.85 * (xr - xl - 3))[0]
    if len(hc) < 2:
        raise RuntimeError("horizontal frame-line detection failed")
    return xl, xr, hc.min(), hc.max()


def cluster(vals, gap=5, minsize=1):
    """Cluster nearby 1-D positions and return the group means."""
    vals = np.sort(np.asarray(vals, float))
    if len(vals) == 0:
        return []
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(c)) for c in groups if len(c) >= minsize]


def detect_ticks(a, xl, xr, yt, yb):
    """Detect axis tick marks and check their counts against the template."""
    g = a.mean(axis=2)
    band = (g[yb + 2: yb + 8, :] < 150).sum(axis=0)
    xs = np.where(band >= 2)[0]
    xs = xs[(xs >= xl - 5) & (xs <= xr + 5)]
    band2 = (g[:, max(xl - 6, 0): max(xl - 2, 1)] < 150).sum(axis=1)
    ys = np.where(band2 >= 2)[0]
    ys = ys[(ys >= yt - 8) & (ys <= yb + 8)]
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("tick-mark detection failed")
    xt = cluster(xs, gap=4)
    ytk = cluster(ys, gap=4)
    if len(xt) != len(TICK_X_VALS) or len(ytk) != len(TICK_Y_VALS):
        raise RuntimeError(f"abnormal tick count: X={len(xt)} (expected {len(TICK_X_VALS)}) Y={len(ytk)} (expected {len(TICK_Y_VALS)})")
    dxs, dys = np.diff(xt), np.diff(ytk)
    if abs(np.mean(dxs) - np.mean(dys)) > 2.0:
        raise RuntimeError("X and Y tick spacing are inconsistent")
    return xt, ytk


def fit_axis_auto(px_sorted, vals_ascending, name):
    """Least-squares pixel-to-mm axis fit over the tick positions (either direction)."""
    best = None
    for vals in (vals_ascending, vals_ascending[::-1]):
        if len(px_sorted) != len(vals):
            continue
        px = np.asarray(px_sorted, float)
        vv = np.asarray(vals, float)
        A = np.vstack([px, np.ones_like(px)]).T
        coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
        err = np.abs(A @ coef - vv).max()
        if best is None or err < best[1]:
            best = (coef, err)
    if best is None:
        raise RuntimeError(f"{name} axis: unexpected number of ticks")
    return best[0]


def detect_colorbar(a, xl):
    """Extract the RGB strip of the colour bar, left of the plot frame."""
    left = a[:, : max(xl - 4, 1)]
    sat = left.max(axis=2).astype(int) - left.min(axis=2).astype(int)
    colcnt = (sat > 40).sum(axis=0)
    cols = np.where(colcnt > 45)[0]
    if len(cols) == 0:
        raise RuntimeError("colour bar not found")
    segs, cur = [], [cols[0]]
    for c in cols[1:]:
        if c - cur[-1] <= 3:
            cur.append(c)
        else:
            segs.append(cur)
            cur = [c]
    segs.append(cur)
    seg = max(segs, key=len)
    c0, c1 = seg[0] + 2, seg[-1] - 2
    if c1 < c0:
        c0, c1 = seg[0], seg[-1]
    strip = left[:, c0:c1 + 1]
    stripsat = sat[:, c0:c1 + 1] > 40
    rows = np.where(stripsat.sum(axis=1) > (c1 - c0 + 1) * 0.7)[0]
    if len(rows) == 0:
        raise RuntimeError("colour-bar row-range detection failed")
    rgb = np.median(strip[rows.min():rows.max() + 1], axis=1).astype(float)
    if len(rgb) < 2:
        raise RuntimeError("too few valid colour-bar rows")
    return rgb


def invert_curve(a, curve_rgb, curve_dose):
    """Convert pixel colours to dose through the colour curve; returns (dose, valid mask)."""
    H, W, _ = a.shape
    flat = a.reshape(-1, 3).astype(np.float32)
    curve_rgb = curve_rgb.astype(np.float32)
    N, M = len(flat), len(curve_rgb)
    idx = np.empty(N, dtype=np.int32)
    dist = np.empty(N, dtype=np.float32)
    B = 120000
    for i in range(0, N, B):
        chunk = flat[i:i + B]
        d = ((chunk[:, None, :] - curve_rgb[None, :, :]) ** 2).sum(-1)
        idx[i:i + B] = d.argmin(1)
        dist[i:i + B] = np.sqrt(d.min(1))
    P = curve_rgb.astype(np.float64)
    D = np.asarray(curve_dose, dtype=np.float64)

    def seg_proj(s0):
        p0, p1 = P[s0], P[s0 + 1]
        v = p1 - p0
        L2 = (v ** 2).sum(1)
        L2 = np.where(L2 > 0, L2, 1e-9)
        tt = np.clip(((flat.astype(np.float64) - p0) * v).sum(1) / L2, 0, 1)
        proj = p0 + tt[:, None] * v
        dd = np.linalg.norm(flat.astype(np.float64) - proj, axis=1)
        dose = D[s0] + tt * (D[s0 + 1] - D[s0])
        return dd, dose

    s_dn = np.clip(idx - 1, 0, M - 2)
    s_up = np.clip(idx, 0, M - 2)
    dd_dn, dose_dn = seg_proj(s_dn)
    dd_up, dose_up = seg_proj(s_up)
    use_up = dd_up < dd_dn
    dd_seg = np.where(use_up, dd_up, dd_dn)
    dose_seg = np.where(use_up, dose_up, dose_dn)
    dose = np.where(dd_seg < dist, dose_seg, D[np.clip(idx, 0, M - 1)])
    valid = dist < DIST_THRESH
    return dose.reshape(H, W), valid.reshape(H, W)


def clean_and_fill(dose, valid):
    """Tidy the valid mask and fill invalid pixels from their nearest valid neighbours."""
    vf = ndimage.binary_opening(valid, structure=np.ones((2, 2)))
    vf = ndimage.binary_closing(vf, structure=np.ones((2, 2)))
    dose = np.where(vf, dose, np.nan)
    invalid = ~vf
    invalid_d = ndimage.binary_dilation(invalid, iterations=2)
    ind = ndimage.distance_transform_edt(invalid_d, return_distances=False, return_indices=True)
    filled = dose[tuple(ind)]
    med = ndimage.median_filter(filled, size=3)
    use_med = vf & ~ndimage.binary_dilation(invalid, iterations=3)
    final = np.where(use_med, med, filled)
    return final, vf


def recover_map(a, dmax_cgy, dmin_cgy):
    """Recover one dose map: frame/tick detection -> LUT inversion -> fill -> mm axes.
    Returns (dose with NaN, xmm, ymm, colour curve, info dict)"""
    xl, xr, yt, yb = detect_frame(a)
    try:
        xt, ytk = detect_ticks(a, xl, xr, yt, yb)
        cx = fit_axis_auto(xt, TICK_X_VALS, "X")
        cy = fit_axis_auto(ytk, TICK_Y_VALS, "Y")
        calib = "ticks"
    except RuntimeError:
        if a.shape[0] != EXPECTED_IMG_H or a.shape[1] != EXPECTED_IMG_W:
            raise
        cx, cy = CALIB_FALLBACK["cx"], CALIB_FALLBACK["cy"]
        calib = "fallback"
    rgb = detect_colorbar(a, xl)
    M = len(rgb)
    curve_dose = dmax_cgy + (np.arange(M) / (M - 1)) * (dmin_cgy - dmax_cgy)
    dose, valid = invert_curve(a, rgb, curve_dose)
    final, vf = clean_and_fill(dose, valid)
    box = np.zeros_like(vf)
    box[yt + 2:yb - 1, xl + 2:xr - 1] = True
    vf &= box
    final = np.where(vf, final, np.nan)
    xmm = cx[0] * np.arange(a.shape[1]) + cx[1]
    ymm = cy[0] * np.arange(a.shape[0]) + cy[1]
    if ymm[0] > ymm[-1]:          # sort to ascending y
        final, ymm = final[::-1], ymm[::-1]
    if xmm[0] > xmm[-1]:          # sort to ascending x (pymedphys requires monotonic axes)
        final, xmm = final[:, ::-1], xmm[::-1]
    info = {"frame": (xl, xr, yt, yb), "calib": calib, "valid_pct": float(vf.mean())}
    return final, xmm, ymm, rgb, info


def cursor_readout(dose_filled, vf, xmm, ymm, x_mm=0.0, y_mm=0.0, r_mm=2.0):
    """Dose at the POI cursor of the report. The crosshair usually covers the
    exact pixel, so take the median over a small neighbourhood of valid pixels"""
    if not np.any(vf):
        return float("nan")
    ax = (xmm[-1] - xmm[0]) / (len(xmm) - 1)
    ay = (ymm[-1] - ymm[0]) / (len(ymm) - 1)
    ix = int(round((x_mm - xmm[0]) / ax))
    iy = int(round((y_mm - ymm[0]) / ay))
    r2 = max(int(round(r_mm / abs(ay))), 1)
    c2 = max(int(round(r_mm / abs(ax))), 1)
    nb = np.zeros_like(vf)
    nb[max(iy - r2, 0):iy + r2 + 1, max(ix - c2, 0):ix + c2 + 1] = True
    sel = vf & nb
    if not np.any(sel):
        return float("nan")
    return float(np.nanmedian(dose_filled[sel]))


# ---------------------------------------------------------------- detector grid
def _block_edges(total, n):
    """Split [0, total) into n nearly equal segments; integer boundaries."""
    return np.round(np.linspace(0, total, n + 1)).astype(int)


def box_average(a, n):
    """Block averaging over non-integer boundaries via cumulative sums.
    Only valid pixels enter a block; a fully invalid block stays NaN.
    Returns (blocks[n,n], counts[n,n], (y_edges, x_edges))"""
    H, W = a.shape
    if n > H or n > W:
        raise ValueError(f"block count {n} exceeds the array size ({H},{W})")
    ye = _block_edges(H, n)
    xe = _block_edges(W, n)
    valid = np.isfinite(a)
    av = np.where(valid, a, 0.0).astype(np.float64)
    nv = valid.astype(np.float64)

    def cs(x):
        p = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
        p[1:, 1:] = x
        return p.cumsum(0).cumsum(1)

    SA, SN = cs(av), cs(nv)
    blocks = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        y0, y1 = ye[i], ye[i + 1]
        for j in range(n):
            x0, x1 = xe[j], xe[j + 1]
            s = SA[y1, x1] - SA[y0, x1] - SA[y1, x0] + SA[y0, x0]
            c = SN[y1, x1] - SN[y0, x1] - SN[y1, x0] + SN[y0, x0]
            counts[i, j] = c
            if c > 0:
                blocks[i, j] = s / c
    return blocks, counts, (ye, xe)


def nan_uniform_filter(a, size):
    """Uniform mean filter that ignores NaNs.

    Emulates the detector aperture on the fine grid: it removes sub-chamber
    detail without changing the sampling density, so the interpolation used by
    the gamma search keeps its accuracy. Downsampling to 32x32 does the
    opposite and is therefore not used to test the sampling asymmetry.

    size: int (square kernel) or (size_y, size_x) rectangular kernel, in pixels.
    """
    if size is None:
        return a
    if isinstance(size, (tuple, list)):
        sy, sx = int(size[0]), int(size[1])
    else:
        sy = sx = int(size)
    if sy < 2 and sx < 2:
        return a
    w = np.ones((max(sy, 1), max(sx, 1)), dtype=np.float64)
    valid = np.isfinite(a).astype(np.float64)
    num = ndimage.convolve(np.where(np.isfinite(a), a, 0.0), w, mode="constant", cval=0.0)
    den = ndimage.convolve(valid, w, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


def to_detector(arr, xmm, ymm, n=DET_N):
    """Crop to the valid-region rectangle and block-average onto an n x n
    detector grid (block averaging approximates the chamber volume averaging).

    Returns (blocks, (y_mm, x_mm), counts, geom), or (None,)*4 on failure.
    geom = (y0, x0, ye_full, xe_full): block boundaries in full-image
    coordinates. It is derived once from the measured map and reused for the
    TPS map and the DCM via block_with_edges(), so all three share identical
    block boundaries. Taking each map's bounding box separately shifts the
    boundaries by a pixel or two and adds a false misregistration to the
    controls.
    """
    vf = np.isfinite(arr)
    if vf.sum() < n * n:
        return None, None, None, None
    ys, xs = np.where(vf)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (y1 - y0 + 1) < n or (x1 - x0 + 1) < n:
        return None, None, None, None
    sub = arr[y0:y1 + 1, x0:x1 + 1]
    try:
        blocks, counts, (ye, xe) = box_average(sub, n)
    except ValueError:
        return None, None, None, None
    ysub = np.asarray(ymm)[y0:y1 + 1]
    xsub = np.asarray(xmm)[x0:x1 + 1]
    yc = np.array([ysub[ye[i]:ye[i + 1]].mean() for i in range(n)], dtype=np.float64)
    xc = np.array([xsub[xe[j]:xe[j + 1]].mean() for j in range(n)], dtype=np.float64)
    if not (np.all(np.diff(yc) > 1e-6) and np.all(np.diff(xc) > 1e-6)):
        return None, None, None, None   # axes must be strictly increasing or pymedphys raises
    geom = (y0, x0, ye + y0, xe + x0)   # convert to full-image coordinates
    return blocks, (yc, xc), counts, geom


def block_with_edges(arr, geom, n=DET_N):
    """Block-average an arbitrary dose plane with the shared geom from to_detector().

    Only valid (non-NaN) pixels enter; a fully invalid block stays NaN.
    Returns (blocks[n,n], counts[n,n], worst-block fill fraction)"""
    if geom is None:
        return None, None, None
    y0, x0, ye, xe = geom
    H, W = arr.shape
    if y0 < 0 or x0 < 0 or ye[-1] > H or xe[-1] > W:
        return None, None, None
    valid = np.isfinite(arr)
    av = np.where(valid, arr, 0.0).astype(np.float64)
    nv = valid.astype(np.float64)

    def cs(x):
        p = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
        p[1:, 1:] = x
        return p.cumsum(0).cumsum(1)

    SA, SN = cs(av), cs(nv)
    blocks = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a0, a1 = ye[i], ye[i + 1]
        for j in range(n):
            b0, b1 = xe[j], xe[j + 1]
            s = SA[a1, b1] - SA[a0, b1] - SA[a1, b0] + SA[a0, b0]
            c = SN[a1, b1] - SN[a0, b1] - SN[a1, b0] + SN[a0, b0]
            counts[i, j] = c
            if c > 0:
                blocks[i, j] = s / c
    area = np.outer(np.diff(ye), np.diff(xe)).astype(np.float64)
    fill = counts / np.maximum(area, 1.0)
    return blocks, counts, float(fill[area > 0].min()) if (area > 0).any() else None


# ---------------------------------------------------------------- DCM
def read_dcm(dcm_path):
    """Read a single-frame RT Dose plane; returns (dose in cGy, u_axis, v_axis, row spacing)."""
    ds = pydicom.dcmread(str(dcm_path))
    nf = getattr(ds, "NumberOfFrames", 1)
    if nf and int(nf) != 1:
        raise RuntimeError(f"DCM is multi-frame ({nf}), this script supports single-frame dose planes only")
    scale = float(getattr(ds, "DoseGridScaling", 1.0))
    dose = ds.pixel_array.astype(np.float64) * scale
    units = str(getattr(ds, "DoseUnits", "GY")).upper()
    if units == "GY":
        dose *= 100.0
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    sp = [float(x) for x in ds.PixelSpacing]
    drow, dcol = sp[0], sp[1]
    u_axis = np.dot(ipp, iop[:3]) + np.arange(dose.shape[1]) * dcol
    v_axis = np.dot(ipp, iop[3:]) + np.arange(dose.shape[0]) * drow
    return dose, u_axis, v_axis, float(drow)


# ---------------------------------------------------------------- registration
def ncc(a, b, m):
    """Masked Pearson correlation of two arrays."""
    if m.sum() < 1000:
        return -1.0
    a0 = a[m].astype(np.float64)
    b0 = b[m].astype(np.float64)
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()
    den = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
    return float((a0 * b0).sum() / den) if den > 0 else -1.0


def register_dcm(cmp_rec, valid_cmp, X, Y, u_axis, v_axis, dcm_cgy):
    """Register the DCM onto the display grid: 4 flip combinations x translations,
    seeded at grid-centre alignment. Returns (sx, sy, du, dv, ncc, sampled,
    coverage mask, escalated). Only the two best seeds are refined; if the NCC
    gate is still missed, the remaining seeds are tried as well."""
    interp = RegularGridInterpolator((v_axis, u_axis), dcm_cgy,
                                     bounds_error=False, fill_value=np.nan)
    du0 = float((u_axis[0] + u_axis[-1]) / 2)
    dv0 = float((v_axis[0] + v_axis[-1]) / 2)

    def samp(sx, sy, du, dv):
        return interp(np.stack([sy * Y + dv, sx * X + du], axis=-1))

    seeds = []
    for sx in (1, -1):
        for sy in (1, -1):
            s = samp(sx, sy, du0, dv0)
            m = valid_cmp & np.isfinite(s)
            seeds.append((ncc(cmp_rec, s, m), sx, sy))
    seeds.sort(reverse=True)

    def opt_seed(sx, sy):
        """Single seed: coarse search plus refinement; returns (ncc, sx, sy, du, dv, sampled array)"""
        loc = (-2.0, du0, dv0)
        for du in du0 + np.arange(-COARSE_RANGE, COARSE_RANGE + 1e-9, COARSE_STEP):
            for dv in dv0 + np.arange(-COARSE_RANGE, COARSE_RANGE + 1e-9, COARSE_STEP):
                s = samp(sx, sy, du, dv)
                c = ncc(cmp_rec, s, valid_cmp & np.isfinite(s))
                if c > loc[0]:
                    loc = (c, du, dv)
        for du in np.arange(loc[1] - REFINE_RANGE, loc[1] + REFINE_RANGE + 1e-9, REFINE_STEP):
            for dv in np.arange(loc[2] - REFINE_RANGE, loc[2] + REFINE_RANGE + 1e-9, REFINE_STEP):
                s = samp(sx, sy, du, dv)
                c = ncc(cmp_rec, s, valid_cmp & np.isfinite(s))
                if c > loc[0]:
                    loc = (c, du, dv)
        return (loc[0], sx, sy, loc[1], loc[2], samp(sx, sy, loc[1], loc[2]))

    best = (-2.0, 1, 1, du0, dv0, None)
    for seed_ncc, sx, sy in seeds[:2]:
        cand = opt_seed(sx, sy)
        if cand[0] > best[0]:
            best = cand
    # Direction-anomaly guard: if the NCC is still below the gate after the first two seeds have
    # been refined, the remaining flipped seeds are tried. Cases that pass normally pay nothing;
    # this guards against a correctly flipped DCM ranked third or fourth being cut off by the seed list
    escalated = False
    if best[0] < GATE_NCC and len(seeds) > 2:
        escalated = True
        for seed_ncc, sx, sy in seeds[2:]:
            cand = opt_seed(sx, sy)
            if cand[0] > best[0]:
                best = cand
    n, sx, sy, du, dv, s = best
    if s is None:
        s = samp(sx, sy, du, dv)
    covered = np.isfinite(s)
    return sx, sy, du, dv, n, np.where(covered, s, 0.0), covered, escalated


# ---------------------------------------------------------------- gamma
def run_gamma(axes_ref, ref, axes_evl, evl, dp, dd, gnorm, cutoff_pct):
    """Compute gamma via pymedphys.gamma; unsupported keywords are dropped with a warning.
    Returns (pass rate %, mean gamma, gamma map)"""
    axes_ref = (np.asarray(axes_ref[0], float), np.asarray(axes_ref[1], float))
    axes_evl = (np.asarray(axes_evl[0], float), np.asarray(axes_evl[1], float))
    ref = np.asarray(ref, dtype=np.float64)
    evl = np.asarray(evl, dtype=np.float64)
    kw = dict(dose_percent_threshold=float(dp), distance_mm_threshold=float(dd),
              lower_percent_dose_cutoff=float(cutoff_pct), interp_fraction=10,
              max_gamma=2.0, local_gamma=False, global_normalisation=float(gnorm))
    if _PM_GAMMA_PARAMS:
        dropped = [k for k in kw if k not in _PM_GAMMA_PARAMS]
        for k in dropped:
            kw.pop(k)
        if dropped:
            key = ",".join(sorted(dropped))
            if key not in _WARNED_GAMMA:
                _WARNED_GAMMA.add(key)
                logging.warning("pymedphys.gamma does not support parameter %s, dropping it (if interp_fraction is dropped, "
                                "the gamma sub-pixel interpolation density falls back to the default and the pass rate may be biased low; upgrading pymedphys is recommended)", dropped)
    try:
        g = pm_gamma(axes_ref, ref, axes_evl, evl, **kw)
    except TypeError:
        # signature probing failed and the call still raised TypeError: retry
        # with the three core arguments only. global_normalisation is then lost
        # (pymedphys normalises to the reference maximum), so the retry warns.
        if "typeerror_retry" not in _WARNED_GAMMA:
            _WARNED_GAMMA.add("typeerror_retry")
            logging.warning("pymedphys.gamma failed with the full argument set (TypeError); retried with the minimal core arguments; "
                            "global_normalisation/local_gamma were not applied, so the results may differ slightly from the report; "
                            "upgrading pymedphys is recommended")
        kw_min = {k: kw[k] for k in ("dose_percent_threshold", "distance_mm_threshold",
                                     "lower_percent_dose_cutoff") if k in kw}
        g = pm_gamma(axes_ref, ref, axes_evl, evl, **kw_min)
    g = np.asarray(g, dtype=np.float64)
    v = np.isfinite(g)
    if v.sum() == 0:
        return float("nan"), float("nan"), g
    return 100.0 * float(np.mean(g[v] <= 1.0)), float(np.mean(g[v])), g


def gamma_map_png(g, path):
    """Save a gamma map as a small blue-white-red PNG (gamma clipped to [0, 2])."""
    if g is None:
        return
    g = np.asarray(g, dtype=np.float64)
    if g.size == 0:
        return
    img = np.full((*g.shape, 3), 255, dtype=np.uint8)
    v = np.clip(np.nan_to_num(g, nan=0.0), 0, 2) / 2.0
    img[..., 0] = (66 + v * 189).astype(np.uint8)
    img[..., 1] = (146 + 109 * np.sin(np.pi * v)).astype(np.uint8)
    img[..., 2] = (255 - v * 189).astype(np.uint8)
    img[~np.isfinite(g)] = 255
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------- single case ------
def process_case(pdf_path, dcm_path, qa_dir, save_qa=False, do_ab=True):
    """Full pipeline for one PDF/DCM pair; returns the result row."""
    t0 = time.time()
    row = {"flag": "", "error": ""}

    def setflag(msg):
        row["flag"] = (row["flag"] + ";" + msg) if row["flag"] else msg

    # ---- 1. PDF text ----
    with fitz.open(str(pdf_path)) as doc:
        raw_text = "\n".join(pg.get_text("text") for pg in doc)
    rep = parse_report_text(raw_text)
    if rep["rep_pass"] is None or rep["dp_rep"] is None:
        raise RuntimeError(f"report field parsing failed: dp={rep['dp_rep']} pass={rep['rep_pass']}")
    row.update({f"rep_{k}": rep[k] for k in
                ("dp_rep", "dd_rep", "thresh_pct", "rep_pass", "rep_avg_gamma",
                 "dmax_cgy", "poi_ref_max", "poi_cmp_max",
                 "poi_ref_cursor", "poi_cmp_cursor")})
    # the colour-scale lower limit is a rasterised label with no text field,
    # so it is not stored as a rep_ column; the estimate goes to est_dmin_cgy
    dmax = rep["dmax_cgy"]
    if not dmax:
        dmax = rep["poi_ref_max"]      # fall back to the POI reference maximum, still a report value
        if dmax:
            setflag("dmax_fallback_poi")
    if not dmax:
        raise RuntimeError("dmax unavailable: neither the colour-bar range nor the POI reference maximum could be parsed; "
                           "refusing to fall back silently to a hard-coded value (274.208 applies to one plan only)")
    gnorm = float(dmax)
    # dmin is set below; the pre-inversion runs with dmin = 0
    cutoff_pct = rep["thresh_pct"] or 5.0

    # ---- 2. Image extraction and inversion ----
    ref_img, cmp_img = extract_pdf_images(pdf_path, None)
    # the lower scale limit is not in the text layer. Estimate it: invert once
    # with dmin = 0 and read the 0.1th percentile of the valid pixels (the LUT
    # is affine), then invert both maps again with that value
    rec_ref0, xmm, ymm, _, _ = recover_map(ref_img, dmax, 0.0)
    if np.any(np.isfinite(rec_ref0)):
        dmin_est = float(np.nanpercentile(rec_ref0, 0.1))
    else:
        dmin_est = np.nan
    if np.isfinite(dmin_est) and 0.2 <= dmin_est <= 80:
        dmin = dmin_est
    else:
        dmin = 0.0
        setflag("dmin_fallback")
    row["est_dmin_cgy"] = round(dmin, 3)   # estimated value, not a report field
    rec_ref, xmm, ymm, _, info_r = recover_map(ref_img, dmax, dmin)
    rec_cmp, xmm_c, ymm_c, _, info_c = recover_map(cmp_img, dmax, dmin)
    # both maps must share the display axes. A +/-1 px tick jitter moves the
    # fitted end points by ~0.05 mm, so an exact axis match would flag most
    # cases; 0.5 mm (~half a pixel) is harmless at 3%/3 mm
    axis_ok = (abs(xmm_c[0] - xmm[0]) < AXIS_MISMATCH_TOL_MM
               and abs(xmm_c[-1] - xmm[-1]) < AXIS_MISMATCH_TOL_MM
               and abs(ymm_c[0] - ymm[0]) < AXIS_MISMATCH_TOL_MM
               and abs(ymm_c[-1] - ymm[-1]) < AXIS_MISMATCH_TOL_MM)
    if rec_cmp.shape != rec_ref.shape or not axis_ok:
        setflag("cmp_axis_mismatch")
    row["ref_valid_pct"] = round(info_r["valid_pct"] * 100, 2)
    row["cmp_valid_pct"] = round(info_c["valid_pct"] * 100, 2)
    if info_r["calib"] == "fallback" or info_c["calib"] == "fallback":
        setflag("calib_fallback")

    # POI cursor readout (recovery self-check)
    vf_r = np.isfinite(rec_ref)
    vf_c = np.isfinite(rec_cmp)
    filled_r = np.where(vf_r, rec_ref, 0.0)
    filled_c = np.where(vf_c, rec_cmp, 0.0)
    if rep["poi_ref_cursor"]:
        row["rec_ref_cursor"] = round(cursor_readout(filled_r, vf_r, xmm, ymm), 2)
        row["dev_ref_cursor"] = round(row["rec_ref_cursor"] - rep["poi_ref_cursor"], 2)
        if abs(row["dev_ref_cursor"]) > GATE_CURSOR_PCT / 100.0 * gnorm:
            setflag("cursor_dev")
    if rep["poi_cmp_cursor"]:
        row["rec_cmp_cursor"] = round(cursor_readout(filled_c, vf_c, xmm, ymm), 2)
        row["dev_cmp_cursor"] = round(row["rec_cmp_cursor"] - rep["poi_cmp_cursor"], 2)

    # ---- 3. DCM ----
    dcm_cgy, u_axis, v_axis, dcm_spacing = read_dcm(dcm_path)
    row["dcm_max_cgy"] = round(float(np.max(dcm_cgy)), 2)
    row["dcm_pixels"] = int(dcm_cgy.size)
    row["dcm_spacing_mm"] = round(dcm_spacing, 3)
    if rep["poi_cmp_max"]:
        row["dcm_vs_poi_max"] = round(row["dcm_max_cgy"] - rep["poi_cmp_max"], 2)
        if abs(row["dcm_vs_poi_max"]) > 2.0:
            setflag("dcm_max_mismatch")

    # ---- 4. Registration ----
    X, Y = np.meshgrid(np.asarray(xmm, float), np.asarray(ymm, float))
    sx, sy, du, dv, nccv, dcm_on_grid, covered, reg_escal = register_dcm(
        np.where(vf_c, rec_cmp, 0.0), vf_c, X, Y, u_axis, v_axis, dcm_cgy)
    row.update({"reg_ncc": round(nccv, 4), "reg_sx": sx, "reg_sy": sy,
                "reg_du": round(du, 2), "reg_dv": round(dv, 2),
                "reg_coverage": round(float(covered[vf_c].mean()), 4)})
    if reg_escal:
        setflag("reg_seed_escalated")   # first two seeds missed the gate; all flips were tried
    if nccv < GATE_NCC:
        setflag("low_reg_ncc")
    if covered[vf_c].mean() < GATE_COVERAGE:
        setflag("low_coverage")

    # ---- 5. Dose-domain error (recovered comparison map vs DCM) ----
    m = vf_c & covered
    if m.sum() > 100:
        diff = rec_cmp[m] - dcm_on_grid[m]
        row["dose_bias"] = round(float(np.mean(diff)), 2)
        row["dose_sigma"] = round(float(np.std(diff)), 2)
        # the uncovered region is NaN before the gradient is taken: dcm_on_grid
        # is 0 there and the step would inflate the gradient on the inner side
        # of the boundary
        grad_map = np.where(covered, dcm_on_grid, np.nan)
        gy, gx = np.gradient(grad_map)
        gmap = np.hypot(gy, gx)
        m2 = m & np.isfinite(gmap)
        if m2.sum() >= 4:
            diff2 = rec_cmp[m2] - dcm_on_grid[m2]
            grad = gmap[m2]
            q1, q3 = np.quantile(grad, 0.25), np.quantile(grad, 0.75)
            s_smooth = diff2[grad < q1]
            s_grad = diff2[grad >= q3]
            row["dose_sigma_smooth"] = round(float(np.std(s_smooth)), 2) if s_smooth.size > 10 else ""
            row["dose_sigma_grad"] = round(float(np.std(s_grad)), 2) if s_grad.size > 10 else ""
        else:
            row["dose_sigma_smooth"] = ""
            row["dose_sigma_grad"] = ""
    row["sat_pct"] = round(float((rec_cmp[vf_c] >= gnorm - 0.01).mean() * 100), 3)

    # ---- 6. Downsampling of the measured map to the MatriXX detector grid ----
    # myQA evaluates gamma on the 32x32 detector grid only, so the measured map
    # is downsampled; cumulative-sum averaging keeps the blocks exact even for
    # a non-integer pitch
    ref0 = np.where(vf_r, rec_ref, 0.0)
    cmp0 = np.where(vf_c, rec_cmp, 0.0)
    n_uncov = int(((ref0 >= 0.05 * gnorm) & ~covered).sum())
    row["uncov_pts"] = n_uncov

    axes = (np.asarray(ymm, float), np.asarray(xmm, float))
    # only assigned in the else branch; None here keeps a failed downsampling harmless
    dcm_det_g = None
    cmp_det_g = None
    tps_aps_g = None
    dcm_aps_g = None
    geom = None
    meas_det, ax_det, det_counts, geom = to_detector(rec_ref, xmm, ymm, DET_N)
    if meas_det is None:
        meas_det, ax_det = np.where(vf_r, rec_ref, 0.0), axes
        row["det_n"] = 0
        row["det_pitch_x_mm"] = round(float(abs(xmm[1] - xmm[0])), 3)
        row["det_pitch_y_mm"] = round(float(abs(ymm[1] - ymm[0])), 3)
        row["det_pts"] = int(vf_r.sum())
        row["det_fill_min"] = ""
        setflag("detgrid_fallback")
    else:
        row["det_n"] = DET_N
        # block sizes alternate floor/ceil, so the pitch comes from the overall centre span
        px = float(abs(ax_det[1][-1] - ax_det[1][0]) / (DET_N - 1))
        py = float(abs(ax_det[0][-1] - ax_det[0][0]) / (DET_N - 1))
        row["det_pitch_x_mm"] = round(px, 3)
        row["det_pitch_y_mm"] = round(py, 3)
        row["det_pts"] = int(np.isfinite(meas_det).sum())
        tol = DET_PITCH_TOL
        if not ((1 - tol) * DET_NOMINAL_PITCH_MM <= px <= (1 + tol) * DET_NOMINAL_PITCH_MM
                and (1 - tol) * DET_NOMINAL_PITCH_MM <= py <= (1 + tol) * DET_NOMINAL_PITCH_MM):
            setflag("det_pitch_off")
        # 2x2 factorial controls: evaluation-side sampling x dose source
        #   gamma_plan     : measured blocks vs TPS map (fine grid)        <- main result
        #   gamma_plan_det : measured blocks vs TPS map (detector grid)
        #   gamma_DCM      : measured blocks vs DCM (fine grid)
        #   gamma_DCM_det  : measured blocks vs DCM (detector grid)
        # all share the geom of the measured map, so block boundaries are identical
        cmp_det, _, fill_c = block_with_edges(rec_cmp, geom, DET_N)
        dcm_masked = np.where(covered, dcm_on_grid, np.nan)   # keep uncovered zeros out of the block average
        dcm_det, _, fill_d = block_with_edges(dcm_masked, geom, DET_N)
        cmp_det_g = np.where(np.isfinite(cmp_det), cmp_det, 0.0) if cmp_det is not None else None
        dcm_det_g = np.where(np.isfinite(dcm_det), dcm_det, 0.0) if dcm_det is not None else None
        if cmp_det_g is None or dcm_det_g is None:
            setflag("det_samegrid_failed")
        # the shared geom fixes the boundaries but not which pixels are valid
        # inside each block; fill_r vs fill_c/fill_d shows how strictly the
        # "same sampling" claim holds for gamma_plan_det/gamma_DCM_det
        area_r = np.outer(np.diff(geom[2]), np.diff(geom[3])).astype(np.float64)
        fill_r = float((det_counts / np.maximum(area_r, 1.0))[area_r > 0].min()) if (area_r > 0).any() else None
        row["det_fill_meas"] = round(fill_r, 3) if fill_r is not None else ""
        fm = [f for f in (fill_c, fill_d) if f is not None]
        row["det_fill_min"] = round(min(fm), 3) if fm else ""
        row["det_fill_gap"] = (round(fill_r - min(fm), 3)
                               if (fill_r is not None and fm) else "")
        if fm and min(fm) < DET_FILL_MIN_FLAG:
            setflag("low_det_fill")   # worst block mostly invalid; its average is not representative
        # aperture-smoothing control: kernel size from pitch/pixel spacing in
        # each direction (anisotropic), applied on the fine grid
        pix_x = abs(float(xmm[1] - xmm[0]))
        pix_y = abs(float(ymm[1] - ymm[0]))
        ap_x = max(int(round(px / pix_x)), 1) if pix_x > 0 else 9
        ap_y = max(int(round(py / pix_y)), 1) if pix_y > 0 else ap_x
        row["aperture_x_px"] = ap_x
        row["aperture_y_px"] = ap_y
        tps_aps = nan_uniform_filter(np.where(vf_c, rec_cmp, np.nan), (ap_y, ap_x))
        dcm_aps = nan_uniform_filter(np.where(covered, dcm_on_grid, np.nan), (ap_y, ap_x))
        tps_aps_g = np.nan_to_num(tps_aps, nan=0.0)
        dcm_aps_g = np.nan_to_num(dcm_aps, nan=0.0)
    # invalid detector points become 0 and are removed by the low-dose cutoff
    meas_g = np.where(np.isfinite(meas_det), meas_det, 0.0)

    # ---- 7. gamma ----
    dp_rep, dd_rep = float(rep["dp_rep"]), float(rep["dd_rep"])
    todo = {}
    if (dp_rep, dd_rep) != (3.0, 3.0):
        todo[(dp_rep, dd_rep)] = "rep"
    todo[(3.0, 3.0)] = "33"
    main_tag = "rep" if (dp_rep, dd_rep) != (3.0, 3.0) else "33"

    g_maps = {}
    for (dp, dd), tag in todo.items():
        # default direction, as configured in myQA: measured = reference,
        # planned = evaluation; the reversed control below checks this choice
        pa, ma, gA = run_gamma(ax_det, meas_g, axes, cmp0, dp, dd, gnorm, cutoff_pct)
        row[f"g_plan_{tag}"] = round(pa, 2)
        row[f"g_plan_{tag}_mean"] = round(ma, 3)
        # gamma_plan_det: the evaluation map is block-averaged too, so block averaging alone is not a confounder
        if cmp_det_g is not None:
            pat, mat, _ = run_gamma(ax_det, meas_g, ax_det, cmp_det_g, dp, dd, gnorm, cutoff_pct)
            row[f"g_plan_det_{tag}"] = round(pat, 2)
            row[f"g_plan_det_{tag}_mean"] = round(mat, 3)

        # reversed control: TPS as reference, measured map as evaluation
        if do_ab:
            p2, m2, _ = run_gamma(axes, cmp0, ax_det, meas_g, dp, dd, gnorm, cutoff_pct)
            row[f"g_plan_{tag}_dirT"] = round(p2, 2)
            row[f"g_plan_{tag}_dirT_mean"] = round(m2, 3)

        # gamma_DCM: measured vs DCM
        pb, mb, gB = run_gamma(ax_det, meas_g, axes, dcm_on_grid, dp, dd, gnorm, cutoff_pct)
        row[f"g_DCM_{tag}"] = round(pb, 2)
        row[f"g_DCM_{tag}_mean"] = round(mb, 3)
        # gamma_DCM_det: the DCM is block-averaged to 32x32 as well; if it moves
        # towards gamma_plan, the low gamma_DCM was mainly sampling asymmetry
        if dcm_det_g is not None:
            pdt, mdt, _ = run_gamma(ax_det, meas_g, ax_det, dcm_det_g, dp, dd, gnorm, cutoff_pct)
            row[f"g_DCM_det_{tag}"] = round(pdt, 2)
            row[f"g_DCM_det_{tag}_mean"] = round(mdt, 3)
        # aperture-smoothing control (fine grid retained): gamma_plan_sm / gamma_DCM_sm
        if tps_aps_g is not None:
            pas_, mas_, _ = run_gamma(ax_det, meas_g, axes, tps_aps_g, dp, dd, gnorm, cutoff_pct)
            row[f"g_plan_sm_{tag}"] = round(pas_, 2)
            row[f"g_plan_sm_{tag}_mean"] = round(mas_, 3)
        if dcm_aps_g is not None:
            pbs_, mbs_, _ = run_gamma(ax_det, meas_g, axes, dcm_aps_g, dp, dd, gnorm, cutoff_pct)
            row[f"g_DCM_sm_{tag}"] = round(pbs_, 2)
            row[f"g_DCM_sm_{tag}_mean"] = round(mbs_, 3)
        g_maps[tag] = (gA, gB)

    # TPS self-consistency at 3%/3 mm: recovered TPS map vs DCM. Both are TPS
    # dose, so a low value points to the registration residual or DCM resolution
    ps, ms, _ = run_gamma(axes, cmp0, axes, dcm_on_grid, 3.0, 3.0, gnorm, cutoff_pct)
    row["g_Self_33"] = round(ps, 2)
    row["g_Self_33_mean"] = round(ms, 3)

    # ---- 8. Deviation from the reported values ----
    row["d_plan_pp"] = round(row[f"g_plan_{main_tag}"] - rep["rep_pass"], 2)
    row["d_DCM_pp"] = round(row[f"g_DCM_{main_tag}"] - rep["rep_pass"], 2)
    if f"g_DCM_det_{main_tag}" in row:
        row["d_DCM_det_pp"] = round(row[f"g_DCM_det_{main_tag}"] - rep["rep_pass"], 2)
    if f"g_plan_det_{main_tag}" in row:
        row["d_plan_det_pp"] = round(row[f"g_plan_det_{main_tag}"] - rep["rep_pass"], 2)
        row["d_ds_plan_pp"] = round(row[f"g_plan_det_{main_tag}"] - row[f"g_plan_{main_tag}"], 2)   # downsampling effect (plan evaluation)
    # d_ds_DCM_pp / d_src_det_pp need both gamma_plan_det and gamma_DCM_det; no implicit
    # assumption that the two succeed together
    if f"g_plan_det_{main_tag}" in row and f"g_DCM_det_{main_tag}" in row:
        row["d_ds_DCM_pp"] = round(row[f"g_DCM_det_{main_tag}"] - row[f"g_DCM_{main_tag}"], 2)  # downsampling effect (DCM evaluation)
        row["d_src_det_pp"] = round(row[f"g_DCM_det_{main_tag}"] - row[f"g_plan_det_{main_tag}"], 2)  # source effect (under downsampling)
    if f"g_plan_sm_{main_tag}" in row and f"g_DCM_sm_{main_tag}" in row:
        row["d_plan_sm_pp"] = round(row[f"g_plan_sm_{main_tag}"] - rep["rep_pass"], 2)
        row["d_DCM_sm_pp"] = round(row[f"g_DCM_sm_{main_tag}"] - rep["rep_pass"], 2)
        row["d_ap_plan_pp"] = round(row[f"g_plan_sm_{main_tag}"] - row[f"g_plan_{main_tag}"], 2)   # aperture effect (plan evaluation)
        row["d_ap_DCM_pp"] = round(row[f"g_DCM_sm_{main_tag}"] - row[f"g_DCM_{main_tag}"], 2)   # aperture effect (DCM evaluation) - key
        row["d_src_sm_pp"] = round(row[f"g_DCM_sm_{main_tag}"] - row[f"g_plan_sm_{main_tag}"], 2)  # source effect (under aperture matching)
    if do_ab:
        row["d_plan_dirT_pp"] = round(row[f"g_plan_{main_tag}_dirT"] - rep["rep_pass"], 2)
    gate = GATE_PP_DIFF_STRICT if dp_rep <= 3.5 else GATE_PP_DIFF_LOOSE
    if abs(row["d_plan_pp"]) > gate:
        setflag(f"d_plan_gt_{gate:g}pp")

    # ---- 9. QA figure (flagged cases, or all cases with save_qa) ----
    if save_qa or row["flag"]:
        qa_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r'[\\/:*?"<>|]', "_", pdf_path.stem[:40])
        gA, gB = g_maps.get("rep", g_maps.get("33", (None, None)))
        gamma_map_png(gA, qa_dir / f"{stem}_gplan.png")
        gamma_map_png(gB, qa_dir / f"{stem}_gDCM.png")
        side = np.concatenate([
            np.nan_to_num(rec_ref, nan=0.0), np.nan_to_num(rec_cmp, nan=0.0), dcm_on_grid], axis=1)
        t = np.clip(side / max(gnorm, 1.0), 0, 1)
        img8 = np.stack([t * 255, t * 255, 255 * (1 - t)], -1).astype(np.uint8)
        Image.fromarray(img8).save(qa_dir / f"{stem}_maps.png")

    # self-check: recompute geom from rec_ref and require an exact match, so any
    # change to the valid region after to_detector() fails loudly here
    if geom is not None:
        _ret = to_detector(rec_ref, xmm, ymm)
        geom_re = _ret[3] if _ret is not None else None
        if geom_re is None:
            # same input but None returned: report it directly instead of
            # letting the subscript raise a TypeError that hides the cause
            raise RuntimeError("geom self-check failed: to_detector returned None on recomputation (input unchanged but the result differs)")
        same = (geom_re[0] == geom[0] and geom_re[1] == geom[1]
                and np.array_equal(geom_re[2], geom[2]) and np.array_equal(geom_re[3], geom[3]))
        if not same:
            raise RuntimeError("geom block-boundary self-check failed: the valid region of rec_ref changed after to_detector")

    row["seconds"] = round(time.time() - t0, 1)
    return row


# ---------------------------------------------------------------- pairing --------
def pair_files(root: Path):
    """Pair PDFs with same-named DCMs sitting in the same parent directory; returns (pairs, skipped)."""
    pdfs = [p for p in root.rglob("*")
            if p.suffix.lower() == ".pdf" and not p.name.startswith("~$") and p.is_file()]
    dcms = [p for p in root.rglob("*")
            if p.suffix.lower() in (".dcm", ".dicm") and not p.name.startswith("~$") and p.is_file()]
    dcm_by_stem = {}
    for d in dcms:
        dcm_by_stem.setdefault(d.stem.lower(), []).append(d)
    pdf_stems = {p.stem.lower() for p in pdfs}
    pairs, skipped = [], []
    for p in pdfs:
        key = p.stem.lower()
        if key not in dcm_by_stem:
            skipped.append((str(p), "no DCM with the same name found"))
            continue
        lst = dcm_by_stem[key]
        # accept only a DCM in the same parent directory (no cross-batch matches)
        same_dir = [d for d in lst if d.parent.name == p.parent.name]
        if not same_dir:
            # same-named DCM only elsewhere: treat as unmatched
            cand_dirs = ";".join(sorted({d.parent.name for d in lst}))
            skipped.append((str(p),
                            f"DCM with the same name but a different parent directory (pdf_dir={p.parent.name}; dcm_dirs={cand_dirs})"))
            continue
        chosen = same_dir[0]
        if len(lst) > 1:
            skipped.append((str(p), f"DCMs with the same name: {len(lst)} found, using the one in the same parent directory"))
        pairs.append((p, chosen, "stem+dir"))
    for d in dcms:
        if d.stem.lower() not in pdf_stems:
            skipped.append((str(d), "no PDF with the same name found"))
    return pairs, skipped


# ---------------------------------------------------------------- main program ---
CSV_COLS = ["case_id", "rel_pdf", "rel_dcm", "status", "error", "flag", "seconds",
            "rep_dp_rep", "rep_dd_rep", "rep_thresh_pct", "rep_rep_pass", "rep_rep_avg_gamma",
            "rep_dmax_cgy", "est_dmin_cgy", "rep_poi_ref_max", "rep_poi_cmp_max",
            "rep_poi_ref_cursor", "rep_poi_cmp_cursor",
            "ref_valid_pct", "cmp_valid_pct",
            "rec_ref_cursor", "dev_ref_cursor", "rec_cmp_cursor", "dev_cmp_cursor",
            "dcm_max_cgy", "dcm_pixels", "dcm_spacing_mm", "dcm_vs_poi_max",
            "reg_ncc", "reg_sx", "reg_sy", "reg_du", "reg_dv", "reg_coverage",
            "dose_bias", "dose_sigma", "dose_sigma_smooth", "dose_sigma_grad", "sat_pct",
            "uncov_pts",
            "det_n", "det_pitch_x_mm", "det_pitch_y_mm", "det_pts",
            "det_fill_min", "det_fill_meas", "det_fill_gap",
            "aperture_x_px", "aperture_y_px",
            "g_plan_rep", "g_plan_rep_mean", "g_plan_rep_dirT", "g_plan_rep_dirT_mean", "g_DCM_rep", "g_DCM_rep_mean",
            "g_DCM_det_rep", "g_DCM_det_rep_mean", "g_plan_det_rep", "g_plan_det_rep_mean",
            "g_plan_sm_rep", "g_plan_sm_rep_mean", "g_DCM_sm_rep", "g_DCM_sm_rep_mean",
            "g_plan_33", "g_plan_33_mean", "g_plan_33_dirT", "g_plan_33_dirT_mean", "g_DCM_33", "g_DCM_33_mean",
            "g_DCM_det_33", "g_DCM_det_33_mean", "g_plan_det_33", "g_plan_det_33_mean",
            "g_plan_sm_33", "g_plan_sm_33_mean", "g_DCM_sm_33", "g_DCM_sm_33_mean",
            "g_Self_33", "g_Self_33_mean",
            "d_plan_pp", "d_DCM_pp", "d_DCM_det_pp", "d_plan_det_pp", "d_plan_sm_pp", "d_DCM_sm_pp",
            "d_ds_plan_pp", "d_ds_DCM_pp", "d_src_det_pp",
            "d_ap_plan_pp", "d_ap_DCM_pp", "d_src_sm_pp", "d_plan_dirT_pp"]


def _to_float(v):
    """Values read back from the CSV may be strings; converted before summarising"""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v).strip())
    except Exception:
        return None


def main():
    global DET_FILL_MIN_FLAG
    ap = argparse.ArgumentParser(description="Batch gamma analysis of myQA verification report PDFs with DCM comparison")
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="root directory (containing PDF/DCM in subdirectories)")
    ap.add_argument("--out", default=None, help="output directory (default: <root>/_gamma_output2026)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pairs (for trial runs)")
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"), default=None,
                    help="shard: process shard I of N (several windows can run in parallel)")
    ap.add_argument("--all-qa", action="store_true", help="save QA figures for all cases (default: flagged cases only)")
    ap.add_argument("--no-ab", action="store_true", help="skip the reversed-reference gamma (saves about half the time)")
    ap.add_argument("--fill-thresh", type=float, default=DET_FILL_MIN_FLAG,
                    help="low_det_fill threshold (worst-block valid-pixel fraction); "
                         "calibrate from the det_fill_min distribution if needed; 0 disables")
    ap.add_argument("--debug", action="store_true", help="print the full traceback on ERROR")
    args = ap.parse_args()
    DET_FILL_MIN_FLAG = float(args.fill_thresh)   # command-line override of the provisional threshold

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"root directory does not exist: {root}")
    out_dir = Path(args.out) if args.out else root / "_gamma_output2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "qa_images"
    shard_i, shard_n = (args.shard if args.shard else (0, 1))
    csv_path = out_dir / (f"results_shard{shard_i}of{shard_n}.csv" if shard_n > 1 else "results.csv")
    skipped_path = out_dir / ("skipped.csv" if shard_n == 1 else f"skipped_shard{shard_i}of{shard_n}.csv")

    pairs, skipped = pair_files(root)
    with open(skipped_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "reason"])
        w.writerows(skipped)

    if shard_n > 1:
        pairs = [p for i, p in enumerate(pairs) if i % shard_n == shard_i]
    if args.limit:
        pairs = pairs[: args.limit]

    # Resume: only OK cases are skipped; ERROR cases are retried
    all_rows = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))
    done = {r["rel_pdf"] for r in all_rows if r.get("status") == "OK"}
    pairs = [p for p in pairs if str(p[0].relative_to(root)) not in done]

    print(f"Root: {root}")
    print(f"Paired {len(pairs) + len(done)} pairs | skipped {len(skipped)} unmatched files -> {skipped_path.name}")
    print(f"To process this run: {len(pairs)} cases | output: {csv_path}")
    print(f"Detector grid: {DET_N}x{DET_N} (nominal pitch {DET_NOMINAL_PITCH_MM} mm) | reversed control: {'off' if args.no_ab else 'on'}")
    print(f"Gates: NCC>{GATE_NCC} coverage>{GATE_COVERAGE} |d_plan|<=strict {GATE_PP_DIFF_STRICT}/loose {GATE_PP_DIFF_LOOSE}pp")
    print("-" * 70)

    def flush():
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

    t_start = time.time()
    n_ok = n_err = 0
    for i, (pdf_path, dcm_path, _method) in enumerate(pairs):
        rel = str(pdf_path.relative_to(root))
        cid = f"{shard_i}x{i:04d}" if shard_n > 1 else f"{i:04d}"
        t0 = time.time()
        print(f"[{i + 1}/{len(pairs)}] {rel}")
        try:
            row = process_case(pdf_path, dcm_path, qa_dir, args.all_qa, do_ab=not args.no_ab)
            row.update({"case_id": cid, "rel_pdf": rel,
                        "rel_dcm": str(dcm_path.relative_to(root)),
                        "status": "OK", "error": ""})
            n_ok += 1
            ga = row.get("g_plan_rep", row.get("g_plan_33"))
            extra = f" dirT={row.get('g_plan_rep_dirT', row.get('g_plan_33_dirT'))}%" if not args.no_ab else ""
            print(f"    OK  criteria {row['rep_dp_rep']}%/{row['rep_dd_rep']} mm reported={row['rep_rep_pass']}% | "
                  f"g_plan={ga}%{extra} g_DCM={row.get('g_DCM_rep', row.get('g_DCM_33'))}% | "
                  f"pitch={row.get('det_pitch_x_mm')}mm pts={row.get('det_pts')} | "
                  f"NCC={row['reg_ncc']} [{row['seconds']}s]"
                  f"{' FLAG:' + row['flag'] if row['flag'] else ''}")
        except Exception as e:
            row = {"case_id": cid, "rel_pdf": rel, "rel_dcm": str(dcm_path.relative_to(root)),
                   "status": "ERROR", "error": f"{type(e).__name__}: {e}", "flag": "",
                   "seconds": round(time.time() - t0, 1)}
            n_err += 1
            print(f"    ERROR  {row['error']}")
            if args.debug:
                traceback.print_exc()
        all_rows = [r for r in all_rows if r.get("rel_pdf") != rel] + [row]
        if (i + 1) % WRITE_EVERY == 0 or i == len(pairs) - 1:
            flush()

    # ---- summary ----
    ok_rows = [r for r in all_rows if r.get("status") == "OK"]
    lines = ["", "=" * 70, f"Done: OK={n_ok} ERROR={n_err} | total {(time.time() - t_start) / 60:.1f} min"]
    if n_ok + n_err:
        el = (time.time() - t_start) / max(n_ok + n_err, 1)
        lines.append(f"mean per case {el:.0f} s | remaining {(len(pairs) - n_ok - n_err) * el / 60:.0f} min")
    if ok_rows:
        def stat(key):
            vals = [v for v in (_to_float(r.get(key)) for r in ok_rows) if v is not None]
            if not vals:
                return "no data"
            return (f"n={len(vals)} median {np.median(vals):.2f} "
                    f"mean {np.mean(vals):.2f} σ{np.std(vals):.2f}")
        lines.append(f"gamma_plan - reported deviation (pp): {stat('d_plan_pp')}")
        lines.append(f"gamma_DCM - reported deviation (pp): {stat('d_DCM_pp')}")
        lines.append(f"gamma_DCM,det (detector-grid DCM) - reported deviation (pp): {stat('d_DCM_det_pp')}")
        lines.append(f"gamma_plan,det (detector-grid plan) - reported deviation (pp): {stat('d_plan_det_pp')}")
        lines.append("--- Factorial decomposition A: downsampling to 32x32 (coarser evaluation side) ---")
        lines.append(f"  downsampling effect (plan) gamma_plan,det - gamma_plan : {stat('d_ds_plan_pp')}")
        lines.append(f"  downsampling effect (DCM)  gamma_DCM,det - gamma_DCM : {stat('d_ds_DCM_pp')}")
        lines.append(f"  source effect (under downsampling) gamma_DCM,det - gamma_plan,det: {stat('d_src_det_pp')}")
        lines.append("--- Factorial decomposition B: aperture smoothing (fine grid retained); tests the sampling-asymmetry hypothesis ---")
        lines.append(f"  aperture effect (plan) gamma_plan,sm - gamma_plan : {stat('d_ap_plan_pp')}")
        lines.append(f"  aperture effect (DCM)  gamma_DCM,sm - gamma_DCM : {stat('d_ap_DCM_pp')}   <- key")
        lines.append(f"  source effect (under aperture matching) gamma_DCM,sm - gamma_plan,sm: {stat('d_src_sm_pp')}")
        lines.append(f"  source-effect robustness: downsampling {stat('d_src_det_pp')} | aperture matching {stat('d_src_sm_pp')}"
                     "  (close -> robust estimate; large difference -> downsampling distorts the source effect)")
        # judge by the interaction (d_ap_DCM - d_ap_plan), not by d_ap_DCM alone:
        # smoothing moves both evaluation paths, only the difference says
        # whether aperture matching closes the gamma_DCM - gamma_plan gap
        dA = [v for v in (_to_float(r.get('d_ap_plan_pp')) for r in ok_rows) if v is not None]
        dB = [v for v in (_to_float(r.get('d_ap_DCM_pp')) for r in ok_rows) if v is not None]
        ga = [v for v in (_to_float(r.get('d_plan_pp')) for r in ok_rows) if v is not None]
        gb = [v for v in (_to_float(r.get('d_DCM_pp')) for r in ok_rows) if v is not None]
        if dA and dB and ga and gb:
            deficit = np.mean(gb) - np.mean(ga)       # gamma_DCM - gamma_plan; negative means gamma_DCM is lower
            closure = np.mean(dB) - np.mean(dA)       # interaction; positive means aperture matching narrows the gap
            pct = closure / abs(deficit) * 100 if abs(deficit) > 1e-6 else float("nan")
            if abs(deficit) < 0.5:
                verdict = "the gamma_DCM - gamma_plan gap is <0.5 pp, so this criterion does not apply"
            elif pct >= 50:
                verdict = (f"aperture matching removes {closure:+.2f}pp({pct:.0f}% of the gap) "
                           f"-> sampling asymmetry is the main cause of the gamma_DCM deficit")
            elif -20 <= pct < 50:
                verdict = (f"aperture matching barely changes the gap ({closure:+.2f}pp, {pct:.0f}%) "
                           f"-> the gamma_DCM deficit is unrelated to sampling and comes from other differences between the DCM and the PDF map")
            else:
                verdict = (f"aperture matching widens the gap by {abs(closure):.2f}pp({pct:.0f}%) "
                           f"-> sampling asymmetry is not the cause; the fine sampling of the DCM was helping gamma_DCM, "
                           f"and smoothing exposes a larger real mismatch")
            lines.append(f"  verdict: gamma_DCM - gamma_plan={deficit:+.2f} pp | interaction (d_ap_DCM_pp - d_ap_plan_pp)={closure:+.2f}pp "
                         f"| {verdict}")
        if not args.no_ab:
            lines.append(f"gamma_plan reversed (TPS as reference) - reported deviation (pp): {stat('d_plan_dirT_pp')}")
        lines.append(f"mean gamma deviation (gamma_plan - reported): {stat('g_plan_33_mean')} | reported mean {stat('rep_rep_avg_gamma')}")
        lines.append(f"TPS self-consistency gamma (plan vs DCM) 3/3: {stat('g_Self_33')}")
        lines.append(f"detector pitch: x {stat('det_pitch_x_mm')} | y {stat('det_pitch_y_mm')}")
        lines.append(f"valid detector points: {stat('det_pts')}")
        lines.append(f"block fill: evaluation-side minimum {stat('det_fill_min')} | measured-side minimum {stat('det_fill_meas')} "
                     f"| difference {stat('det_fill_gap')}  (if the evaluation-side minimum is generally <0.6, the low_det_fill gate "
                     "loses discrimination; calibrate --fill-thresh from the 5th percentile of this column, or pass 0 to disable)")
        flagged = [r["rel_pdf"] for r in ok_rows if r.get("flag")]
        lines.append(f"flagged cases: {len(flagged)}" + (": " + "; ".join(flagged[:8]) if flagged else ""))
        errs = [f"{r['rel_pdf']}: {r['error']}" for r in all_rows if r.get("status") == "ERROR"]
        if errs:
            lines.append("ERROR details (first 5): " + " | ".join(errs[:5]))
    lines.append("=" * 70)

    # ---- extreme cases: |Delta_plan| > 5 pp ----
    EXTREME_THRESH = 5.0
    extreme = []
    for r in ok_rows:
        v = _to_float(r.get("d_plan_pp"))
        if v is not None and abs(v) > EXTREME_THRESH:
            extreme.append(r)
    if extreme:
        # keep only the key columns so that the table stays focused
        keep = ["case_id", "rel_pdf", "rel_dcm", "flag",
                "rep_dp_rep", "rep_dd_rep", "rep_rep_pass",
                "g_plan_33", "g_plan_rep", "d_plan_pp", "d_DCM_pp",
                "dcm_max_cgy", "rep_poi_cmp_max", "dcm_vs_poi_max",
                "reg_ncc", "reg_coverage",
                "det_fill_min", "det_fill_meas", "det_fill_gap",
                "dose_sigma", "sat_pct", "g_Self_33",
                "est_dmin_cgy", "rep_dmax_cgy"]
        keep = [c for c in keep if c in CSV_COLS]
        extreme_sorted = sorted(extreme,
                                key=lambda r: _to_float(r.get("d_plan_pp")) or 0.0)
        extreme_path = out_dir / "extreme_cases_dplan_gt5pp.csv"
        with open(extreme_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
            w.writeheader()
            w.writerows(extreme_sorted)
        pos = sum(1 for r in extreme if (_to_float(r.get("d_plan_pp")) or 0) > 0)
        lines.append("")
        lines.append(f"[extreme cases |d_plan|>{EXTREME_THRESH} pp] total {len(extreme)} cases "
                     f"(high by {pos} cases, low by {len(extreme) - pos} cases)")
        lines.append(f"  output: {extreme_path}")
        for r in extreme_sorted:
            lines.append(f"  - {r['case_id']} {r['rel_pdf']}: "
                         f"reported={r['rep_rep_pass']}% recovered gamma_plan={r.get('g_plan_33') or r.get('g_plan_rep')}% "
                         f"d_plan={r['d_plan_pp']}pp flag={r['flag'] or '-'}")
    report = "\n".join(lines)
    print(report)
    sname = f"summary_shard{shard_i}of{shard_n}.txt" if shard_n > 1 else "summary.txt"
    (out_dir / sname).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
