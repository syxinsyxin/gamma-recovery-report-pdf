# -*- coding: utf-8 -*-
"""RT Dose (DCM) handling: coronal-plane extraction, detector-grid resampling, NCC registration.

Two uses in the closed-loop validation:
1. resample the measurement plane (coronal) of a DICOM RT Dose onto the
   32x32 / pitch 7.619 mm MatriXX detector grid;
2. register a report/recovered map to the DCM plane (manuscript Secs. 2.2/2.3):
   a 2D search over (slice position, in-plane shift), maximising the normalized
   cross-correlation.

Conventions:
- patient coordinates: x = pos[0] + col*col_spacing, y = pos[1] + row*row_spacing,
  z = pos[2] + GridFrameOffsetVector (frames stacked along z, standard axial
  acquisition). RT Dose with a non-axial ImageOrientationPatient
  (!= (1,0,0,0,1,0)) must be resliced first.
- a "coronal plane" means a fixed y (A-P) position -> a (z, x) plane.
- row/col orientation of report images vs DCM planes may differ (transpose or
  flips); align_dcm_to_detector_map takes an orientation spec
  ('normal'|'T'|'UD'|'LR', combinable with '+').

Requires numpy, scipy; load_rt_dose additionally needs pydicom.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

# ---------------------------------------------------------------------------
# DICOM input
# ---------------------------------------------------------------------------

def load_rt_dose(path):
    """Read a DICOM RT Dose; returns geometry + dose array (dose = pixel_array * DoseGridScaling)."""
    import pydicom  # deferred import: pip install pydicom

    ds = pydicom.dcmread(path)
    dose = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    return {
        "dose": dose,
        "row_spacing": float(ds.PixelSpacing[0]),
        "col_spacing": float(ds.PixelSpacing[1]),
        "position": [float(v) for v in ds.ImagePositionPatient],
        "offsets": np.asarray([float(v) for v in ds.GridFrameOffsetVector], dtype=float),
        "units": str(getattr(ds, "DoseUnits", "")),
        "for_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
    }


# ---------------------------------------------------------------------------
# Plane extraction and resampling
# ---------------------------------------------------------------------------

def sample_coronal_slice(dcm, y_mm):
    """Bilinearly interpolate a coronal plane at y=y_mm (mm, patient coordinates).

    Returns
    -------
    plane : (nz, nx) 2D dose plane
    z, x  : physical coordinates of the two axes (mm)
    """
    dose = dcm["dose"]
    rs = dcm["row_spacing"]
    pos = dcm["position"]
    t = (float(y_mm) - pos[1]) / rs
    i0 = int(np.floor(t))
    w = t - i0
    if i0 < 0 or i0 + 1 >= dose.shape[1]:
        raise ValueError(f"y_mm={y_mm} outside the y range of the DCM")
    plane = (1.0 - w) * dose[:, i0, :] + w * dose[:, i0 + 1, :]
    z = pos[2] + dcm["offsets"]
    x = pos[0] + np.arange(dose.shape[2]) * dcm["col_spacing"]
    return plane, z, x


def sample_to_detector_grid(plane, z, x, pitch=7.619, n=32, center=None, ss=8):
    """Resample a dose plane onto the n x n detector grid (chamber-area block average, supersampled).

    Each chamber cell covers pitch x pitch; ss x ss sub-samples inside the cell
    are averaged (= area weighting). Sub-samples outside the plane are NaN and
    aggregated with nanmean; a fully outside cell stays NaN.

    Returns
    -------
    det : (n, n) array
    center : (zc, xc) grid center actually used
    """
    if center is None:
        center = (0.5 * (z[0] + z[-1]), 0.5 * (x[0] + x[-1]))
    zc, xc = float(center[0]), float(center[1])
    half = pitch * n / 2.0
    zf = zc - half + (np.arange(n * ss) + 0.5) * (pitch / ss)
    xf = xc - half + (np.arange(n * ss) + 0.5) * (pitch / ss)

    dz = z[1] - z[0]
    dx = x[1] - x[0]
    zi = (zf - z[0]) / dz
    xi = (xf - x[0]) / dx
    ZZ, XX = np.meshgrid(zi, xi, indexing="ij")
    vals = map_coordinates(
        plane, [ZZ.ravel(), XX.ravel()], order=1, mode="constant", cval=np.nan
    ).reshape(n * ss, n * ss)
    with np.errstate(invalid="ignore"):
        det = np.nanmean(vals.reshape(n, ss, n, ss), axis=(1, 3))
    return det, (zc, xc)


def block_average(dose2d, factor):
    """Block-average a fine-grid map onto a coarser grid (factor x factor; must divide evenly)."""
    a = np.asarray(dose2d, dtype=float)
    n0 = a.shape[0] // factor
    n1 = a.shape[1] // factor
    a = a[: n0 * factor, : n1 * factor]
    return a.reshape(n0, factor, n1, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# NCC and alignment search
# ---------------------------------------------------------------------------

def ncc(a, b):
    """Normalized cross-correlation (Pearson r), skipping NaNs on either side."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def align_dcm_to_detector_map(
    dcm,
    target32,
    pitch=7.619,
    n=32,
    y_candidates=None,
    margin_mm=8.0,
    ss=4,
    orientation="normal",
):
    """2D search over (coronal slice y, in-plane shift) for the DCM sampling with max NCC vs target32.

    Each candidate slice is sampled once on a fine grid covering n+2*pad
    chambers (pad = ceil(margin_mm/pitch), sub-sample = pitch/ss); an n x n
    window is then slid over it, block-averaged to n x n (NaN-aware) and scored
    with NCC. Shift resolution = pitch/ss; the final det32 is resampled at the
    best (y, shift) with ss=8.

    Parameters
    ----------
    dcm : dict from load_rt_dose (or any dict with the same keys, for synthetic tests)
    target32 : (n, n) registration target (e.g. the detector-grid version of the recovered plan map)
    y_candidates : coronal y positions to try (mm). Default: every 1.0 mm over the DCM y range.
    margin_mm : search radius for the in-plane shift (mm).
    ss : sub-sampling factor for the search (shift step = pitch/ss). Keep <= 4 to bound memory/time.
    orientation : orientation correction of target32 relative to the DCM plane,
                  'normal' | 'T' | 'UD' | 'LR' and combinations such as 'T+UD'.
                  Depends on the report's row/col convention; verify on a few cases first.

    Returns
    -------
    dict(best_y_mm, best_shift_dz, best_shift_dx, ncc, det32, n_slices_evaluated)
    """
    t = np.asarray(target32, dtype=float)
    t = _apply_orientation(t, orientation)

    pos = dcm["position"]
    rs = dcm["row_spacing"]
    y_lo = pos[1]
    y_hi = pos[1] + (dcm["dose"].shape[1] - 1) * rs
    if y_candidates is None:
        y_candidates = np.arange(y_lo, y_hi + 1e-6, 1.0)

    pad = int(np.ceil(margin_mm / pitch))
    n_fine = (n + 2 * pad) * ss  # fine-grid points per side (1 sub-pixel = pitch/ss)
    win = n * ss                 # sliding-window side (in sub-pixels)

    best = None
    for y_mm in y_candidates:
        try:
            plane, z, x = sample_coronal_slice(dcm, y_mm)
        except ValueError:
            continue
        if z.size < 2 or x.size < 2:
            continue
        # fine grid centred on the DCM plane centre, spanning n+2*pad chambers
        zc0 = 0.5 * (z[0] + z[-1])
        xc0 = 0.5 * (x[0] + x[-1])
        half_ext = (n_fine * (pitch / ss)) / 2.0
        zf = zc0 - half_ext + (np.arange(n_fine) + 0.5) * (pitch / ss)
        xf = xc0 - half_ext + (np.arange(n_fine) + 0.5) * (pitch / ss)
        dz = z[1] - z[0]
        dx = x[1] - x[0]
        zi = (zf - z[0]) / dz
        xi = (xf - x[0]) / dx
        ZZ, XX = np.meshgrid(zi, xi, indexing="ij")
        P = map_coordinates(
            plane, [ZZ.ravel(), XX.ravel()], order=1, mode="constant", cval=np.nan
        ).reshape(n_fine, n_fine)

        n_off = n_fine - win
        for sy in range(n_off + 1):
            for sx in range(n_off + 1):
                blk = P[sy : sy + win, sx : sx + win]
                fin = np.isfinite(blk)
                if not fin.any():
                    continue
                # NaN-aware block average (avoids RuntimeWarning from nanmean on all-NaN cells)
                nf = fin.reshape(n, ss, n, ss).sum(axis=(1, 3))
                sm = np.where(fin, blk, 0.0).reshape(n, ss, n, ss).sum(axis=(1, 3))
                with np.errstate(invalid="ignore", divide="ignore"):
                    det = np.where(nf > 0, sm / np.maximum(nf, 1), np.nan)
                c = ncc(det, t)
                if np.isfinite(c) and (best is None or c > best["ncc"]):
                    shift_dz = (sy - pad * ss) * (pitch / ss)  # relative to centre, mm
                    shift_dx = (sx - pad * ss) * (pitch / ss)
                    best = {
                        "best_y_mm": float(y_mm),
                        "best_shift_dz": float(shift_dz),
                        "best_shift_dx": float(shift_dx),
                        "ncc": float(c),
                        "n_slices_evaluated": len(y_candidates),
                    }
    if best is None:
        raise RuntimeError("registration failed: NCC could not be computed for any candidate slice")

    # resample the final detector-grid map at the best (y, shift) with finer resolution
    plane, z, x = sample_coronal_slice(dcm, best["best_y_mm"])
    zc = 0.5 * (z[0] + z[-1]) + best["best_shift_dz"]
    xc = 0.5 * (x[0] + x[-1]) + best["best_shift_dx"]
    det32, center_used = sample_to_detector_grid(
        plane, z, x, pitch=pitch, n=n, center=(zc, xc), ss=8
    )
    best["det32"] = det32
    best["center_used"] = center_used
    best["ncc_final"] = ncc(_apply_orientation(det32, "normal"), t)
    return best


def _apply_orientation(a, spec):
    a = np.asarray(a, dtype=float)
    spec = (spec or "normal").upper()
    for tok in spec.split("+"):
        tok = tok.strip()
        if tok in ("", "NORMAL"):
            pass
        elif tok == "T":
            a = a.T
        elif tok == "UD":
            a = a[::-1, :]
        elif tok == "LR":
            a = a[:, ::-1]
        else:
            raise ValueError(f"unknown orientation token: {tok}")
    return a


# ---------------------------------------------------------------------------
# Orientation auto-detection (align row/col conventions of the raw matrix vs recovered maps)
# ---------------------------------------------------------------------------

# All 8 dihedral transforms (transpose/flip combinations) cover every orientation
# mismatch that can occur between a CSV matrix and a report image. Each spec is
# a combination string understood by _apply_orientation.
ORIENTATION_SPECS = ("normal", "T", "UD", "LR", "T+UD", "UD+T", "UD+LR", "UD+LR+T")


def best_orientation(a, b, specs=ORIENTATION_SPECS):
    """Pick the dihedral transform of b that best matches a (max NCC).

    The row/col convention of the CSV-exported 32x32 chamber matrix relative to
    the report/recovered maps is unknown, so it is detected per case. The choice
    should be consistent across cases; check individual cases if it is not.

    Returns
    -------
    dict(orientation, ncc, transformed, scores={spec: ncc})
    """
    best = None
    scores = {}
    for s in specs:
        t = _apply_orientation(b, s)
        c = ncc(a, t)
        scores[s] = float(c) if np.isfinite(c) else float("nan")
        if np.isfinite(c) and (best is None or c > best["ncc"]):
            best = {"orientation": s, "ncc": float(c), "transformed": t}
    if best is None:
        raise RuntimeError("orientation detection failed: no transform gave a valid NCC")
    best["scores"] = scores
    return best


# ---------------------------------------------------------------------------
# Absolute dose-scale fit (QC)
# ---------------------------------------------------------------------------

def fit_dose_scale(measured, predicted, mask=None):
    """Least-squares fit measured ~= a*predicted + b; returns a, b, r, rmse.

    Used as a sanity check of the absolute dose scale (cursor/POI consistency):
    a clearly off-unity a or a large |b| indicates the scale recovery went wrong
    (one of the failure modes discussed in the manuscript).
    """
    m = np.asarray(measured, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    ok = np.isfinite(m) & np.isfinite(p)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool).ravel()
    if ok.sum() < 3:
        return {"a": np.nan, "b": np.nan, "r": np.nan, "rmse": np.nan, "n": int(ok.sum())}
    m, p = m[ok], p[ok]
    A = np.stack([p, np.ones_like(p)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - (a * p + b)
    r = np.corrcoef(m, p)[0, 1]
    return {
        "a": float(a),
        "b": float(b),
        "r": float(r),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "n": int(ok.sum()),
    }
