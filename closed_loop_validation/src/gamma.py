# -*- coding: utf-8 -*-
"""2D gamma analysis, matching the conventions of the manuscript (Sec. 2.6).

The reference dose sits on the coarse grid (here: the 32x32 detector matrix,
7.619 mm pitch); the evaluation dose is interpolated onto a finer grid
(ref_spacing / interp_factor) and the gamma minimum is searched there. Note the
asymmetry matters: a coarse map on the *evaluation* side systematically lowers
the pass rate.

Dose-difference normalisation:
- global (global_norm=True): delta-D = dd% of the chosen basis
  ('ref_max' | 'eval_max' | a fixed value via global_value);
- local (global_norm=False): delta-D = dd% of the reference dose at each point.

Low-dose cutoff: only reference points with dose >= cutoff enter the statistics
('ref_max' = percentage of the reference maximum, 'fixed' = absolute dose).

ref and eval must share the same origin and orientation (align them before
calling). Rows are y, columns are x, spacings in mm; dose units of both maps
must match (cGy vs Gy is up to the caller).

Requires numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def gamma_analysis(
    ref_dose,
    ref_spacing,
    eval_dose,
    eval_spacing,
    dd_percent=3.0,
    dta_mm=3.0,
    cutoff_percent=5.0,
    cutoff_basis="ref_max",
    cutoff_value=None,
    global_norm=True,
    global_basis="ref_max",
    global_value=None,
    interp_factor=10,
    max_gamma=2.0,
):
    """Compute a 2D gamma (coarse reference, evaluation interpolated to a fine grid).

    Parameters
    ----------
    ref_dose : (Nr, Nc) array
        Reference dose (typically the measurement / detector grid).
    ref_spacing : (dy, dx) in mm
    eval_dose : (Ne1, Ne2) array
        Evaluation dose (e.g. the plan dose, grid may differ).
    eval_spacing : (dy, dx) in mm
    dd_percent : dose-difference criterion (%)
    dta_mm : distance-to-agreement criterion (mm)
    cutoff_percent / cutoff_basis / cutoff_value :
        low-dose cutoff; 'ref_max' = % of reference maximum, 'fixed' = absolute
        dose via cutoff_value.
    global_norm / global_basis / global_value :
        dose-difference normalisation; 'ref_max' | 'eval_max' | 'fixed'
        (global_value in dose units).
    interp_factor : fine-grid refinement factor relative to the reference spacing.
    max_gamma : upper bound of the search; window radius = max_gamma * dta_mm.

    Returns
    -------
    dict:
        pass_rate           pass rate (%)
        gamma_map           (Nr, Nc) gamma values, NaN where not evaluated
        mean_gamma          mean gamma over evaluated points
        n_eval, n_pass      evaluated / passing point counts
        dd_abs, cutoff_abs  absolute thresholds actually used (for the record)
        params              echo of the input parameters
    """
    ref = np.asarray(ref_dose, dtype=float)
    ev = np.asarray(eval_dose, dtype=float)
    if ref.ndim != 2 or ev.ndim != 2:
        raise ValueError("ref_dose and eval_dose must be 2D arrays")
    rs = (float(ref_spacing[0]), float(ref_spacing[1]))
    es = (float(eval_spacing[0]), float(eval_spacing[1]))

    ref_max = float(np.nanmax(ref))
    ev_max = float(np.nanmax(ev))

    # low-dose cutoff
    if cutoff_basis == "ref_max":
        cutoff_abs = cutoff_percent / 100.0 * ref_max
    elif cutoff_basis == "fixed":
        if cutoff_value is None:
            raise ValueError("cutoff_basis='fixed' requires cutoff_value")
        cutoff_abs = float(cutoff_value)
    else:
        raise ValueError("cutoff_basis must be 'ref_max' or 'fixed'")

    # dose-difference normalisation
    if global_norm:
        if global_basis == "ref_max":
            dd_abs = dd_percent / 100.0 * ref_max
        elif global_basis == "eval_max":
            dd_abs = dd_percent / 100.0 * ev_max
        elif global_basis == "fixed":
            if global_value is None:
                raise ValueError("global_basis='fixed' requires global_value")
            # 2026-11-17: dd_abs used to be set to global_value itself, which diluted
            # the DD criterion by dd_percent (gamma degraded to a pure DTA test and a
            # noisy reference passed everything). Now matches pymedphys
            # global_normalisation semantics: dd_abs = dd% x global_value.
            dd_abs = dd_percent / 100.0 * float(global_value)
        else:
            raise ValueError("global_basis must be 'ref_max', 'eval_max' or 'fixed'")
    else:
        dd_abs = None  # per-point local normalisation

    # grid coordinates, origin fixed at 0
    fy = np.arange(ref.shape[0]) * rs[0]
    fx = np.arange(ref.shape[1]) * rs[1]
    ey = np.arange(ev.shape[0]) * es[0]
    ex = np.arange(ev.shape[1]) * es[1]

    margin = max_gamma * dta_mm
    y_max = max(fy[-1], ey[-1])
    x_max = max(fx[-1], ex[-1])
    fine_dy = min(rs[0], es[0]) / interp_factor
    fine_dx = min(rs[1], es[1]) / interp_factor
    # fine grid anchored at 0 with step = min(ref, eval) spacing / interp_factor, so
    # fine points coincide exactly with points of both input grids; otherwise a
    # spurious pure-distance gamma shows up at the reference points.
    my = int(np.ceil(margin / fine_dy))
    mx = int(np.ceil(margin / fine_dx))
    y_hi_steps = int(np.ceil((y_max + margin) / fine_dy))
    x_hi_steps = int(np.ceil((x_max + margin) / fine_dx))
    Fy = np.arange(-my, y_hi_steps + 1) * fine_dy
    Fx = np.arange(-mx, x_hi_steps + 1) * fine_dx

    interp = RegularGridInterpolator(
        (ey, ex), ev, method="linear", bounds_error=False, fill_value=np.nan
    )
    YY, XX = np.meshgrid(Fy, Fx, indexing="ij")
    ev_fine = interp(np.stack([YY.ravel(), XX.ravel()], axis=-1)).reshape(
        Fy.size, Fx.size
    )

    search_r = max_gamma * dta_mm
    gamma_map = np.full(ref.shape, np.nan)
    dta2 = dta_mm ** 2

    ref_mask = ref >= cutoff_abs
    ii, jj = np.nonzero(ref_mask)

    for i, j in zip(ii, jj):
        if global_norm:
            DD = dd_abs
        else:
            DD = dd_percent / 100.0 * ref[i, j]
        if DD <= 0:
            continue
        y0, x0 = fy[i], fx[j]
        lo_y = np.searchsorted(Fy, y0 - search_r)
        hi_y = np.searchsorted(Fy, y0 + search_r, side="right")
        lo_x = np.searchsorted(Fx, x0 - search_r)
        hi_x = np.searchsorted(Fx, x0 + search_r, side="right")
        if hi_y <= lo_y or hi_x <= lo_x:
            continue
        Wy = Fy[lo_y:hi_y]
        Wx = Fx[lo_x:hi_x]
        W = ev_fine[lo_y:hi_y, lo_x:hi_x]
        dist2 = (Wy[:, None] - y0) ** 2 + (Wx[None, :] - x0) ** 2
        with np.errstate(invalid="ignore"):
            g = np.sqrt(((W - ref[i, j]) / DD) ** 2 + dist2 / dta2)
        gmin = np.nanmin(g)
        if np.isfinite(gmin):
            gamma_map[i, j] = min(gmin, max_gamma)

    evaluated = np.isfinite(gamma_map)
    n_eval = int(evaluated.sum())
    n_pass = int((gamma_map[evaluated] <= 1.0).sum())
    pass_rate = 100.0 * n_pass / n_eval if n_eval else float("nan")
    mean_gamma = float(np.nanmean(gamma_map)) if n_eval else float("nan")

    return {
        "pass_rate": pass_rate,
        "gamma_map": gamma_map,
        "mean_gamma": mean_gamma,
        "n_eval": n_eval,
        "n_pass": n_pass,
        "n_ref_points": int(ref_mask.sum()),
        "dd_abs": dd_abs,
        "cutoff_abs": cutoff_abs,
        "params": {
            "dd_percent": dd_percent,
            "dta_mm": dta_mm,
            "cutoff_percent": cutoff_percent,
            "cutoff_basis": cutoff_basis,
            "global_norm": global_norm,
            "global_basis": global_basis if global_norm else "local",
            "interp_factor": interp_factor,
            "max_gamma": max_gamma,
        },
    }
