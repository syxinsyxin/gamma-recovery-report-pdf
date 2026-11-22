# -*- coding: utf-8 -*-
"""Closed-loop validation driver.

Purpose: break the circularity of validating recovered PDF dose maps only
against themselves. The raw measurement data kept from the myQA sessions
(32x32 ion-chamber dose matrices exported as CSV; .mcc also supported) serve
as a ground truth on the measured side, and the whole chain
    report PDF -> recovered dose maps -> gamma re-analysis
is checked end-to-end against that ground truth. See manuscript Sec. on
"raw measurement cross-check".

Quantities produced per case:
1) dose-domain fidelity (recovered measured map vs ground-truth matrix):
   - per-point bias (median/mean), SD, max |diff|;
   - gradient stratification: points split at the median |grad| of the ground
     truth, SD reported per layer plus the high/low ratio (same recipe as
     manuscript Table 4);
   - absolute scale fit (a, b, r, rmse) of recovered ~ a*ground_truth + b,
     covering the dose-scale failure mode discussed in the paper.
2) orientation and unit self-checks (data-usability gates):
   - orientation_used: best of 8 dihedral transforms of the ground truth
     (should be consistent across cases);
   - unit_ratio_raw_vs_rec: max-dose ratio ground_truth/recovered. ~1 means
     the raw CSV is already on an absolute scale usable for the closed loop;
     deviations point to counts or a different calibration.
3) gamma closed loop (main result):
   - gamma_rawX = gamma(reference = ground-truth 32x32 matrix; evaluation =
     the same distributions as the report pipeline), criteria mirrored from
     the report; dev_rawX = recomputed pass rate - declared pass rate;
   - compared against dev_rec (recovered measured map as reference) and the
     report-side devA from the v7 pipeline. If |dev_rawX| is of the same
     magnitude as devA and the limits of agreement are no wider, the recovery
     chain is independently validated.
4) plane registration quality: NCC, best slice position and shift of the DCM
   plane against the recovered plan map.

Usage:
  1) prepare a case list (cases.csv, template in the README; raw_path points
     to a CSV/.mcc per case);
  2) python validate.py --cases cases.csv --out results/
  3) outputs: results/per_case.csv, summary.json, ba_dev_raw.png,
     ba_dev_rec.png (+ ba_dev_rawB.png)

Selftest (no real data, no pydicom needed):
  python validate.py --selftest
  builds three synthetic cases (one CSV deliberately rotated 90 deg) and runs
  the full pipeline, including orientation detection and unit checks.

Dependencies: numpy, scipy, matplotlib (plots), pydicom (only for the DCM path).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gamma as gm  # noqa: E402
import registration as rg  # noqa: E402
import mcc_parser as mcc  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_criteria(text):
    """'3%/3mm' -> (3.0, 3.0)"""
    t = (text or "").replace("％", "%").replace(" ", "")
    dd, dta = t.split("/")
    return float(dd.rstrip("%")), float(dta.rstrip("mm").replace("毫米", ""))


def _to_det_grid(a, n=32):
    """Arbitrary grid -> n x n (block-average when divisible; passthrough if already n x n)."""
    a = np.asarray(a, dtype=float)
    if a.shape == (n, n):
        return a
    if a.shape[0] % n == 0 and a.shape[1] % n == 0:
        return rg.block_average(a, a.shape[0] // n)
    raise ValueError(f"array shape {a.shape} cannot be mapped to {n}x n")


def gradient_layers(dose2d):
    """Split points into high/low layers at the median |grad|; returns (high_mask, median)."""
    gy, gx = np.gradient(np.asarray(dose2d, dtype=float))
    g = np.sqrt(gy ** 2 + gx ** 2)
    med = float(np.nanmedian(g))
    return g >= med, med


def bland_altman(x, y, label_x="x", label_y="y"):
    """Bland-Altman stats: bias (mean difference), LoA (+/-1.96 SD), fraction outside."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = y - x
    mean = (x + y) / 2.0
    if d.size == 0:
        nan = float("nan")
        return {
            "n": 0, "bias": nan, "sd": nan, "loa_low": nan, "loa_high": nan,
            "pct_outside": nan, "mean_x": nan, "mean_y": nan,
            "label_x": label_x, "label_y": label_y,
        }
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if d.size > 1 else float("nan")
    loa = (bias - 1.96 * sd, bias + 1.96 * sd) if np.isfinite(sd) else (np.nan, np.nan)
    outside = float(np.mean((d < loa[0]) | (d > loa[1]))) if np.isfinite(sd) else np.nan
    return {
        "n": int(d.size),
        "bias": bias,
        "sd": sd,
        "loa_low": float(loa[0]),
        "loa_high": float(loa[1]),
        "pct_outside": outside,
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "label_x": label_x,
        "label_y": label_y,
    }


def plot_ba(ba, values_x, values_y, out_png, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(values_x, float)
    y = np.asarray(values_y, float)
    m = np.isfinite(x) & np.isfinite(y)
    mean = (x[m] + y[m]) / 2.0
    d = y[m] - x[m]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    ax.scatter(mean, d, s=18, alpha=0.7, edgecolor="none")
    ax.axhline(ba["bias"], color="C1", lw=1.2, label=f"bias={ba['bias']:+.2f}")
    ax.axhline(ba["loa_high"], color="C1", ls="--", lw=1.0, label=f"LoA {ba['loa_high']:+.2f}")
    ax.axhline(ba["loa_low"], color="C1", ls="--", lw=1.0, label=f"LoA {ba['loa_low']:+.2f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Mean of ({ba['label_x']}, {ba['label_y']}) [%]")
    ax.set_ylabel(f"{ba['label_y']} − {ba['label_x']} [pp]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# raw measurement input (CSV 32x32 / .mcc / .npy)
# ---------------------------------------------------------------------------

_NUM_TOKEN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _write_synthetic_csv(path, matrix, header=None):
    """Write a synthetic CSV matrix (selftest helper)."""
    with open(path, "w", encoding="utf-8-sig") as f:
        if header:
            f.write(header + "\n")
        for row in np.asarray(matrix):
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def _load_csv_matrix(path, n_det):
    """Read a 32x32 ion-chamber matrix CSV; returns (cGy matrix, meta).

    Two formats are supported:
    A) the coordinate-annotated myQA export: header line "Name: Integral <ts>, ...,
       Dose unit: mGy", next line holds the x-coordinate labels, data rows start
       with the y-coordinate label. mGy is converted to cGy automatically, the
       provenance name and the detector axes are extracted (true pitch rebuilt
       from (x_max - x_min)/(n-1));
    B) a plain numeric matrix (optionally with a numeric header row, dropped by
       the modal row width).
    """
    meta = {"format": "csv", "source_block": ""}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    if not lines:
        raise ValueError(f"CSV is empty: {path}")

    # ---- format A: coordinate-annotated myQA export ----
    if re.search(r"Dose unit\s*:", lines[0]):
        m_unit = re.search(r"Dose unit\s*:\s*(\w+)", lines[0])
        m_name = re.search(r"Name\s*:\s*([^,\r\n]+)", lines[0])
        unit = (m_unit.group(1).strip().lower() if m_unit else "")
        meta["raw_unit"] = unit
        meta["raw_name"] = (m_name.group(1).strip() if m_name else "")
        toks = [_NUM_TOKEN.findall(ln) for ln in lines[1:]]
        toks = [t for t in toks if t]
        ok_shape = (
            len(toks) >= n_det + 1
            and len(toks[0]) == n_det
            and all(len(t) == n_det + 1 for t in toks[1:n_det + 1])
        )
        if ok_shape:
            xc = np.asarray([float(t) for t in toks[0]], dtype=float)
            yc = np.asarray([float(t[0]) for t in toks[1:n_det + 1]], dtype=float)
            m = np.asarray([[float(v) for v in t[1:n_det + 1]]
                            for t in toks[1:n_det + 1]], dtype=float)
            if m.shape == (n_det, n_det) and np.all(np.diff(xc) > 0) \
                    and np.all(np.diff(yc) > 0):
                scale = 0.1 if unit == "mgy" else (1.0 if unit in ("cgy", "gy", "") else 1.0)
                if unit == "gy":
                    scale = 100.0
                meta["det_x"] = xc
                meta["det_y"] = yc
                meta["pitch_from_coords"] = float((xc[-1] - xc[0]) / (n_det - 1))
                return m * scale, meta
        raise ValueError(
            f"failed to parse coordinate-annotated myQA CSV: {path} "
            f"({len(toks)} numeric rows; expected an x-coord line + 32 rows of [y label + 32 values])"
        )

    # ---- format B: plain numeric matrix ----
    rows = []
    for line in lines:
        tokens = _NUM_TOKEN.findall(line)
        if tokens:
            rows.append([float(t) for t in tokens])
    if not rows:
        raise ValueError(f"no numeric rows found in CSV: {path}")
    n2 = n_det * n_det
    from collections import Counter
    widths = Counter(len(r) for r in rows)
    w = widths.most_common(1)[0][0]
    sel = [r for r in rows if len(r) == w]
    if w == n_det and len(sel) == n_det:
        return np.asarray(sel, dtype=float), meta
    flat = [v for r in sel for v in r]
    if len(flat) == n2:
        return np.asarray(flat, dtype=float).reshape(n_det, n_det), meta
    flat_all = [v for r in rows for v in r]
    if len(flat_all) == n2:
        return np.asarray(flat_all, dtype=float).reshape(n_det, n_det), meta
    raise ValueError(
        f"CSV cannot be reshaped to a {n_det}x{n_det} matrix: {len(rows)} numeric rows "
        f"(row-width distribution {dict(widths)}, {len(flat_all)} values in total, expected {n2}). "
        "Inspect the file manually."
    )


def _load_raw_matrix(path, n_det):
    """Dispatch on extension; returns (matrix, meta)."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".npy":
        m = np.load(path)
        meta = {"format": "npy", "source_block": ""}
    elif ext in (".csv", ".txt"):
        m, meta = _load_csv_matrix(path, n_det)   # meta carries raw_unit/raw_name/det_x/det_y etc.
        meta.setdefault("format", "csv")
        meta.setdefault("source_block", "")
    elif ext == ".mcc":
        res = mcc.parse_mcc(path, expected_shape=(n_det, n_det),
                            block_index=None)
        m = res["matrix"]
        meta = {"format": "mcc", "source_block": res["source_block"]}
    else:
        raise ValueError(f"unsupported raw-measurement format: {path} (supported: .csv/.txt/.mcc/.npy)")
    m = np.asarray(m, dtype=float)
    if m.shape != (n_det, n_det):
        m = _to_det_grid(m, n_det)
    return m, meta


# ---------------------------------------------------------------------------
# gamma computation (pymedphys preferred, builtin fallback)
# ---------------------------------------------------------------------------

try:
    from pymedphys import gamma as _PM_GAMMA  # noqa: F401
except Exception:  # optional dependency
    _PM_GAMMA = None


def _gamma_pair(axes_ref, ref, axes_evl, evl, dp, dd, cutoff_pct, gnorm, engine="auto"):
    """One gamma computation: (ref, axes_ref) vs (evl, axes_evl).

    Settings mirror the report pipeline exactly:
    DD criterion = dp %, DTA = dd mm, cutoff = cutoff_pct % of gnorm,
    global normalisation = gnorm (the report Dmax), no local gamma.

    engine="auto": pymedphys first, fall back to the bundled gamma_analysis
    (equivalent configuration; approximate) when pymedphys is missing or the
    call fails. Published numbers use pymedphys.

    Returns
    -------
    (pass_rate, mean_gamma, engine_used)
    """
    ref0 = np.asarray(np.where(np.isfinite(ref), ref, 0.0), dtype=np.float64)
    evl0 = np.asarray(np.where(np.isfinite(evl), evl, 0.0), dtype=np.float64)
    axes_ref = (np.asarray(axes_ref[0], float), np.asarray(axes_ref[1], float))
    axes_evl = (np.asarray(axes_evl[0], float), np.asarray(axes_evl[1], float))
    if engine in ("auto", "pymedphys") and _PM_GAMMA is not None:
        try:
            g = np.asarray(_PM_GAMMA(
                axes_ref, ref0, axes_evl, evl0,
                dose_percent_threshold=float(dp),
                distance_mm_threshold=float(dd),
                lower_percent_dose_cutoff=float(cutoff_pct),
                interp_fraction=10, max_gamma=2.0,
                local_gamma=False, global_normalisation=float(gnorm),
            ), dtype=float)
            v = np.isfinite(g)
            if v.sum() == 0:
                return float("nan"), float("nan"), "pymedphys"
            return (
                100.0 * float(np.mean(g[v] <= 1.0)),
                float(np.mean(g[v])),
                "pymedphys",
            )
        except TypeError as e:
            print(f"[warn] pymedphys.gamma call failed ({e}); falling back to the builtin engine for this case",
                  file=sys.stderr)
    # fallback: builtin engine, equivalent configuration (cutoff = declared cutoff% x gnorm,
    # global normalisation = gnorm)
    def _sp(ax):
        return (float(np.median(np.diff(ax[0]))), float(np.median(np.diff(ax[1]))))

    r = gm.gamma_analysis(
        ref0, _sp(axes_ref), evl0, _sp(axes_evl),
        dd_percent=float(dp), dta_mm=float(dd),
        cutoff_basis="fixed", cutoff_value=float(cutoff_pct) / 100.0 * float(gnorm),
        global_norm=True, global_basis="fixed", global_value=float(gnorm),
        interp_factor=10, max_gamma=2.0,
    )
    return r["pass_rate"], r["mean_gamma"], "builtin"


# ---------------------------------------------------------------------------
# per-case processing
# ---------------------------------------------------------------------------

def run_case(case, cfg):
    """Process one case; returns the per-case result dict (fields documented in the README)."""
    out = {"case_id": case.get("case_id", "?")}
    n_det = cfg.get("n_det", 32)
    pitch = cfg.get("pitch_mm", 7.619)
    interp_factor = cfg.get("interp_factor", 10)

    # ---- 1) raw measurement -> n x n ground-truth matrix (CSV / .mcc / .npy) ----
    raw_path = case.get("raw_path") or case.get("csv_path") or case.get("mcc_path")
    raw32, raw_meta = _load_raw_matrix(raw_path, n_det)
    out["raw_format"] = raw_meta["format"]
    out["raw_source_block"] = raw_meta.get("source_block", "")
    out["raw_max"] = float(np.nanmax(raw32))

    # ---- 2) recovered-map artifacts: v7 npz (preferred) or standalone npy (legacy path) ----
    npz_path = case.get("det_npz")
    axes_det = axes_disp = None
    dp = dd = cutoff_pct = gnorm = None
    if npz_path:
        Z = np.load(npz_path, allow_pickle=False)
        meas_det = np.asarray(Z["meas_det"], dtype=float)
        cmp_det = np.asarray(Z["cmp_det"], dtype=float)
        dcm_det = np.asarray(Z["dcm_det"], dtype=float)
        dcm_on_grid = np.asarray(Z["dcm_on_grid"], dtype=float)
        rec_plan_full = np.asarray(Z["rec_plan"], dtype=float)
        axes_det = (np.asarray(Z["det_y"], float), np.asarray(Z["det_x"], float))
        axes_disp = (np.asarray(Z["ymm"], float), np.asarray(Z["xmm"], float))
        rec_meas32 = meas_det          # v7 shared-geom block average, used as-is
        rec_plan32 = cmp_det
        pitch_eff = float(np.median(np.diff(axes_det[1])))
        out["det_pitch_eff_mm"] = float(f"{pitch_eff:.4f}")
        out["v7_reg_ncc"] = float(Z["reg_ncc"])

        # CSV-coordinate diagnostic channel (when provided by the exporter)
        meas_det_cg = (np.asarray(Z["meas_det_cg"], dtype=float)
                       if "meas_det_cg" in Z.files else None)
        for k in ("raw_unit", "raw_name", "provenance_ok", "pitch_from_coords",
                  "id_method", "id_ncc"):
            if k in Z.files:
                out[k] = Z[k].item() if Z[k].ndim == 0 else Z[k]

        def _p(k, cast=float, default=None):
            v = case.get(k)
            if v in (None, ""):
                v = Z[k].item() if k in Z.files else default
            return cast(v) if v not in (None, "") else default

        dp = _p("dp"); dd = _p("dd")
        cutoff_pct = _p("cutoff_pct"); gnorm = _p("gnorm")
        mode = str(case.get("mode") or (Z["mode"].item() if "mode" in Z.files else "") or "")
        out["dp"], out["dd"] = dp, dd
        out["cutoff_pct"], out["gnorm"], out["mode"] = cutoff_pct, gnorm, mode
        if mode.lower().startswith("local"):
            out["norm_mismatch"] = 1   # v7 gamma assumed global; a local report needs a per-case review
        orient_base = meas_det_cg if meas_det_cg is not None else rec_meas32
        out["geom_cg_used"] = bool(meas_det_cg is not None)
    else:
        rec_meas = np.load(case["rec_meas_npy"])
        rec_plan_full = np.load(case["rec_plan_npy"])
        rec_meas32 = _to_det_grid(rec_meas, n_det)
        rec_plan32 = _to_det_grid(rec_plan_full, n_det)
        orient_base = rec_meas32

    # ---- 2b) orientation auto-detection (ground truth vs recovered measured map;
    #          prefers the CSV-geometry diagnostic version) ----
    if cfg.get("orientation_auto", True):
        o = rg.best_orientation(orient_base, raw32)
        raw32 = o["transformed"]
        out["orientation_used"] = o["orientation"]
        out["orient_ncc"] = o["ncc"]
    else:
        out["orientation_used"] = "fixed:" + str(cfg.get("orientation", "normal"))
        out["orient_ncc"] = rg.ncc(rec_meas32, raw32)

    # ---- 2c) unit self-check: max-dose ratio ground truth / recovered ----
    m_rec = float(np.nanmax(rec_meas32))
    m_raw = float(np.nanmax(raw32))
    ratio = m_raw / m_rec if m_rec else float("nan")
    out["unit_ratio_raw_vs_rec"] = ratio
    if np.isfinite(ratio) and abs(ratio - 1.0) <= 0.15:
        out["units_flag"] = "consistent"
    elif np.isfinite(ratio) and 0.5 <= ratio <= 2.0:
        out["units_flag"] = "scale_offset_check"
    elif np.isfinite(ratio):
        out["units_flag"] = "SCALE_MISMATCH"
        print(
            f"[warn] {out['case_id']}: ground-truth/recovered max-dose ratio = {ratio:.4g}; "
            "looks like counts vs absolute dose (or a calibration offset); "
            "interpret gamma results in relative terms only",
            file=sys.stderr,
        )
    else:
        out["units_flag"] = "unknown"

    # ---- 3) dose-domain fidelity: recovered measured map vs ground truth ----
    diff = rec_meas32 - raw32
    fin = np.isfinite(diff)
    out["dose_bias_median"] = float(np.nanmedian(diff))
    out["dose_bias_mean"] = float(np.nanmean(diff))
    out["dose_sd"] = float(np.nanstd(diff))
    out["dose_max_abs"] = float(np.nanmax(np.abs(diff)))
    hi, gmed = gradient_layers(raw32)
    out["grad_median"] = gmed
    out["sd_high_grad"] = float(np.nanstd(diff[hi]))
    out["sd_low_grad"] = float(np.nanstd(diff[~hi]))
    r_hl = (
        out["sd_high_grad"] / out["sd_low_grad"]
        if out["sd_low_grad"] and np.isfinite(out["sd_low_grad"]) and out["sd_low_grad"] > 0
        else float("nan")
    )
    out["sd_ratio_high_low"] = float(r_hl)
    out["scale_fit"] = rg.fit_dose_scale(rec_meas32, raw32)  # recovered ~ a*ground_truth + b

    # ---- 3b) CSV-geometry diagnostic channel: separates geometric-calibration
    #          error from dose-recovery error ----
    if npz_path and meas_det_cg is not None:
        diff_cg = meas_det_cg - raw32
        out["dose_bias_cg_median"] = float(np.nanmedian(diff_cg))
        out["dose_sd_cg"] = float(np.nanstd(diff_cg))
        out["dose_max_cg"] = float(np.nanmax(np.abs(diff_cg)))
        fit_cg = rg.fit_dose_scale(meas_det_cg, raw32)
        out["scale_cg_a"] = fit_cg["a"]
        out["scale_cg_b"] = fit_cg["b"]
        out["scale_cg_r"] = fit_cg["r"]

    # ---- 4) gamma closed loop ----
    declared = float(case["declared_gpr"])
    out["declared_gpr"] = declared

    if npz_path:
        # ---- 4a) v7 artifact mode: four raw-side gammas mirroring v7 gamma_A/gamma_Adet/gamma_B/gamma_Bdet.
        # The only difference from v7 is the reference side: the ground-truth matrix
        # (raw chamber doses) instead of the recovered map -- the core of the closed loop.
        engine = cfg.get("gamma_engine", "auto")

        def _pair(aref, rref, aevl, eevl, tag):
            pr, mg, eng = _gamma_pair(aref, rref, aevl, eevl,
                                      dp, dd, cutoff_pct, gnorm, engine)
            out[f"gamma_{tag}"] = pr
            out[f"gamma_{tag}_mean"] = mg
            out[f"dev_{tag}"] = (pr - declared) if np.isfinite(pr) else np.nan
            out["gamma_engine"] = eng
            return pr

        prA = _pair(axes_det, raw32, axes_disp, rec_plan_full, "rawA")    # mirrors gamma_A
        _pair(axes_det, raw32, axes_det, cmp_det, "rawAdet")              # mirrors gamma_Adet
        _pair(axes_det, raw32, axes_disp, dcm_on_grid, "rawB")            # mirrors gamma_B
        _pair(axes_det, raw32, axes_det, dcm_det, "rawBdet")              # mirrors gamma_Bdet
        # aliases so that the aggregate/BA plots match the gamma_A structure
        out["gamma_raw"] = out["gamma_rawA"]
        out["dev_raw"] = out["dev_rawA"]

        # v7 comparison columns; dev_rawX - devX_v7 = net effect of swapping the
        # reference (recovered map -> ground truth)
        for src, dst in (("devA_pp", "devA_v7"), ("devAdet_pp", "devAdet_v7"),
                         ("devB_pp", "devB_v7"), ("devBdet_pp", "devBdet_v7")):
            v = case.get(src)
            if v not in (None, ""):
                try:
                    out[dst] = float(v)
                except (TypeError, ValueError):
                    pass
        for tag in ("A", "Adet", "B", "Bdet"):
            a = out.get(f"dev_raw{tag}")
            b = out.get(f"dev{tag}_v7")
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                out[f"delta_dev{tag}"] = float(a) - float(b)
    elif case.get("dcm_path"):
        # ---- 4b) legacy path: no v7 artifacts, register the DCM ourselves (needs pydicom) ----
        dd_pct, dta_mm = _parse_criteria(case.get("criteria", "3%/3mm"))
        norm = (case.get("norm") or "global").lower()
        dcm = rg.load_rt_dose(case["dcm_path"])
        align = rg.align_dcm_to_detector_map(
            dcm, rec_plan32, pitch=pitch, n=n_det,
            margin_mm=cfg.get("margin_mm", 8.0),
            orientation=cfg.get("orientation", "normal"),
        )
        out.update(
            {
                "align_y_mm": align["best_y_mm"],
                "align_shift_dz_mm": align["best_shift_dz"],
                "align_shift_dx_mm": align["best_shift_dx"],
                "align_ncc": align["ncc_final"],
            }
        )
        plane, z, x = rg.sample_coronal_slice(dcm, align["best_y_mm"])
        zc = align["center_used"][0]
        xc = align["center_used"][1]
        half = pitch * n_det / 2.0
        nf_y = int(np.ceil((2 * half) / (pitch / interp_factor))) + 1
        zf = zc - half + np.arange(nf_y) * (pitch / interp_factor)
        xf = xc - half + np.arange(nf_y) * (pitch / interp_factor)
        from scipy.ndimage import map_coordinates

        dz = z[1] - z[0]
        dx = x[1] - x[0]
        ZZ, XX = np.meshgrid((zf - z[0]) / dz, (xf - x[0]) / dx, indexing="ij")
        ev_fine = map_coordinates(
            plane, [ZZ.ravel(), XX.ravel()], order=1, mode="constant", cval=np.nan
        ).reshape(nf_y, nf_y)
        g_raw = gm.gamma_analysis(
            raw32, (pitch, pitch), ev_fine, (pitch / interp_factor,) * 2,
            dd_percent=dd_pct, dta_mm=dta_mm,
            cutoff_percent=cfg.get("cutoff_percent", 5.0),
            global_norm=(norm == "global"),
            interp_factor=1,
        )
        out["gamma_raw"] = g_raw["pass_rate"]
        out["gamma_raw_mean"] = g_raw["mean_gamma"]
        out["dev_raw"] = g_raw["pass_rate"] - declared
        g_rec = gm.gamma_analysis(
            rec_meas32, (pitch, pitch), ev_fine, (pitch / interp_factor,) * 2,
            dd_percent=dd_pct, dta_mm=dta_mm,
            cutoff_percent=cfg.get("cutoff_percent", 5.0),
            global_norm=(norm == "global"),
            interp_factor=1,
        )
        out["gamma_rec"] = g_rec["pass_rate"]
        out["dev_rec"] = g_rec["pass_rate"] - declared
    else:
        out["gamma_raw"] = np.nan
        out["dev_raw"] = np.nan
        out["gamma_rec"] = np.nan
        out["dev_rec"] = np.nan

    if case.get("declared_mean_gamma"):
        try:
            out["declared_mean_gamma"] = float(case["declared_mean_gamma"])
        except (TypeError, ValueError):
            out["declared_mean_gamma"] = np.nan

    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

FIELDS = [
    "case_id", "raw_format", "raw_source_block", "raw_max",
    "raw_unit", "raw_name", "provenance_ok", "pitch_from_coords",
    "orientation_used", "orient_ncc", "unit_ratio_raw_vs_rec", "units_flag",
    "dp", "dd", "cutoff_pct", "gnorm", "mode", "norm_mismatch",
    "det_pitch_eff_mm", "gamma_engine", "v7_reg_ncc",
    "geom_cg_used", "id_method", "id_ncc",
    "dose_bias_median", "dose_bias_mean", "dose_sd", "dose_max_abs",
    "dose_bias_cg_median", "dose_sd_cg", "dose_max_cg",
    "scale_cg_a", "scale_cg_b", "scale_cg_r",
    "grad_median", "sd_high_grad", "sd_low_grad", "sd_ratio_high_low",
    "scale_a", "scale_b", "scale_r", "scale_rmse",
    "align_y_mm", "align_shift_dz_mm", "align_shift_dx_mm", "align_ncc",
    "declared_gpr", "gamma_raw", "dev_raw", "gamma_rec", "dev_rec",
    "gamma_rawA", "dev_rawA", "gamma_rawAdet", "dev_rawAdet",
    "gamma_rawB", "dev_rawB", "gamma_rawBdet", "dev_rawBdet",
    "devA_v7", "devAdet_v7", "devB_v7", "devBdet_v7",
    "delta_devA", "delta_devAdet", "delta_devB", "delta_devBdet",
    "declared_mean_gamma",
]


def _flatten(out):
    row = dict(out)
    fit = out.get("scale_fit", {})
    row["scale_a"] = fit.get("a", np.nan)
    row["scale_b"] = fit.get("b", np.nan)
    row["scale_r"] = fit.get("r", np.nan)
    row["scale_rmse"] = fit.get("rmse", np.nan)
    return {k: row.get(k, "") for k in FIELDS}


def aggregate(results, out_dir):
    """Write per_case.csv / summary.json and the BA plots."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(_flatten(r))

    declared = [r.get("declared_gpr") for r in results]
    g_raw = [r.get("gamma_raw") for r in results]
    g_rec = [r.get("gamma_rec") for r in results]
    dev_raw = [r.get("dev_raw") for r in results]
    dev_rec = [r.get("dev_rec") for r in results]

    def med(v):
        v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    summary = {
        "n_cases": len(results),
        "median_dev_raw": med(dev_raw),
        "median_dev_rec": med(dev_rec),
        "median_dose_bias": med([r.get("dose_bias_median") for r in results]),
        "median_sd_ratio_high_low": med([r.get("sd_ratio_high_low") for r in results]),
        "median_align_ncc": med([r.get("align_ncc") for r in results]),
        # v7 artifact mode: four raw-side deviations and their paired differences
        # (the core closed-loop result)
        "median_dev_rawA": med([r.get("dev_rawA") for r in results]),
        "median_dev_rawAdet": med([r.get("dev_rawAdet") for r in results]),
        "median_dev_rawB": med([r.get("dev_rawB") for r in results]),
        "median_dev_rawBdet": med([r.get("dev_rawBdet") for r in results]),
        "median_delta_devA": med([r.get("delta_devA") for r in results]),
        "median_delta_devAdet": med([r.get("delta_devAdet") for r in results]),
        "median_delta_devB": med([r.get("delta_devB") for r in results]),
        "median_delta_devBdet": med([r.get("delta_devBdet") for r in results]),
        "median_devA_v7": med([r.get("devA_v7") for r in results]),
        "median_devAdet_v7": med([r.get("devAdet_v7") for r in results]),
        "median_devB_v7": med([r.get("devB_v7") for r in results]),
        "median_devBdet_v7": med([r.get("devBdet_v7") for r in results]),
        "median_scale_a": med([r.get("scale_a") for r in results]),
        "median_unit_ratio": med([r.get("unit_ratio_raw_vs_rec") for r in results]),
        "orientations_used": sorted({r.get("orientation_used") for r in results
                                     if r.get("orientation_used")}),
        "units_flags": sorted({r.get("units_flag") for r in results
                               if r.get("units_flag")}),
    }
    ba_raw = bland_altman(declared, g_raw, "declared GPR", "gamma_raw")
    ba_rec = bland_altman(declared, g_rec, "declared GPR", "gamma_rec")
    g_rawB = [r.get("gamma_rawB") for r in results]
    ba_rawB = bland_altman(declared, g_rawB, "declared GPR", "gamma_rawB")
    summary["ba_gamma_raw_vs_declared"] = ba_raw
    summary["ba_gamma_rec_vs_declared"] = ba_rec
    summary["ba_gamma_rawB_vs_declared"] = ba_rawB
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    try:
        plot_ba(
            ba_raw, declared, g_raw,
            os.path.join(out_dir, "ba_dev_raw.png"),
            "Bland–Altman: gamma_raw (mcc-referenced) vs declared",
        )
        plot_ba(
            ba_rec, declared, g_rec,
            os.path.join(out_dir, "ba_dev_rec.png"),
            "Bland–Altman: gamma_rec (PDF-recovered, measured side) vs declared",
        )
        plot_ba(
            ba_rawB, declared, g_rawB,
            os.path.join(out_dir, "ba_dev_rawB.png"),
            "Bland–Altman: gamma_rawB (raw chambers vs DCM) vs declared",
        )
    except Exception as e:  # plotting must not break the main flow
        print(f"[warn] BA plotting failed: {e}", file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _selftest(out_dir):
    """End-to-end selftest on synthetic cases (no pydicom, no real data).

    Covers: CSV reading (with header line), a CSV deliberately rotated 90 deg
    (orientation auto-detection), the .mcc path; dose-domain fidelity, scale
    fits, aggregation and the BA plotting pipeline.
    """
    import tempfile

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    tmp = tempfile.mkdtemp(prefix="closedloop_selftest_")

    # synthetic 32x32 "ground truth": smooth elliptical dose distribution (cGy scale)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32), indexing="ij")
    base = 200.0 * np.exp(-((yy ** 2) / 0.8 + (xx ** 2) / 0.35)) + 5.0

    results = []
    expect_orient = ["normal", "UD+T", "normal"]  # case1 is deliberately rotated 90 deg
    for k, shift_cells in enumerate([0, 1, 2]):
        shifted = np.roll(base, shift_cells, axis=1) * (1.0 + 0.01 * k)
        raw32 = shifted + rng.normal(0, 0.5, shifted.shape)

        if k == 1:  # CSV, deliberately rotated (tests orientation auto-detection)
            raw_path = os.path.join(tmp, f"case{k}.csv")
            _write_synthetic_csv(raw_path, np.rot90(raw32, 1), header="row,col,dose")
        elif k == 0:  # CSV with a header line
            raw_path = os.path.join(tmp, f"case{k}.csv")
            _write_synthetic_csv(raw_path, raw32, header="32x32 ion chamber dose (cGy)")
        else:  # .mcc path compatibility
            raw_path = os.path.join(tmp, f"case{k}.mcc")
            mcc.write_synthetic_mcc(raw_path, raw32)

        # "recovered measured map" = ground truth + small noise; "recovered plan" = ground truth
        rec_meas32 = raw32 + rng.normal(0, 1.0, raw32.shape)
        rec_plan32 = shifted
        rec_meas_npy = os.path.join(tmp, f"rec_meas_{k}.npy")
        rec_plan_npy = os.path.join(tmp, f"rec_plan_{k}.npy")
        np.save(rec_meas_npy, rec_meas32)
        np.save(rec_plan_npy, rec_plan32)

        case = {
            "case_id": f"SYN{k:03d}",
            "raw_path": raw_path,
            "rec_meas_npy": rec_meas_npy,
            "rec_plan_npy": rec_plan_npy,
            "dcm_path": "",  # selftest skips the DCM/gamma closed loop
            "declared_gpr": 96.0 + k,
            "criteria": "3%/3mm",
            "norm": "global",
        }
        r = run_case(case, cfg={"n_det": 32, "pitch_mm": 7.619,
                                "orientation_auto": True})
        assert r["orientation_used"] == expect_orient[k], \
            f"orientation mismatch: {r['orientation_used']}, expected {expect_orient[k]}"
        assert r["units_flag"] == "consistent", r["units_flag"]
        results.append(r)

    # ---- SYN003: v7-artifact (npz) mode, raw side deliberately rotated 90 deg ----
    # the display grid must be wider than the detector grid (32 x 7.619 = 243.8 mm),
    # matching a real report
    ax1d = np.linspace(-150.0, 150.0, 301)
    yy2, xx2 = np.meshgrid(ax1d, ax1d, indexing="ij")
    rec_plan = 200.0 * np.exp(-((yy2 ** 2) / (2 * 55.0 ** 2)
                                + (xx2 ** 2) / (2 * 35.0 ** 2))) + 5.0
    noise = rng.normal(0, 0.8, rec_plan.shape)
    mask = np.sqrt(yy2 ** 2 + xx2 ** 2) < 138
    rec_meas_full = np.where(mask, rec_plan * 0.98 + noise, np.nan)
    det_c = (np.arange(32) - 15.5) * 7.619
    from scipy.interpolate import RegularGridInterpolator

    interp_m = RegularGridInterpolator((ax1d, ax1d),
                                       np.where(np.isfinite(rec_meas_full),
                                                rec_meas_full, 0.0))
    interp_p = RegularGridInterpolator((ax1d, ax1d), rec_plan)
    Yg, Xg = np.meshgrid(det_c, det_c, indexing="ij")
    pts = np.stack([Yg.ravel(), Xg.ravel()], axis=-1)
    meas_det = interp_m(pts).reshape(32, 32)
    cmp_det = interp_p(pts).reshape(32, 32)
    dcm_on_grid = rec_plan + rng.normal(0, 0.5, rec_plan.shape)
    dcm_on_grid[~mask] = 0.0
    dcm_det = interp_p(np.stack([Yg.ravel(), Xg.ravel()], axis=-1)
                       ).reshape(32, 32) + rng.normal(0, 0.4, (32, 32))
    gnorm = float(rec_plan.max())
    # ground truth = the v7-geometry interpolation; the v7 geometry adds a 1.5
    # geometric error component while the CSV geometry adds only 0.2;
    # raw32 differs from the truth by 0.5 -> sd(v7 geom)~1.58 should exceed
    # sd(CSV geom)~0.54
    meas_det_true = meas_det
    meas_det = meas_det_true + rng.normal(0, 1.5, meas_det.shape)
    meas_det_cg = meas_det_true + rng.normal(0, 0.2, meas_det.shape)
    raw32 = meas_det_true + rng.normal(0, 0.5, meas_det.shape)
    raw_npz = os.path.join(tmp, "SYN003_cl.npz")
    np.savez_compressed(
        raw_npz, rec_meas=rec_meas_full, rec_plan=rec_plan,
        meas_det=meas_det, cmp_det=cmp_det, dcm_det=dcm_det,
        dcm_on_grid=dcm_on_grid, covered=mask,
        det_y=det_c, det_x=det_c,
        det_y_csv=det_c, det_x_csv=det_c,
        meas_det_cg=meas_det_cg,
        xmm=ax1d, ymm=ax1d,
        dp=np.array(3.0), dd=np.array(3.0), cutoff_pct=np.array(5.0),
        gnorm=np.array(gnorm), mode=np.array("global"), reg_ncc=np.array(0.995),
        raw_unit=np.array("mgy"), raw_name=np.array("Integral 2099-01-01 00:00:00.00"),
        provenance_ok=np.array(True), pitch_from_coords=np.array(7.6194),
        id_method=np.array("labels"), id_ncc=np.array(0.999),
    )
    # myQA coordinate-annotated CSV (mGy units, x10, deliberately rotated 90 deg)
    coords = (det_c * 1.0).round(1)
    raw3_path = os.path.join(tmp, "case3.csv")
    with open(raw3_path, "w", encoding="utf-8-sig") as f:
        f.write(f"Name: Integral 2099-01-01 00:00:00.00, Length unit: mm, "
                f"Dose unit: mGy\n")
        f.write(",".join(f"{v:.1f}" for v in coords) + "\n")
        rot = np.rot90(raw32, 1) * 10.0
        for i, yv in enumerate(coords):
            f.write(f"{yv:.1f}," + ",".join(f"{v:.6f}" for v in rot[i]) + "\n")
    case3 = {
        "case_id": "SYN003",
        "raw_path": raw3_path,
        "det_npz": raw_npz,
        "declared_gpr": 97.0,
        "dp": "3", "dd": "3", "cutoff_pct": "5", "gnorm": str(gnorm),
        "mode": "global",
        "devA_pp": "-1.5", "devAdet_pp": "-4.4",
    }
    r3 = run_case(case3, cfg={"n_det": 32, "pitch_mm": 7.619,
                              "orientation_auto": True})
    assert r3["orientation_used"] == "UD+T", r3["orientation_used"]
    assert r3["units_flag"] == "consistent", r3["units_flag"]
    assert r3.get("raw_unit") == "mgy", r3.get("raw_unit")
    assert r3.get("provenance_ok") in (True, "True", 1), r3.get("provenance_ok")
    assert abs(r3["raw_max"] - float(np.max(raw32))) < 1e-6  # mGy -> cGy conversion must be exact
    assert r3.get("geom_cg_used") is True or r3.get("geom_cg_used") == "True"
    assert np.isfinite(r3["gamma_rawA"]) and np.isfinite(r3["delta_devA"])
    assert r3["gamma_engine"] in ("pymedphys", "builtin")
    assert abs(r3["delta_devA"] - (r3["dev_rawA"] - (-1.5))) < 1e-9
    sd_v7, sd_cg = r3["dose_sd"], r3["dose_sd_cg"]
    assert sd_cg < sd_v7, (
        f"CSV-geometry channel should reduce the scatter: sd(v7 geom)={sd_v7:.4f} "
        f"sd(CSV geom)={sd_cg:.4f} geom_cg_used={r3.get('geom_cg_used')}"
    )
    print(f"[selftest] SYN003 npz+myQA-format OK: engine={r3['gamma_engine']} "
          f"gamma_rawA={r3['gamma_rawA']:.2f} dev_rawA={r3['dev_rawA']:+.2f} "
          f"delta_devA={r3['delta_devA']:+.2f} "
          f"sd(v7 geom)={r3['dose_sd']:.3f} sd(CSV geom)={r3['dose_sd_cg']:.3f}")
    results.append(r3)

    summary = aggregate(results, out_dir)
    print("[selftest] OK (incl. orientation detection and unit checks); summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description="closed-loop validation driver (CSV/.mcc ground truth)")
    ap.add_argument("--cases", help="path to cases.csv")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--n-det", type=int, default=32)
    ap.add_argument("--pitch-mm", type=float, default=7.619)
    ap.add_argument("--interp-factor", type=int, default=10)
    ap.add_argument("--cutoff-percent", type=float, default=5.0)
    ap.add_argument("--margin-mm", type=float, default=8.0)
    ap.add_argument("--orientation", default="normal",
                    help="fixed orientation used when auto-detection is disabled: normal|T|UD|LR, combinable with +")
    ap.add_argument("--fixed-orientation", action="store_true",
                    help="disable orientation auto-detection and force --orientation")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cfg = {
        "n_det": args.n_det,
        "pitch_mm": args.pitch_mm,
        "interp_factor": args.interp_factor,
        "cutoff_percent": args.cutoff_percent,
        "margin_mm": args.margin_mm,
        "orientation": args.orientation,
        "orientation_auto": not args.fixed_orientation,
    }

    if args.selftest:
        _selftest(args.out)
        return
    if not args.cases:
        ap.error("--cases cases.csv or --selftest is required")

    with open(args.cases, newline="", encoding="utf-8-sig") as f:
        cases = [dict(r) for r in csv.DictReader(f)]
    print(f"read {len(cases)} cases")

    results = []
    for i, case in enumerate(cases, 1):
        cid = case.get("case_id", f"case{i}")
        try:
            r = run_case(case, cfg)
            results.append(r)
            print(
                f"[{i}/{len(cases)}] {cid}: gamma_raw={r.get('gamma_raw', float('nan')):.2f} "
                f"dev_raw={r.get('dev_raw', float('nan')):+.2f}pp "
                f"NCC={r.get('align_ncc', float('nan')):.4f}"
            )
        except Exception as e:
            print(f"[{i}/{len(cases)}] {cid}: FAILED - {e}", file=sys.stderr)

    summary = aggregate(results, args.out)
    print("done. summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
