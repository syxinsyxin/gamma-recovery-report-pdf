# -*- coding: utf-8 -*-
"""3 x 2 (reference-side x evaluation-side) gamma decomposition on the closed-loop cohort.

The main queue of the report pipeline only varied the evaluation side; the
reference was always the recovered measured map, itself a smoothed re-rendering
of the report image. The closed-loop data suggest that reference-side smoothing
alone raises the pass rate (the buried Sec. 2.5 result: a coarse map on the
reference side 100%, on the evaluation side 98.2%). This script adds the
reference side to the factor design:

  reference R (all on the det axes, identical reference points, so differences
  between R maps are pure dose-value differences):
    R1 raw    : raw chamber matrix (orientation auto-detected; most detail)
    R2 raw_sm : first-order model of the rendering chain (raw bilinearly
                upsampled to the display grid and resampled back to 32x32)
    R3 rec    : recovered measured map block-averaged (meas_det, i.e. the v7
                reference side; least detail)
  evaluation E:
    E1 plan   : recovered plan dose, full display grid (rec_plan; v7 gamma_A
                evaluation side)
    E2 dcm    : registered RT Dose on the display grid, zeroed outside the
                covered mask (same recipe as the v7 gamma_B evaluation side)

Six gammas per case (3 x 2), all computed through validate._gamma_pair with
exactly the same settings as the report pipeline. Decomposition (per case +
medians):
  reference-side effect  refEffect(R1-R3 | E) = gamma(R1,E) - gamma(R3,E)
  smoothing response     refEffect(R2-R3 | E), refEffect(R1-R2 | E)
  evaluation-side effect srcEffect(E1-E2 | R) = gamma(R,E1) - gamma(R,E2)
  interaction            = srcEffect(R1) - srcEffect(R3)
If |interaction| is large, the "dose source effect" is mostly driven by the
detail-content ratio between the two sides rather than by the source itself -
the quantitative footprint of the circularity discussed in the paper.

Usage:
  python factor3x2.py --cases cl_export/cases_closedloop.csv --out results_3x2/
  python factor3x2.py --cases ... --engine builtin --limit 3      # quick trial
  python factor3x2.py --selftest <empty dir>                     # synthetic end-to-end check

numpy, scipy; pymedphys optional - published numbers use the pymedphys engine.
Read-only w.r.t. validate.py and the report pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate as V  # noqa: E402  reuse _gamma_pair / _load_raw_matrix / _write_synthetic_csv
import registration as rg  # noqa: E402


# ---------------------------------------------------------------------------
def _render_smooth(raw32, axes_det, axes_disp):
    """First-order model of the rendering chain: bilinear upsample to the display
    grid, then resample back to the detector grid.

    Equivalent low-pass to myQA rendering (coarse matrix -> fine display grid)
    plus the block average back to 32x32. R2 vs R3 then differs only by the
    recovery error; R1 vs R2 differs exactly by the rendering smoothing.
    """
    fy, fx = axes_det
    dy, dx = axes_disp
    interp = RegularGridInterpolator((fy, fx), np.where(np.isfinite(raw32), raw32, 0.0),
                                     method="linear", bounds_error=False,
                                     fill_value=None)
    YY, XX = np.meshgrid(dy, dx, indexing="ij")
    fine = interp(np.stack([YY, XX], axis=-1))
    back = RegularGridInterpolator((dy, dx), fine, method="linear",
                                   bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    return back(np.stack([yy, xx], axis=-1))


def run_case_3x2(case, engine="auto", n_det=32):
    out = {"case_id": case.get("case_id", "?")}
    npz_path = case.get("det_npz")
    if not npz_path:
        raise ValueError(f"{out['case_id']}: det_npz missing (3x2 decomposition requires the npz mode)")
    Z = np.load(npz_path, allow_pickle=False)

    meas_det = np.asarray(Z["meas_det"], dtype=float)
    rec_plan_full = np.asarray(Z["rec_plan"], dtype=float)
    dcm_on_grid = np.asarray(Z["dcm_on_grid"], dtype=float)
    covered = (np.asarray(Z["covered"], dtype=float) if "covered" in Z.files
               else np.ones_like(dcm_on_grid))
    axes_det = (np.asarray(Z["det_y"], float), np.asarray(Z["det_x"], float))
    axes_disp = (np.asarray(Z["ymm"], float), np.asarray(Z["xmm"], float))

    def _p(k):
        v = case.get(k)
        if v in (None, ""):
            v = Z[k].item() if k in Z.files else None
        return float(v) if v not in (None, "") else None

    dp, dd = _p("dp"), _p("dd")
    cutoff_pct, gnorm = _p("cutoff_pct"), _p("gnorm")
    declared = float(case["declared_gpr"])

    # raw matrix + orientation (identical to validate.py step 2b)
    raw32, raw_meta = V._load_raw_matrix(case["raw_path"], n_det)
    meas_det_cg = (np.asarray(Z["meas_det_cg"], dtype=float)
                   if "meas_det_cg" in Z.files else None)
    o = rg.best_orientation(meas_det_cg if meas_det_cg is not None else meas_det, raw32)
    raw32 = o["transformed"]
    out["orientation_used"] = o["orientation"]
    out["orient_ncc"] = o["ncc"]

    raw_sm = _render_smooth(raw32, axes_det, axes_disp)  # R2
    rec_ref = meas_det  # R3
    dcm_evl = np.where(covered > 0, dcm_on_grid, 0.0)  # E2 (same recipe as v7 gamma_B)

    refs = {"raw": raw32, "raw_sm": raw_sm, "rec": rec_ref}
    evls = {"plan": (axes_disp, rec_plan_full), "dcm": (axes_disp, dcm_evl)}

    for rn, rmat in refs.items():
        for en, (aevl, emat) in evls.items():
            pr, mg, eng = V._gamma_pair(axes_det, rmat, aevl, emat,
                                        dp, dd, cutoff_pct, gnorm, engine)
            out[f"g_{rn}_{en}"] = pr
            out[f"g_{rn}_{en}_mean"] = mg
            out[f"dev_{rn}_{en}"] = (pr - declared) if np.isfinite(pr) else np.nan
            out["gamma_engine"] = eng

    g = lambda k: out.get(f"g_{k}", float("nan"))
    for en in ("plan", "dcm"):
        out[f"refEff_raw_rec_{en}"] = g(f"raw_{en}") - g(f"rec_{en}")
        out[f"refEff_rawsm_rec_{en}"] = g(f"raw_sm_{en}") - g(f"rec_{en}")
    for rn in ("raw", "raw_sm", "rec"):
        out[f"srcEff_{rn}"] = g(f"{rn}_plan") - g(f"{rn}_dcm")
    out["interaction"] = out["srcEff_raw"] - out["srcEff_rec"]
    out["delta_devA_pair"] = out["dev_raw_plan"] - float(case.get("v7_devA_pp") or np.nan)
    out["declared_gpr"] = declared
    return out


def aggregate(rows):
    def med(k):
        v = np.asarray([r.get(k, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    keys = ([f"g_{r}_{e}" for r in ("raw", "raw_sm", "rec") for e in ("plan", "dcm")]
            + [f"refEff_raw_rec_{e}" for e in ("plan", "dcm")]
            + [f"refEff_rawsm_rec_{e}" for e in ("plan", "dcm")]
            + [f"srcEff_{r}" for r in ("raw", "raw_sm", "rec")]
            + ["interaction", "delta_devA_pair"])
    return {"n_cases": len(rows),
            **{f"median_{k}": med(k) for k in keys},
            "orientations_used": sorted({str(r.get("orientation_used")) for r in rows}),
            "gamma_engines": sorted({str(r.get("gamma_engine")) for r in rows})}


def main():
    ap = argparse.ArgumentParser(description="reference x evaluation 3x2 gamma decomposition")
    ap.add_argument("--cases", help="cases_closedloop.csv (output of export_for_closedloop)")
    ap.add_argument("--out", default="results_3x2", help="output directory")
    ap.add_argument("--engine", default="auto", choices=["auto", "pymedphys", "builtin"])
    ap.add_argument("--limit", type=int, default=0, help="process only the first N cases (trial run)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic end-to-end check (ignores --cases)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        cases = _selftest(out_dir)
    else:
        if not args.cases:
            ap.error("--cases (or --selftest) is required")
        with open(args.cases, encoding="utf-8-sig") as f:
            cases = [dict(r) for r in csv.DictReader(f)]
        if args.limit:
            cases = cases[:args.limit]
    print(f"read {len(cases)} cases")

    results = []
    for i, case in enumerate(cases, 1):
        try:
            r = run_case_3x2(case, engine=args.engine)
            results.append(r)
            print(f"[{i}/{len(cases)}] {r['case_id']}: "
                  f"raw/plan={r['g_raw_plan']:.2f} rec/plan={r['g_rec_plan']:.2f} "
                  f"raw/dcm={r['g_raw_dcm']:.2f} rec/dcm={r['g_rec_dcm']:.2f} "
                  f"| srcEff(rec)={r['srcEff_rec']:+.2f} srcEff(raw)={r['srcEff_raw']:+.2f} "
                  f"interaction={r['interaction']:+.2f}")
        except Exception as e:
            print(f"[{i}/{len(cases)}] {case.get('case_id', '?')}: FAILED - {e}", file=sys.stderr)

    cols = (["case_id", "orientation_used", "orient_ncc", "gamma_engine", "declared_gpr"]
            + [f"g_{r}_{e}" for r in ("raw", "raw_sm", "rec") for e in ("plan", "dcm")]
            + [f"dev_{r}_{e}" for r in ("raw", "raw_sm", "rec") for e in ("plan", "dcm")]
            + [f"refEff_raw_rec_{e}" for e in ("plan", "dcm")]
            + [f"refEff_rawsm_rec_{e}" for e in ("plan", "dcm")]
            + [f"srcEff_{r}" for r in ("raw", "raw_sm", "rec")]
            + ["interaction", "delta_devA_pair"])
    csv_path = out_dir / "factor3x2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    summary = aggregate(results)
    with open(out_dir / "summary_3x2.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("done. factor3x2:", csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
def _selftest(out_dir):
    """One synthetic case end-to-end: a smooth reference should raise the pass
    rate, a noisy one should lower it."""
    import validate as _V

    rng = np.random.default_rng(11)
    n_det, n_disp = 32, 160
    det = np.linspace(-118.1, 118.1, n_det)
    disp = np.linspace(-119.0, 119.0, n_disp)
    DY, DX = np.meshgrid(disp, disp, indexing="ij")
    # large flat field with steep penumbra: gradient~0 in the flat region, so
    # noise cannot be absorbed by the DTA search -> the detail content of the
    # reference side is what matters here
    from scipy.special import expit
    edge = expit((70.0 - np.sqrt(DY ** 2 + DX ** 2)) / 2.5)
    plan = 150.0 * edge
    rec_plan = plan * (1.0 + 0.006 * np.sin(DY / 9.0) * np.cos(DX / 11.0))  # recovery micro-ripple
    dcm = np.roll(plan, 2, axis=1) * 1.03 + 1.0  # DCM: 3 mm shift + 3% amplitude
    covered = np.ones_like(plan)

    fy, fx = det, det
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    ip = RegularGridInterpolator((disp, disp), plan, bounds_error=False, fill_value=None)
    meas_det = ip(np.stack([yy, xx], axis=-1))
    raw32 = meas_det + rng.normal(0, 6.0, meas_det.shape)  # chamber noise

    cdir = out_dir / "_selftest"
    cdir.mkdir(parents=True, exist_ok=True)
    npz_path = cdir / "0000_cl.npz"
    np.savez(npz_path, meas_det=meas_det, cmp_det=meas_det,
             dcm_det=meas_det, dcm_on_grid=dcm, covered=covered,
             rec_plan=rec_plan, det_y=fy, det_x=fx, ymm=disp, xmm=disp,
             reg_ncc=0.99, dp=3.0, dd=3.0, cutoff_pct=5.0, gnorm=150.0, mode="Global")
    csv_path = cdir / "0000_raw.csv"
    _V._write_synthetic_csv(csv_path, raw32)
    cases_csv = cdir / "cases_closedloop.csv"
    with open(cases_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "raw_path", "det_npz", "declared_gpr",
                                          "dp", "dd", "cutoff_pct", "gnorm", "mode",
                                          "v7_devA_pp"])
        w.writeheader()
        w.writerow({"case_id": "0000", "raw_path": str(csv_path), "det_npz": str(npz_path),
                    "declared_gpr": 98.0, "dp": 3.0, "dd": 3.0, "cutoff_pct": 5.0,
                    "gnorm": 150.0, "mode": "Global", "v7_devA_pp": -1.5})
    with open(cases_csv, encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


if __name__ == "__main__":
    main()
