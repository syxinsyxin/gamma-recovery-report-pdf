# -*- coding: utf-8 -*-
"""Bounding experiment for the common-mode amplitude-bias mechanism (control C4).

Hypothesis: the recovery chain carries a consistent amplitude bias
(recovered ~ a*raw + b; per-case fits give a~1.02, b~+0.9 cGy, cf. manuscript
Table 5). In the (recovered reference x recovered plan) pairing the bias is
shared and cancels; paired against the DCM it is exposed on one side only.
To bound its effect, undo the bias on the recovered reference,
r' = (r - b)/a, and recompute both pairings:

    gamma(rec_corr, DCM)   expected to rise markedly (e.g. 92.6 -> ~95-96)
    gamma(rec_corr, plan)  expected to drop below the uncorrected value

Both moving together supports the bias mechanism; a one-sided or null response
sends us back to the geometry/registration controls.

A symmetric companion control (C5) corrects the evaluation side (plan) with its
own per-case fit against the DCM (a_p, b_p); note that this fit uses the DCM as
baseline, so (a_p, b_p) mixes the chain bias with TPS differences -- diagnostic
only. See columns g_c5_plan_evcorr / g_c5_plan_bothcorr.

Usage: python factor3x2_c4.py --out results_c4 [--limit 3]
(the case list is auto-discovered as in factor3x2; engine auto prefers pymedphys)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate as V
import registration as rg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="cases_closedloop.csv (auto-discovered if omitted)")
    ap.add_argument("--out", default="results_c4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--engine", default="auto", choices=["auto", "pymedphys", "builtin"])
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = []
    if args.cases:
        cand += [Path(args.cases), Path.cwd() / args.cases]
    here = Path(__file__).resolve().parent
    cand += [here / "cases_closedloop.csv", here.parent / "cl_export" / "cases_closedloop.csv",
             here.parent / "cases_closedloop.csv", Path.cwd() / "cl_export" / "cases_closedloop.csv",
             Path.cwd() / "cases_closedloop.csv"]
    cf = next((c for c in cand if c.exists()), None)
    if cf is None:
        sys.exit("[error] cases_closedloop.csv not found; pass its full path via --cases")
    print("cases file:", cf.resolve())
    with open(cf, encoding="utf-8-sig") as f:
        cases = [dict(r) for r in csv.DictReader(f)]
    bases = [cf.resolve().parent, cf.resolve().parent.parent, Path.cwd()]
    for c in cases:
        for k in ("det_npz", "raw_path"):
            p = c.get(k)
            if p and not Path(p).exists():
                for bdir in bases:
                    q = bdir / p
                    if q.exists():
                        c[k] = str(q); break
                    q2 = bdir / Path(p).name
                    if q2.exists():
                        c[k] = str(q2); break
    if args.limit:
        cases = cases[:args.limit]
    print("read", len(cases), "cases")

    rows = []
    for i, case in enumerate(cases, 1):
        cid = case.get("case_id", "?")
        try:
            Z = np.load(case["det_npz"], allow_pickle=False)
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

            raw32, _m = V._load_raw_matrix(case["raw_path"], 32)
            meas_cg = (np.asarray(Z["meas_det_cg"], dtype=float)
                       if "meas_det_cg" in Z.files else None)
            o = rg.best_orientation(meas_cg if meas_cg is not None else meas_det, raw32)
            raw32 = o["transformed"]

            # per-case calibration fit (Table-5 recipe: recovered ~ a*raw + b, finite points only)
            m = np.isfinite(raw32) & np.isfinite(meas_det)
            a_, b_ = np.polyfit(raw32[m].ravel(), meas_det[m].ravel(), 1)
            rec_corr = (meas_det - b_) / a_

            # C5 companion control: correct the evaluation side (plan) with its own
            # per-case fit vs the DCM (a_p, b_p); the fit baseline is the DCM itself,
            # so (a_p, b_p) mixes the chain bias with TPS differences -- diagnostic only
            mc = (covered > 0) & np.isfinite(rec_plan_full) & np.isfinite(dcm_on_grid)
            a_p, b_p = np.polyfit(dcm_on_grid[mc].ravel(), rec_plan_full[mc].ravel(), 1)
            plan_corr = (rec_plan_full - b_p) / a_p

            dcm_evl = np.where(covered > 0, dcm_on_grid, 0.0)
            g = lambda ref, evl: V._gamma_pair(axes_det, ref, axes_disp, evl,
                                               dp, dd, cutoff_pct, gnorm, args.engine)[0]
            row = {"case_id": cid, "fit_a": a_, "fit_b": b_,
                   "g_rec_dcm_base": g(meas_det, dcm_evl),
                   "g_c4_dcm": g(rec_corr, dcm_evl),
                   "g_rec_plan_base": g(meas_det, rec_plan_full),
                   "g_c4_plan": g(rec_corr, rec_plan_full),
                   "fit_ap": a_p, "fit_bp": b_p,
                   "g_c5_plan_evcorr": g(meas_det, plan_corr),      # eval-side correction only -> expect ~g_rec_dcm_base
                   "g_c5_plan_bothcorr": g(rec_corr, plan_corr),    # both sides corrected  -> expect ~g_c4_dcm
                   "declared_gpr": declared}
            row["delta_dcm"] = row["g_c4_dcm"] - row["g_rec_dcm_base"]
            row["delta_plan"] = row["g_c4_plan"] - row["g_rec_plan_base"]
            rows.append(row)
            print(f"[{i}/{len(cases)}] {cid}: a={a_:.4f} b={b_:+.2f} | "
                  f"rec/dcm {row['g_rec_dcm_base']:.2f}->{row['g_c4_dcm']:.2f} "
                  f"({row['delta_dcm']:+.2f}) | rec/plan {row['g_rec_plan_base']:.2f}->"
                  f"{row['g_c4_plan']:.2f} ({row['delta_plan']:+.2f}) | "
                  f"C5: eval-corr={row['g_c5_plan_evcorr']:.2f} both-corr={row['g_c5_plan_bothcorr']:.2f}")
        except Exception as e:
            print(f"[{i}/{len(cases)}] {cid}: FAILED - {e}", file=sys.stderr)

    if not rows:
        sys.exit("[error] no case succeeded")
    cols = list(rows[0].keys())
    with open(out_dir / "c4_calibration.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    def med(k):
        v = np.asarray([r.get(k, np.nan) for r in rows], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    S = {"n_cases": len(rows), **{f"median_{k}": med(k) for k in cols if k != "case_id"}}
    (out_dir / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    dd_, dp_ = S["median_delta_dcm"], S["median_delta_plan"]
    ev_ = S.get("median_g_c5_plan_evcorr"); bo_ = S.get("median_g_c5_plan_bothcorr")
    print(f"\nmedians: delta(rec/dcm)={dd_:+.2f} pp, delta(rec/plan)={dp_:+.2f} pp")
    if ev_ is not None and bo_ is not None and np.isfinite(ev_):
        print(f"C5 four-cell closure: eval-corr only={ev_:.2f} (expect ~{S['median_g_rec_dcm_base']:.2f}) | "
              f"both corrected={bo_:.2f} (expect ~{S['median_g_c4_dcm']:.2f})")
    if dd_ >= +1.5 and dp_ <= -1.0:
        print("verdict: both directions moved -- common-mode amplitude bias confirmed "
              "(the penumbra/width explanation can be dropped)")
    elif dd_ < +0.5:
        print("verdict: no effect -- the bias is not the main cause; fall back to the "
              "geometry/registration controls (factor3x2_controls.py C1/C2)")
    else:
        print("verdict: partial -- the bias contributes but is not everything; "
              "report the C1/C2 results alongside")
    print("done.", out_dir / "c4_calibration.csv")


if __name__ == "__main__":
    main()
