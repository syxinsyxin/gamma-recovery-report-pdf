# -*- coding: utf-8 -*-
"""Export per-case artifacts from the v7 batch pipeline for the closed-loop validation.

This script does NOT modify batch_gamma_recovery_final_v7.py. It imports the v7
module and replays steps 1-6 of v7's process_case through the very same
functions (recover_map / register_dcm / to_detector / block_with_edges ...),
in the same order, so that the exported artifacts follow the exact code path of
the production results (same registration, same block-edge geometry, same
self-consistent dmin estimate).

If v7 is ever superseded, point --v7 at the new script and re-check the
"contract" comments below.

Per-case export (<out>/<case_id>_cl.npz):
  rec_meas, rec_plan        : PDF-recovered measured/plan maps (full resolution,
                              NaN = invalid, mm axes xmm/ymm)
  meas_det, cmp_det, dcm_det: 32x32 detector-grid versions (shared-geom block
                              average, v7 recipe)
  dcm_on_grid, covered      : RT Dose registered to the display grid + coverage
                              mask (v7 register_dcm output)
  det_y, det_x              : detector centre coordinates (mm, strictly increasing)
  dp, dd, cutoff_pct, gnorm, mode : per-case declared criteria and normalisation
                              basis, for the gamma re-analysis
plus <out>/cases_closedloop.csv, the case list consumed by validate.py.

Usage
-----
  python export_for_closedloop.py \
      --v7 batch_gamma_recovery_final_v7.py \
      --v7-results results25_26_new.csv \
      --root "<data root>" --ids ids30.txt \
      --raw-map raw_map.csv --out cl_export

  ids30.txt   : one case_id per line (the cohort selected from the 600-case table)
  raw_map.csv : two columns case_id,raw_path pointing at each 32x32 raw
                measurement CSV

Dependencies: everything v7 needs (pymupdf/pydicom/pymedphys/numpy/scipy/PIL).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registration as rg_mod  # noqa: E402  (in-package: orientation detection / scale fit)
from validate import _load_raw_matrix  # noqa: E402  (in-package: CSV coordinate parsing)


# ---------------------------------------------------------------------------
# v7 module loading
# ---------------------------------------------------------------------------

def load_v7(v7_path):
    path = os.path.abspath(v7_path)
    if not os.path.isfile(path):
        sys.exit(f"v7 script not found: {path}")
    spec = importlib.util.spec_from_file_location("batch_gamma_recovery_v7", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_results_csv(path):
    """Read the v7 results CSV (utf-8-sig or gb18030; skips repeated header rows
    left by resumed runs)."""
    text = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        sys.exit(f"cannot decode {path} (tried utf-8-sig/gb18030)")
    rows = [r for r in csv.DictReader(text.splitlines())
            if r.get("case_id") and r.get("case_id") != "case_id"]
    return {r["case_id"]: r for r in rows}


# ---------------------------------------------------------------------------
# Per-case export -- replays v7 process_case steps 1-6 (contract: line-by-line
# correspondence with v7)
# ---------------------------------------------------------------------------

def export_case(v7, pdf_path, dcm_path, raw32=None, det_x_csv=None, det_y_csv=None,
                rg=None):
    """Replay v7 process_case steps 1-6 and export the closed-loop artifacts.

    Three add-only extensions over v7 (the main results still share the v7
    production path):
    1) if extract_pdf_images fails (report variants without the
       Reference/Compare label words), self-identify the two dose images by NCC
       against the raw measurement CSV over all embedded page images
       (ref = highest NCC, cmp = runner-up); recorded in id_method/id_ncc;
    2) if the CSV coordinate axes are given (det_x_csv/det_y_csv, the physical
       detector coordinates exported by myQA), additionally produce meas_det_cg
       (CSV-geometry block average) as a diagnostic channel -- coordinates only,
       no dose values, no circularity;
    3) provenance check of the report line "A: <measurement source>" against the
       CSV header "Name:" (provenance_ok).
    """
    # ---- 1. report text and global parameters (contract: v7 step 1) ----
    with v7.fitz.open(str(pdf_path)) as doc:
        raw_text = "\n".join(pg.get_text("text") for pg in doc)
    rep = v7.parse_report_text(raw_text)
    dmax = rep["dmax_cgy"] or rep["poi_ref_max"]
    if not dmax:
        raise RuntimeError("dmax not available (same policy as v7: no hardcoded fallback)")
    gnorm = float(dmax)
    cutoff_pct = rep["thresh_pct"] or 5.0
    import re as _re
    m_src = _re.search(r"^A:\s*(.+?)\s*$", raw_text, _re.M)
    rep_src_a = m_src.group(1).strip() if m_src else ""

    # ---- 2. image extraction + self-consistent dmin + two-map recovery (contract: v7 step 2) ----
    id_method, id_ncc = "labels", np.nan
    try:
        ref_img, cmp_img = v7.extract_pdf_images(pdf_path, None)
    except Exception as e:
        if raw32 is None or rg is None:
            raise RuntimeError(
                f"v7 extract_pdf_images failed ({e}) and no raw CSV is available for the self-identify fallback"
            ) from e
        ref_img, cmp_img, id_ncc = _self_identify(v7, pdf_path, raw32, rg, gnorm)
        id_method = "selfid"
    rec_ref0, xmm, ymm, _, _ = v7.recover_map(ref_img, dmax, 0.0)
    if np.any(np.isfinite(rec_ref0)):
        dmin_est = float(np.nanpercentile(rec_ref0, 0.1))
    else:
        dmin_est = np.nan
    dmin = dmin_est if (np.isfinite(dmin_est) and 0.2 <= dmin_est <= 80) else 0.0
    rec_ref, xmm, ymm, _, _ = v7.recover_map(ref_img, dmax, dmin)
    rec_cmp, xmm_c, ymm_c, _, _ = v7.recover_map(cmp_img, dmax, dmin)

    # ---- 3. DCM loading (contract: v7 step 3) ----
    dcm_cgy, u_axis, v_axis, dcm_spacing = v7.read_dcm(dcm_path)

    # ---- 4. registration (contract: v7 step 4; parameters and gating identical) ----
    vf_c = np.isfinite(rec_cmp)
    X, Y = np.meshgrid(np.asarray(xmm, float), np.asarray(ymm, float))
    sx, sy, du, dv, nccv, dcm_on_grid, covered, _ = v7.register_dcm(
        np.where(vf_c, rec_cmp, 0.0), vf_c, X, Y, u_axis, v_axis, dcm_cgy)

    # ---- 5-6. detector grid and shared-geom block average
    #          (contract: v7 step 6, the 2x2 factorial sampling) ----
    meas_det, det_axes, det_counts, geom = v7.to_detector(rec_ref, xmm, ymm, v7.DET_N)
    if meas_det is None:
        raise RuntimeError("to_detector failed (this case is detgrid_fallback in v7 too; not usable for the closed loop)")
    cmp_det, _, fill_c = v7.block_with_edges(rec_cmp, geom, v7.DET_N)
    dcm_masked = np.where(covered, dcm_on_grid, np.nan)   # zeros outside the coverage must not enter the average
    dcm_det, _, fill_d = v7.block_with_edges(dcm_masked, geom, v7.DET_N)

    out = {
        "rec_meas": rec_ref, "rec_plan": rec_cmp,
        "xmm": np.asarray(xmm, float), "ymm": np.asarray(ymm, float),
        "meas_det": meas_det, "cmp_det": cmp_det, "dcm_det": dcm_det,
        "dcm_on_grid": dcm_on_grid, "covered": covered,
        "det_y": np.asarray(det_axes[0], float), "det_x": np.asarray(det_axes[1], float),
        "dp": rep["dp_rep"], "dd": rep["dd_rep"], "cutoff_pct": cutoff_pct,
        "gnorm": gnorm, "mode": rep["mode"] or "",
        "dmin": dmin, "reg_ncc": nccv, "reg_sx": sx, "reg_sy": sy,
        "reg_du": du, "reg_dv": dv,
        "det_fill_cmp": fill_c, "det_fill_dcm": fill_d,
        "id_method": id_method, "id_ncc": float(id_ncc),
        "rep_srcA": rep_src_a,
    }

    # ---- CSV-coordinate diagnostic channel (coordinate metadata only, no circularity) ----
    if det_x_csv is not None and det_y_csv is not None:
        meas_cg = _block_avg_csvgeom(rec_ref, np.asarray(ymm, float),
                                     np.asarray(xmm, float),
                                     np.asarray(det_y_csv, float),
                                     np.asarray(det_x_csv, float))
        if np.isfinite(meas_cg).sum() > 32 * 32 * 0.5:
            out["meas_det_cg"] = meas_cg
            out["det_y_csv"] = np.asarray(det_y_csv, float)
            out["det_x_csv"] = np.asarray(det_x_csv, float)
    return out


def _self_identify(v7, pdf_path, raw32, rg, gnorm):
    """Report variant without Reference/Compare label words: recover every
    embedded page image and match it against the raw measurement CSV over the
    8 dihedral orientations; highest NCC = measured map, runner-up = plan map."""
    import pymupdf

    scored = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                rs = page.get_image_rects(xref)
                if not rs:
                    continue
                pix = pymupdf.Pixmap(doc, xref)
                if pix.colorspace is not None and pix.colorspace.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                if pix.n < 3 or pix.alpha:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)[:, :, :3].copy()
                try:
                    rec0, xmm, ymm, _, _ = v7.recover_map(a, gnorm, 0.0)
                    if not np.any(np.isfinite(rec0)):
                        continue
                    est = float(np.nanpercentile(rec0, 0.1))
                    dmin = est if (np.isfinite(est) and 0.2 <= est <= 80) else 0.0
                    rec, xmm, ymm, _, _ = v7.recover_map(a, gnorm, dmin)
                    det, _, _, _ = v7.to_detector(rec, xmm, ymm, v7.DET_N)
                except Exception:
                    continue  # non-dose images such as histograms
                if det is None or np.isfinite(det).sum() < 32 * 32 * 0.5:
                    continue
                o = rg.best_orientation(det, raw32)
                scored.append((float(o["ncc"]), xref, a, o["orientation"]))
    scored.sort(key=lambda t: -t[0])
    dose_like = [s for s in scored if s[0] > 0.8]
    if len(dose_like) < 2:
        raise RuntimeError(
            f"self-identify failed: too few dose-like candidates ({len(dose_like)} with NCC>0.8, "
            f"all scores {[f'{s[0]:.3f}' for s in scored]})"
        )
    best = dose_like[0]
    second = dose_like[1]
    print(f"[selfid] ref=xref{best[1]}(NCC={best[0]:.4f}), "
          f"cmp=xref{second[1]}(NCC={second[0]:.4f})")
    return best[2], second[2], best[0]


def _block_avg_csvgeom(rec, ymm, xmm, yc, xc):
    """NaN-aware block average on the CSV coordinate axes (physical detector centres).

    Block edges = centres +/- pitch/2; the pitch is rebuilt as the median
    coordinate difference (avoids rounded label values)."""
    p = float(np.median(np.diff(yc)))
    px = float(np.median(np.diff(xc)))
    ey = np.append(yc - p / 2, yc[-1] + p / 2)
    ex = np.append(xc - px / 2, xc[-1] + px / 2)
    iy = np.digitize(np.asarray(ymm, float), ey) - 1
    ix = np.digitize(np.asarray(xmm, float), ex) - 1
    recf = np.where(np.isfinite(rec), rec, np.nan)
    out = np.full((len(yc), len(xc)), np.nan)
    for i in range(len(yc)):
        for j in range(len(xc)):
            m = (iy[:, None] == i) & (ix[None, :] == j)
            if m.any():
                v = recf[m]
                if np.isfinite(v).any():
                    out[i, j] = np.nanmean(v)
    return out


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

CASES_COLS = [
    "case_id", "raw_path", "det_npz", "dcm_path",
    "declared_gpr", "declared_mean_gamma",
    "dp", "dd", "cutoff_pct", "gnorm", "mode",
    "v7_gA_rep", "v7_gAdet_rep", "v7_gB_rep", "v7_gBdet_rep",
    "v7_devA_pp", "v7_devAdet_pp", "v7_devB_pp", "v7_devBdet_pp",
    "v7_gA_33", "v7_gB_33", "v7_flag", "v7_reg_ncc", "v7_det_pitch_x_mm", "v7_det_pitch_y_mm",
    "raw_unit", "raw_name", "rep_srcA", "provenance_ok",
    "pitch_from_coords", "id_method", "id_ncc",
]


def _load_raw_map(path):
    m = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = (r.get("case_id") or "").strip()
            rp = (r.get("raw_path") or "").strip()
            if cid and rp:
                m[cid] = rp
    return m


def _read_ids(path):
    ids = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                ids.append(t.split(",")[0].strip())
    return ids


def main():
    ap = argparse.ArgumentParser(description="export closed-loop validation artifacts from the v7 pipeline")
    ap.add_argument("--v7", required=True, help="path to batch_gamma_recovery_final_v7.py")
    ap.add_argument("--v7-results", required=True, help="v7 results summary CSV")
    ap.add_argument("--root", required=True, help="data root holding the PDF/DCM files")
    ap.add_argument("--ids", help="file with one case_id per line (default: all OK cases)")
    ap.add_argument("--raw-map", help="case_id,raw_path mapping CSV (exact; use either this or --raw-dir)")
    ap.add_argument("--raw-dir", help="directory of raw measurement CSVs (matched by {PDF/DCM stem}.csv or {case_id}.csv)")
    ap.add_argument("--out", default="cl_export", help="output directory")
    ap.add_argument("--skip-existing", action="store_true", help="skip already-exported cases (resume)")
    args = ap.parse_args()
    if not args.raw_map and not args.raw_dir:
        ap.error("one of --raw-map or --raw-dir is required to locate the 32x32 raw measurement CSVs")

    v7 = load_v7(args.v7)
    results = read_results_csv(args.v7_results)
    raw_map = _load_raw_map(args.raw_map) if args.raw_map else {}
    ids = _read_ids(args.ids) if args.ids else sorted(
        k for k, r in results.items() if r.get("status") == "OK")
    os.makedirs(args.out, exist_ok=True)

    def resolve_raw(cid, r, pdf_path, dcm_path):
        """Raw CSV resolution order: raw-map -> raw-dir/{PDF stem}.csv -> raw-dir/{case_id}.csv
        -> same-stem .csv next to the PDF -> same-stem .csv next to the DCM
        (same pairing convention as v7)."""
        cands = []
        if raw_map.get(cid):
            cands.append(raw_map[cid])
        if args.raw_dir:
            stem = os.path.splitext(os.path.basename(r["rel_pdf"]))[0]
            cands += [os.path.join(args.raw_dir, stem + ".csv"),
                      os.path.join(args.raw_dir, cid + ".csv")]
        cands += [os.path.splitext(pdf_path)[0] + ".csv",
                  os.path.splitext(dcm_path)[0] + ".csv"]
        for c in cands:
            if c and os.path.isfile(c):
                return c
        raise RuntimeError(f"raw measurement CSV not found (tried: {cands})")

    cases_rows, n_ok, n_fail = [], 0, 0
    cases_csv = os.path.join(args.out, "cases_closedloop.csv")
    done = set()
    if args.skip_existing and os.path.isfile(cases_csv):
        with open(cases_csv, encoding="utf-8-sig") as f:
            done = {r["case_id"] for r in csv.DictReader(f)}
        with open(cases_csv, encoding="utf-8-sig") as f:
            cases_rows = list(csv.DictReader(f))

    for i, cid in enumerate(ids, 1):
        if args.skip_existing and cid in done:
            continue
        r = results.get(cid)
        if r is None:
            print(f"[{i}/{len(ids)}] {cid}: not in the v7 results table, skipped", file=sys.stderr)
            continue
        if r.get("status") != "OK":
            print(f"[{i}/{len(ids)}] {cid}: v7 status={r.get('status')}, skipped", file=sys.stderr)
            continue
        try:
            pdf_path = os.path.join(args.root, r["rel_pdf"])
            dcm_path = os.path.join(args.root, r["rel_dcm"])
            raw_path = resolve_raw(cid, r, pdf_path, dcm_path)
            raw32, raw_meta = _load_raw_matrix(raw_path, 32)
            art = export_case(
                v7, pdf_path, dcm_path, raw32=raw32,
                det_x_csv=raw_meta.get("det_x"), det_y_csv=raw_meta.get("det_y"),
                rg=rg_mod)
            # provenance check: report line "A: <measurement source>" vs CSV header "Name:"
            raw_name = raw_meta.get("raw_name", "")

            def _norm(s):
                return " ".join(str(s).split()).lower()

            prov_ok = ""
            if raw_name and art.get("rep_srcA"):
                prov_ok = bool(_norm(raw_name) == _norm(art["rep_srcA"]))
            art["raw_unit"] = raw_meta.get("raw_unit", "")
            art["raw_name"] = raw_name
            art["provenance_ok"] = prov_ok
            art["pitch_from_coords"] = raw_meta.get("pitch_from_coords", "")
            npz_path = os.path.abspath(os.path.join(args.out, f"{cid}_cl.npz"))
            np.savez_compressed(npz_path, **art)
            fills = [f for f in (art["det_fill_cmp"], art["det_fill_dcm"])
                     if np.isfinite(f)]
            row = {
                "case_id": cid,
                "raw_path": os.path.abspath(raw_path),
                "det_npz": npz_path,
                "dcm_path": os.path.abspath(dcm_path),
                "declared_gpr": r.get("rep_rep_pass", ""),
                "declared_mean_gamma": r.get("rep_rep_avg_gamma", ""),
                "dp": art["dp"], "dd": art["dd"], "cutoff_pct": art["cutoff_pct"],
                "gnorm": art["gnorm"], "mode": art["mode"],
                "v7_gA_rep": r.get("gA_rep", ""), "v7_gAdet_rep": r.get("gAdet_rep", ""),
                "v7_gB_rep": r.get("gB_rep", ""), "v7_gBdet_rep": r.get("gBdet_rep", ""),
                "v7_devA_pp": r.get("devA_pp", ""), "v7_devAdet_pp": r.get("devAdet_pp", ""),
                "v7_devB_pp": r.get("devB_pp", ""), "v7_devBdet_pp": r.get("devBdet_pp", ""),
                "v7_gA_33": r.get("gA_33", ""), "v7_gB_33": r.get("gB_33", ""),
                "v7_flag": r.get("flag", ""), "v7_reg_ncc": r.get("reg_ncc", ""),
                "v7_det_pitch_x_mm": r.get("det_pitch_x_mm", ""),
                "v7_det_pitch_y_mm": r.get("det_pitch_y_mm", ""),
                "raw_unit": art.get("raw_unit", ""), "raw_name": raw_name,
                "rep_srcA": art.get("rep_srcA", ""), "provenance_ok": prov_ok,
                "pitch_from_coords": art.get("pitch_from_coords", ""),
                "id_method": art.get("id_method", ""), "id_ncc": art.get("id_ncc", ""),
            }
            cases_rows.append(row)
            n_ok += 1
            fill_str = f"{min(fills):.3f}" if fills else "NA"
            print(f"[{i}/{len(ids)}] {cid}: OK  gnorm={art['gnorm']:.1f} "
                  f"reg_ncc={art['reg_ncc']:.4f} det_fill(min)={fill_str} raw={raw_path}")
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(ids)}] {cid}: FAILED - {e}", file=sys.stderr)

    # rewrite cases_closedloop.csv in full (keeps the column order stable)
    with open(cases_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CASES_COLS)
        w.writeheader()
        for row in cases_rows:
            w.writerow({k: row.get(k, "") for k in CASES_COLS})
    print(f"\ndone: {n_ok} ok, {n_fail} failed. case list: {cases_csv}")
    print("next step: python validate.py --cases {cases_csv} --out results_cl/")


if __name__ == "__main__":
    main()
