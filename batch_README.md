# Dose recovery and γ re-analysis from archived VMAT verification report PDFs

Companion code for the manuscript *"Dose recovery and γ re-analysis from archived VMAT verification report PDFs: a 600-plan study"*.

For each case, the script re-reads the archived myQA verification report (PDF),
rebuilds the measured and planned dose maps from the embedded colour-scale
images, and re-computes γ against the Eclipse DICOM RT Dose export (DCM) of the
same plan. It is a batch tool: one PDF/DCM pair in, one CSV row out, plus
optional QA figures. The whole pipeline is the single file
`batch_gamma_recovery.py`; there are no private dependencies and no compiled
components.

Quick start:

```bash
pip install numpy scipy pillow pymupdf pydicom pymedphys
python batch_gamma_recovery.py <root> --out <outdir>
python batch_gamma_recovery.py <root> --limit 10   # trial run on the first 10 cases
```

`<root>` is any directory containing the report PDFs and their DCM exports
(§2). All options are listed in §3; a run can be interrupted and resumed at
any time.

---

## 1. Requirements

| Item | Version used | Notes |
|---|---|---|
| Python | 3.12 | developed and tested on 3.12; no 3.12-only syntax is relied on |
| numpy, scipy, pillow | any recent | array work, median/morphological filters, image output |
| pymupdf (PyMuPDF) | ≥ 1.24 | the script imports `pymupdf` and falls back to `fitz` |
| pydicom | any recent | DICOM RT Dose reading |
| pymedphys | 0.41.0 | version used for the study; the code does not pin it — the γ signature is probed at import (see below) |

```bash
pip install numpy scipy pillow pymupdf pydicom pymedphys
```

`pymedphys` is the only dependency worth pinning. The study was run with
0.41.0, so `pip install pymedphys==0.41.0` reproduces it exactly. The code
itself does not require that version: at import it probes the signature of
`pymedphys.gamma` and drops unsupported keywords with a warning instead of
raising `TypeError`. A dropped `interp_fraction` lowers the sub-pixel
interpolation density and can bias pass rates low, so check the warnings at
the start of a run.

## 2. Input data

```
<root>/
├── batch_A/
│   ├── patient_001.pdf     <- verification report exported by myQA
│   └── patient_001.dcm     <- Eclipse DICOM RT Dose export of the same plan
└── batch_B/
    └── ...
```

* `XXX.pdf` is paired with `XXX.dcm` by file stem, case-insensitively.
* A pair is accepted only when both files sit in the **same parent directory**,
  so same-named patients from different dates or batches are never matched.
* Unmatched files are listed in `skipped.csv` and do not stop the run.
* Reports must follow the standard two-map template: a Reference (measured)
  map and a Compare (planned) map, each 550 × 360 px with a 66-level colour
  bar. The image size acts as a guard: if tick detection fails and the image
  is not 550 × 360 px, the case errors out instead of falling back to the
  prototype calibration. A different template therefore needs its axis and
  colour-bar parameters re-fitted (see §10).

## 3. Usage

```bash
# full run with an explicit output directory
python batch_gamma_recovery.py "D:/QA/archives" --out D:/out

# trial run on the first 10 cases
python batch_gamma_recovery.py "D:/QA/archives" --limit 10

# four shards in four terminals
python batch_gamma_recovery.py "D:/QA/archives" --shard 0 4
python batch_gamma_recovery.py "D:/QA/archives" --shard 1 4

# skip the reversed-reference control (roughly halves the run time)
python batch_gamma_recovery.py "D:/QA/archives" --no-ab
```

| Option | Meaning |
|---|---|
| `root` | directory scanned recursively for PDF/DCM pairs |
| `--out` | output directory (default `<root>/_gamma_output2026`) |
| `--limit N` | process the first N pairs only |
| `--shard I N` | process shard I of N; results go to shard-specific files |
| `--all-qa` | write QA figures for every case, not only flagged ones |
| `--no-ab` | skip the reversed-reference γ (`g_plan_*_dirT`) |
| `--fill-thresh` | threshold for the `low_det_fill` flag (default 0.6; the study ran with 0.45) |
| `--debug` | print the full traceback on error |

The run is resumable: cases already marked `OK` in `results.csv` are skipped,
`ERROR` cases are retried, and the CSV is flushed to disk every 10 cases.

## 4. Outputs

```
<out>/
├── results.csv                        one row per processed pair
├── skipped.csv                        files that could not be paired
├── summary.txt                        cohort summary and factorial read-out
├── extreme_cases_dplan_gt5pp.csv      cases with |Δ_plan| > 5 pp, key columns only
└── qa_images/
    ├── <case>_gplan.png               γ_plan map
    ├── <case>_gDCM.png                γ_DCM map
    └── <case>_maps.png                measured | recovered TPS | DCM side by side
```

With `--shard I N`, `results.csv` / `skipped.csv` / `summary.txt` carry a
`_shardIofN` suffix. `results.csv` is written as `utf-8-sig`, so Excel opens
it without mojibake.

## 5. γ pathways

Every pathway runs the same γ settings (`interp_fraction = 10`,
`max_gamma = 2.0`, `local_gamma = False`, global normalisation to the
reference maximum, 5% lower-dose cutoff). Two criteria sets are computed for
each pathway:

* `_rep` — the criteria declared in that report (3%/3 mm, 5%/3 mm or 5%/5 mm);
* `_33` — 3%/3 mm for every case.

| Column stem | Paper symbol | Reference | Evaluation | Purpose |
|---|---|---|---|---|
| `g_plan` | γ_plan | recovered measured (32×32) | recovered TPS map (fine grid) | main result: reproduction of the declared pass rate |
| `g_plan_det` | γ_plan,det | recovered measured (32×32) | recovered TPS map, block-averaged | removes block averaging itself as a confounder |
| `g_plan_sm` | γ_plan,sm | recovered measured (32×32) | recovered TPS map, aperture-smoothed on the fine grid | aperture effect with sampling density unchanged |
| `g_DCM` | γ_DCM | recovered measured (32×32) | DCM (fine grid) | independent reference |
| `g_DCM_det` | γ_DCM,det | recovered measured (32×32) | DCM, block-averaged | sampling-matched DCM comparison |
| `g_DCM_sm` | γ_DCM,sm | recovered measured (32×32) | DCM, aperture-smoothed on the fine grid | sampling-matched DCM comparison, fine sampling kept |
| `g_Self` | γ_Self | recovered TPS map | DCM | TPS self-consistency (3%/3 mm only) |
| `g_plan_..._dirT` | γ_plan reversed | recovered TPS map | recovered measured map | reversed reference convention (direction control) |

Each stem has a companion `<stem>_mean` column holding the mean γ of that map.

## 6. Derived columns (percentage points)

Δ columns compare a recovered pass rate with the value declared in the report;
the remaining columns are paired differences between pathways (the factorial
effects of §2.7 of the manuscript).

| Column | Definition |
|---|---|
| `d_plan_pp`, `d_DCM_pp` | γ_plan, γ_DCM minus the declared pass rate (Δ_plan, Δ_DCM) |
| `d_plan_det_pp`, `d_DCM_det_pp` | γ_plan,det, γ_DCM,det minus the declared pass rate |
| `d_plan_sm_pp`, `d_DCM_sm_pp` | γ_plan,sm, γ_DCM,sm minus the declared pass rate |
| `d_plan_dirT_pp` | reversed-convention γ_plan minus the declared pass rate |
| `d_ds_plan_pp` | γ_plan,det − γ_plan (downsampling effect, plan evaluation) |
| `d_ds_DCM_pp` | γ_DCM,det − γ_DCM (downsampling effect, DCM evaluation) |
| `d_src_det_pp` | γ_DCM,det − γ_plan,det (source effect under detector-grid sampling) |
| `d_ap_plan_pp` | γ_plan,sm − γ_plan (aperture effect, plan evaluation) |
| `d_ap_DCM_pp` | γ_DCM,sm − γ_DCM (aperture effect, DCM evaluation) |
| `d_src_sm_pp` | γ_DCM,sm − γ_plan,sm (source effect under aperture matching) |

`summary.txt` reports the median, mean and SD of these columns and reads the
aperture result as the **interaction** `d_ap_DCM_pp − d_ap_plan_pp`, not as the sign of
`d_ap_DCM_pp` alone: smoothing degrades both evaluation paths, so only the
difference shows whether aperture matching narrows the gap between γ_DCM and γ_plan.

## 7. Report and registration columns

| Column | Meaning |
|---|---|
| `rep_dp_rep`, `rep_dd_rep`, `rep_thresh_pct` | dose-difference and distance-to-agreement criteria and the low-dose threshold declared in the report |
| `rep_rep_pass` | pass rate printed in the report |
| `rep_rep_avg_gamma` | average γ stored in the report (`AverageValue`) |
| `rep_dmax_cgy`, `est_dmin_cgy` | upper colour-scale limit (report) and lower limit (estimated by the script, not a report text field) |
| `rep_poi_ref_max`, `rep_poi_cmp_max`, `rep_poi_*_cursor` | point-of-interest table values |
| `rec_ref_cursor`, `rec_cmp_cursor`, `dev_ref_cursor`, `dev_cmp_cursor` | cursor read-out of the recovered maps and its deviation from the report |
| `dcm_max_cgy`, `dcm_pixels`, `dcm_spacing_mm`, `dcm_vs_poi_max` | DCM diagnostics; `dcm_vs_poi_max` is the quantity used by the dose-magnitude gate of section 2.2 |
| `reg_ncc`, `reg_sx`, `reg_sy`, `reg_du`, `reg_dv`, `reg_coverage` | registration quality and transform |
| `dose_bias`, `dose_sigma` | mean and SD of (recovered TPS − registered DCM) over all covered points |
| `dose_sigma_smooth`, `dose_sigma_grad` | SD of the same difference in the low-gradient (<Q1) and high-gradient (≥Q3) strata; left empty when a stratum has ≤10 points |
| `uncov_pts`, `sat_pct` | points above the cutoff outside the DCM coverage, and percentage of saturated pixels |
| `det_n`, `det_pitch_x_mm`, `det_pitch_y_mm`, `det_pts` | detector grid size, effective pitch and number of valid detector points |
| `det_fill_min`, `det_fill_meas`, `det_fill_gap` | worst-block valid-pixel fraction on the evaluation side (`det_fill_min`, the smaller of the TPS-map and DCM blocks), on the measured side, and their gap |
| `aperture_x_px`, `aperture_y_px` | anisotropic aperture kernel, in fine-grid pixels |

Note that `det_fill_meas` / `det_fill_gap` quantify how strictly "same
sampling" holds: the shared geometry fixes the block boundaries but not the
set of valid pixels inside each block.

## 8. QC flags

`flag` holds `|`-separated tokens; a case can carry several.

| Flag | Raised when |
|---|---|
| `cursor_dev` | POI cursor read-out of the recovered map deviates by more than 5% of the case maximum dose from the report value |
| `low_det_fill` | the worst evaluation-side detector block has a valid-pixel fraction below `--fill-thresh` (default 0.6; study value 0.45) |
| `reg_seed_escalated` | NCC stayed below 0.98 for the first two seeds and the remaining flipped seeds were tried |
| `low_reg_ncc` | final NCC marginally below the 0.98 gate |
| `calib_fallback` | axis calibration fell back to the prototype fit |
| `cmp_axis_mismatch` | the axes of the two maps disagree by more than 0.5 mm at the end points |
| `det_pitch_off` | effective detector pitch outside ±12% of the nominal 7.62 mm |
| `detgrid_fallback` | downsampling to the detector grid failed and the fine grid was used instead |
| `dmax_fallback_poi` | colour-bar maximum unavailable; the POI reference maximum was used instead |
| `d_plan_gt_6pp`, `d_plan_gt_3pp` | γ_plan deviates from the declared pass rate by more than the gate (6 pp under strict criteria, 3 pp under second-line criteria) |

Flags are screening aids. They do not remove a case from the analysis; the
hierarchical exclusion of section 2.2 is applied to `results.csv` afterwards,
using `status`, `dcm_vs_poi_max`, `reg_ncc`, `reg_coverage` and `flag`.

## 9. Where the manuscript sections live in the code

| Manuscript | Function / block | Key columns |
|---|---|---|
| 2.1 Cases and paired data | `pair_files()`, `extract_pdf_images()` | `rel_pdf`, `rel_dcm` |
| 2.2 Analysis set definition | `process_case()` error paths; gates are applied to the CSV afterwards | `status`, `dcm_vs_poi_max`, `reg_ncc`, `reg_coverage`, `flag` |
| 2.3 Report data extraction and dose-map recovery | `parse_report_text()`, `extract_pdf_images()`, `detect_frame()`, `detect_ticks()`, `fit_axis_auto()`, `detect_colorbar()`, `invert_curve()`, `clean_and_fill()`, `recover_map()`, `cursor_readout()` | `rep_*`, `est_dmin_cgy`, `dev_ref_cursor` |
| 2.4 DCM registration | `read_dcm()`, `ncc()`, `register_dcm()` | `reg_*` |
| 2.5 Downsampling and aperture matching | `to_detector()`, `block_with_edges()`, `_block_edges()`, `box_average()`, `nan_uniform_filter()` | `det_*`, `aperture_*` |
| 2.6 γ calculation and direction convention | `run_gamma()` | `g_plan_*`, `g_plan_*_dirT` |
| 2.7 Factorial decomposition design | γ pathway block of `process_case()` | `g_plan_det_*`, `g_DCM_det_*`, `d_*` |
| 2.8 Gradient-stratified dose-domain error | dose-domain block of `process_case()` | `dose_bias`, `dose_sigma`, `dose_sigma_smooth`, `dose_sigma_grad` |
| 2.9 Average γ embedded in the report | `parse_report_text()`, γ pathway block | `rep_rep_avg_gamma`, `g_plan_rep_mean` |
| 2.10 Closed-loop validation | same script, run on the 20 cases that retained their original matrix CSV; the 3 × 2 design and the inverse correction are applied to `results.csv` | `g_plan_*`, `g_DCM_*`, `g_Self_33` |
| 2.11 Statistical methods | `summary.txt` block; the remaining statistics are computed from `results.csv` | all |

## 10. Reproducibility notes and known limits

* **Software versions.** The study ran Python 3.12 and pymedphys 0.41.0.
  Other pymedphys versions run too, but unsupported γ keywords are dropped
  with a warning (see §1); a dropped `interp_fraction` may bias pass rates.
* **Report-template dependence.** Frame and tick detection, the 550 × 360 px
  expectation and the 66-level colour bar are properties of the myQA report
  template used here. A different template fails the dimension check or the
  tick count rather than producing silently wrong dose maps. Because the
  colour mapping is rebuilt from the colour bar of each report (not from a
  vendor palette), a change of colour scale alone needs no parameter change.
* **Direction convention.** The measured distribution is the reference and the
  planned dose the evaluation, matching the myQA configuration. The reversed
  convention is available as `g_plan_*_dirT` for verification; it lowers the pass
  rate by roughly 6 pp and is not the operating mode.
* **Search radius.** The effective search radius is 2 × DTA (6 mm at 3 mm/3%),
  whereas the myQA report states 4.5 mm. Points beyond the DTA already have a
  distance term > 1, so pass rates are unaffected; mean γ can shift slightly,
  which is why it is reported as a separate endpoint.
* **Covered region.** Dose outside the DCM coverage is set to 0 on every
  evaluation map and removed by the lower-dose cutoff, so it cannot lower the
  pass rate.
* **Single-frame only.** Multi-frame DICOM RT Dose objects are rejected rather
  than averaged.
* **Two-pass inversion.** `recover_map()` is called twice per map (a warm-up
  pass with `dmin = 0` to estimate the colour-scale lower limit, then the
  final pass). This costs about 1–2 s per case and is kept for clarity; it
  would be the first place to look if run time ever matters.
* **No patient data.** The script writes no images by default except for
  flagged cases, and the QA figures contain dose maps only.

## 11. Licence and citation

Code licence: [to be completed]. If you use this code, please cite the
manuscript above and the archived release (DOI: [to be completed]).
