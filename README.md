# Dose recovery and γ re-analysis from archived VMAT verification report PDFs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24441874.svg)](https://doi.org/10.5281/zenodo.24441874)

Code accompanying the manuscript *"Dose recovery and γ re-analysis from archived VMAT verification report PDFs: a 600-plan study"*.

Archived release: [10.5281/zenodo.24441874](https://doi.org/10.5281/zenodo.24441874) · Repository: <https://github.com/syxinsyxin/gamma-recovery-report-pdf>

The script reconstructs, from each archived myQA verification report PDF, the measured
and planned dose maps embedded as colour-scale images, then re-computes γ against the
Eclipse DICOM RT Dose export (DCM) of the same plan. It is a batch tool: one pair of
files in, one CSV row plus optional QA figures out.

Everything needed for a full run is in this single file; there are no private
dependencies and no compiled components.

---

## 1. Requirements

| Item | Version used | Notes |
|---|---|---|
| Python | 3.12.7 | developed and tested on 3.12; no 3.12-only syntax is relied on |
| numpy | 1.26.4 | array work |
| scipy | 1.13.1 | median and morphological filters, interpolation |
| pillow | 10.4.0 | image output |
| pymupdf (PyMuPDF) | 2.0 | MuPDF 1.28.5 bindings; the script imports `pymupdf` and falls back to `fitz` |
| pydicom | 3.0.1 | DICOM RT Dose reading |
| pymedphys | 0.41.0 | the γ implementation; the signature is probed at import |

Exact versions are pinned in `requirements.txt`.

```bash
pip install numpy scipy pillow pymupdf pydicom pymedphys
```

If `pymedphys` is older than the version used here, unsupported keywords are dropped with a
warning rather than raising `TypeError`. A dropped `interp_fraction` lowers the sub-pixel
interpolation density and can bias pass rates low, so check the warnings at the start of a run.

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
* A pair is accepted only when both files sit in the **same parent directory**, which
  prevents same-named patients from different dates or batches being matched.
* Unmatched files are listed in `skipped.csv` and do not stop the run.
* Reports must be the standard two-map template: a Reference (measured) map and a
  Compare (planned) map, each 550 × 360 px with a 66-level colour bar. The image size is
  used as a guard: if tick detection fails and the image is not 550 × 360 px, the error is
  raised instead of falling back to the prototype calibration. A different template
  therefore needs re-fitting of the axis and colour-bar parameters (see §10).

## 3. Usage

```bash
# full run with an explicit root
python batch_gamma_recovery_final_v7.py "D:/QA/archives" --out D:/out

# trial run on 10 cases first
python batch_gamma_recovery_final_v7.py "D:/QA/archives" --limit 10

# four shards in four terminals
python batch_gamma_recovery_final_v7.py "D:/QA/archives" --shard 0 4
python batch_gamma_recovery_final_v7.py "D:/QA/archives" --shard 1 4

# skip the reversed-reference control (roughly halves the run time)
python batch_gamma_recovery_final_v7.py "D:/QA/archives" --no-ab
```

| Option | Meaning |
|---|---|
| `root` | directory scanned recursively for PDF/DCM pairs |
| `--out` | output directory (default `<root>/_gamma_output2026`) |
| `--limit N` | process the first N pairs only |
| `--shard I N` | process shard I of N; results go to shard-specific files |
| `--all-qa` | write QA figures for every case, not only flagged ones |
| `--no-ab` | skip the reversed-reference γ (`gA_*_dirT`) |
| `--fill-thresh` | threshold for the `low_det_fill` flag (default 0.6) |
| `--debug` | print the full traceback on error |

The run is resumable: cases already marked `OK` in `results.csv` are skipped, while
`ERROR` cases are retried. Progress is flushed to disk every 10 cases.

## 4. Outputs

```
<out>/
├── results.csv                        one row per processed pair
├── skipped.csv                        files that could not be paired
├── summary.txt                        cohort summary and factorial read-out
├── extreme_cases_devA_gt5pp.csv       cases with |devA| > 5 pp, key columns only
└── qa_images/
    ├── <case>_gA.png                  γ_A map
    ├── <case>_gB.png                  γ_B map
    └── <case>_maps.png                measured | recovered TPS | DCM side by side
```

`results.csv` uses `utf-8-sig` so that Excel opens it without mojibake.

## 5. γ pathways

Every pathway uses the same γ implementation (`interp_fraction = 10`, `max_gamma = 2.0`,
`local_gamma = False`, global normalisation to the reference maximum, 5% lower-dose
cutoff). Two criteria sets are computed for each pathway:

* `_rep` — the criteria declared in that report (3%/3 mm, 5%/3 mm or 5%/5 mm);
* `_33` — 3%/3 mm for every case.

| Column stem | Reference | Evaluation | Purpose |
|---|---|---|---|
| `gA` | recovered measured (32×32) | recovered TPS map (fine grid) | main result: reproduction of the declared pass rate |
| `gAdet` | recovered measured (32×32) | recovered TPS map, block-averaged | removes block averaging itself as a confounder |
| `gAsm` | recovered measured (32×32) | recovered TPS map, aperture-smoothed on the fine grid | aperture effect with sampling density unchanged |
| `gB` | recovered measured (32×32) | DCM (fine grid) | independent reference |
| `gBdet` | recovered measured (32×32) | DCM, block-averaged | sampling-matched DCM comparison |
| `gBsm` | recovered measured (32×32) | DCM, aperture-smoothed on the fine grid | sampling-matched DCM comparison, fine sampling kept |
| `gSelf` | recovered TPS map | DCM | TPS self-consistency (3%/3 mm only) |
| `gA_..._dirT` | recovered TPS map | recovered measured map | reversed reference convention (direction control) |

Each stem has a companion `<stem>_mean` column holding the mean γ of that map.

## 6. Derived columns (percentage points)

| Column | Definition |
|---|---|
| `devA_pp`, `devB_pp` | γ_A, γ_B minus the declared pass rate |
| `devAdet_pp`, `devBdet_pp` | γ_A,det, γ_B,det minus the declared pass rate |
| `devAsm_pp`, `devBsm_pp` | γ_A,sm, γ_B,sm minus the declared pass rate |
| `devA_dirT_pp` | reversed-convention γ minus the declared pass rate |
| `d_same_pp` | γ_A,det − γ_A (downsampling effect, TPS map) |
| `d_same_B_pp` | γ_B,det − γ_B (downsampling effect, DCM) |
| `d_src_det_pp` | γ_B,det − γ_A,det (source effect under matched sampling) |
| `d_ap_A_pp` | γ_A,sm − γ_A (aperture effect, TPS map) |
| `d_ap_B_pp` | γ_B,sm − γ_B (aperture effect, DCM) |
| `d_src_sm_pp` | γ_B,sm − γ_A,sm (source effect under aperture matching) |

`summary.txt` reports the median, mean and SD of these columns and reads the aperture
result as the **interaction** `d_ap_B − d_ap_A`, not as the sign of `d_ap_B` alone:
smoothing degrades both evaluation paths, so only the difference reflects whether
aperture matching narrows the gap between γ_B and γ_A.

## 7. Report and registration columns

| Column | Meaning |
|---|---|
| `rep_dp_rep`, `rep_dd_rep`, `rep_thresh_pct` | dose-difference and distance-to-agreement criteria and the low-dose threshold declared in the report |
| `rep_rep_pass` | pass rate printed in the report |
| `rep_rep_avg_gamma` | average γ stored in the report (`AverageValue`) |
| `rep_dmax_cgy`, `est_dmin_cgy` | upper colour-scale limit (report) and lower limit (self-consistent estimate, not a text field) |
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

Note that `det_fill_meas`/`det_fill_gap` quantify how strictly "same sampling" holds:
the shared geometry fixes the block boundaries but not the set of valid pixels inside
each block.

## 8. QC flags

`flag` holds `|`-separated tokens; a case can carry several.

| Flag | Raised when |
|---|---|
| `cursor_dev` | POI cursor read-out of the recovered map deviates by more than 5% of the case maximum dose from the report value |
| `low_det_fill` | the worst evaluation-side detector block has a valid-pixel fraction below `--fill-thresh` (default 0.6) |
| `reg_seed_escalated` | NCC stayed below 0.98 for the first two seeds and the remaining flipped seeds were tried |
| `low_reg_ncc` | final NCC marginally below the 0.98 gate |
| `calib_fallback` | axis calibration fell back to the prototype fit |
| `cmp_axis_mismatch` | the axes of the two maps disagree by more than 0.5 mm at the end points |
| `det_pitch_off` | effective detector pitch outside ±12% of the nominal 7.62 mm |
| `detgrid_fallback` | downsampling to the detector grid failed and the fine grid was used instead |
| `dmax_fallback_poi` | colour-bar maximum unavailable; the POI reference maximum was used instead |
| `devA_gt_6pp`, `devA_gt_3pp` | γ_A deviates from the declared pass rate by more than the gate (6 pp under strict criteria, 3 pp under second-line criteria) |

Flags are screening aids. They do not remove a case from the analysis; the hierarchical
exclusion of section 2.2 is applied to `results.csv` afterwards, using `status`,
`dcm_vs_poi_max`, `reg_ncc`, `reg_coverage` and `flag`.

## 9. Where the manuscript sections live in the code

| Manuscript | Function / block | Key columns |
|---|---|---|
| 2.1 Cases and paired data | `pair_files()`, `extract_pdf_images()` | `rel_pdf`, `rel_dcm` |
| 2.2 Analysis set definition | `process_case()` error paths; gates are applied to the CSV afterwards | `status`, `dcm_vs_poi_max`, `reg_ncc`, `reg_coverage`, `flag` |
| 2.3 Report data extraction and dose-map recovery | `parse_report_text()`, `extract_pdf_images()`, `detect_frame()`, `detect_ticks()`, `fit_axis_auto()`, `detect_colorbar()`, `invert_curve()`, `clean_and_fill()`, `recover_map()`, `cursor_readout()` | `rep_*`, `est_dmin_cgy`, `dev_ref_cursor` |
| 2.4 DCM registration | `read_dcm()`, `ncc()`, `register_dcm()` | `reg_*` |
| 2.5 Downsampling and aperture matching | `to_detector()`, `block_with_edges()`, `_block_edges()`, `box_average()`, `nan_uniform_filter()` | `det_*`, `aperture_*` |
| 2.6 γ calculation and direction convention | `run_gamma()` | `gA_*`, `gA_*_dirT` |
| 2.7 Factorial decomposition design | γ pathway block of `process_case()` | `gAdet_*`, `gBdet_*`, `d_*` |
| 2.8 Gradient-stratified dose-domain error | dose-domain block of `process_case()` | `dose_bias`, `dose_sigma`, `dose_sigma_smooth`, `dose_sigma_grad` |
| 2.9 Average γ embedded in the report | `parse_report_text()`, γ pathway block | `rep_rep_avg_gamma`, `gA_rep_mean` |
| 2.10 Closed-loop validation | same script, run on the 20 cases that retained their original matrix CSV; the 3 × 2 design and the inverse correction are applied to `results.csv` | `gA_*`, `gB_*`, `gSelf_33` |
| 2.11 Statistical methods | `summary.txt` block; the remaining statistics are computed from `results.csv` | all |

## 10. Reproducibility notes and known limits

* **Report-template dependence.** Frame and tick detection, the 550 × 360 px expectation
  and the 66-level colour bar are properties of the myQA report template used here. A
  different template fails the dimension check or the tick count, rather than producing
  silently wrong dose maps. Because the colour mapping is rebuilt from the colour bar of
  each report rather than from a vendor palette, a change of colour scale alone needs no
  parameter change.
* **Direction convention.** The measured distribution is the reference and the planned
  dose the evaluation, matching the myQA configuration. The reversed convention is
  available as `gA_*_dirT` for verification; it lowers the pass rate by roughly 6 pp and
  is not the operating mode.
* **Search radius.** The effective search radius is 2 × DTA (6 mm at 3 mm/3%),
  whereas the myQA report states 4.5 mm. Points beyond the DTA already have a distance
  term > 1, so this does not change pass rates; it can shift mean γ slightly, which is
  why mean γ is reported as a separate endpoint.
* **Covered region.** Dose outside the DCM coverage is set to 0 on every evaluation map
  and removed by the lower-dose cutoff, so it cannot lower the pass rate.
* **Single-frame only.** Multi-frame DICOM RT Dose objects are rejected rather than
  averaged.
* **Two-pass inversion.** `recover_map()` is called twice per map (a warm-up pass with
  `dmin = 0` to estimate the colour-scale lower limit, then the final pass). This costs
  about 1–2 s per case and is kept for clarity; it is the obvious target if the run time
  needs to be reduced.
* **No patient data.** The script writes no images by default except for flagged cases,
  and the QA figures contain dose maps only.

## 11. Licence and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite
the manuscript above and the archived release:

> Xin S. Gamma recovery and re-analysis from archived VMAT verification report PDFs.
> Zenodo (2026). doi:10.5281/zenodo.24441874

The citation metadata in `CITATION.cff` is kept in sync with each release, so the
"Cite this repository" button on GitHub produces both APA and BibTeX entries.
