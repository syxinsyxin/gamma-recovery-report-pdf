# Closed-Loop Validation of VMAT Report Verification

Code for the closed-loop validation described in the manuscript: the dose maps
and gamma pass rates originally reported by archived verification PDFs are
recovered and re-analysed, and the **raw detector measurements** (32x32
ion-chamber dose matrices exported from myQA, CSV; `.mcc` also supported) are
used as an independent reference on the measured side. The full chain

    report PDF -> recovered dose maps -> gamma re-analysis

is then checked end-to-end against that reference.

Companion analyses in this package:

- a 3 x 2 (reference-side x evaluation-side) gamma decomposition
  (`factor3x2.py`), separating the reference-side smoothing effect from the
  evaluation-side dose-source effect;
- a bounding experiment for the common-mode amplitude bias of the recovery
  chain (`factor3x2_c4.py`).

## Repository layout

```
closed_loop_validation/
├── README.md                 this file
├── requirements.txt          dependencies
├── cases_template.csv        case-list template (manual mode)
├── src/
│   ├── gamma.py              2D gamma engine (built-in fallback; published
│   │                         numbers use the pymedphys engine)
│   ├── registration.py       RT Dose coronal extraction, detector-grid
│   │                         resampling, NCC registration, orientation
│   │                         auto-detection (8 dihedral transforms),
│   │                         absolute scale fit
│   ├── mcc_parser.py         lenient IBA .mcc parser (skeleton + inspector)
│   ├── export_for_closedloop.py  per-case artifact export from the v7 batch
│   │                              pipeline (needs that script; see below)
│   ├── validate.py           main driver: per-case metrics, raw-side gammas,
│   │                         summary, Bland-Altman plots
│   ├── factor3x2.py          3 x 2 reference x evaluation gamma decomposition
│   ├── factor3x2_c4.py       amplitude-bias correction control (C4/C5)
│   ├── results_3x2/          example output (authors' data, anonymised ids)
│   └── results_c4/           example output (authors' data, anonymised ids)
└── tests/
    ├── test_gamma.py         unit tests, synthetic data only
    └── test_registration.py  unit tests, synthetic data only
```

## Environment

Python >= 3.10 (tested on 3.12).

```bash
pip install -r requirements.txt
```

`pymedphys` is optional but recommended: the numbers reported in the paper use
the pymedphys gamma engine. Without it the built-in engine in `gamma.py` is
used as a fallback (equivalent configuration, slightly different pass rates).
`pydicom` is only needed for the RT Dose path; `pymupdf` only for
`export_for_closedloop.py`.

## Quick start (no real data required)

```bash
cd tests
python test_gamma.py          # expect: "all gamma tests passed"
python test_registration.py   # expect: "all registration tests passed"

cd ../src
python validate.py --selftest             # synthetic end-to-end run of the driver
python factor3x2.py --selftest --out st_dir/   # synthetic run of the decomposition
```

The selftests generate synthetic cases (including a deliberately rotated CSV to
exercise orientation detection) and verify the full pipeline without pydicom or
any real measurement.

## Running on your own data

### Option A - manual mode (no batch pipeline)

Build a case list from `cases_template.csv`:

| column | meaning |
|---|---|
| `case_id` | arbitrary identifier |
| `raw_path` | 32x32 raw measurement (`.csv` / `.txt` / `.mcc` / `.npy`) |
| `rec_meas_npy` | recovered measured map, full display grid (`.npy`, NaN = invalid) |
| `rec_plan_npy` | recovered plan map, full display grid (`.npy`) |
| `dcm_path` | RT Dose file for the case (optional) |
| `declared_gpr` | gamma pass rate declared in the report [%] |
| `declared_mean_gamma` | mean gamma declared in the report (optional) |
| `criteria`, `norm` | e.g. `3%/3mm`, `global` |

```bash
python src/validate.py --cases cases.csv --out results/
```

Outputs in `results/`: `per_case.csv`, `summary.json`, and Bland-Altman plots.

### Option B - full pipeline mode (as in the paper)

`export_for_closedloop.py` replays the v7 batch pipeline
(`batch_gamma_recovery_final_v7.py`, provided separately by the authors) per
case and writes one npz artifact per case plus a case list. It requires the v7
script, the v7 results CSV, the PDF/DCM data root, and the raw measurement CSVs
(via `--raw-map case_id,raw_path` or `--raw-dir`):

```bash
python src/export_for_closedloop.py \
    --v7 batch_gamma_recovery_final_v7.py \
    --v7-results results25_26_new.csv \
    --root <data root> \
    --ids ids30.txt \
    --raw-map raw_map.csv \
    --out cl_export

python src/validate.py --cases cl_export/cases_closedloop.csv --out results_cl/
python src/factor3x2.py --cases cl_export/cases_closedloop.csv --out results_3x2/
python src/factor3x2_c4.py --out results_c4/
```

Reviewers reproducing from the exported npz artifacts can start at step 2.

## Input formats

**Raw measurement CSV, format A** (myQA coordinate-annotated export):
first line `Name: Integral <timestamp>, ..., Dose unit: mGy`, second line the x
coordinates, data rows lead with the y coordinate. Handled automatically,
including unit conversion to cGy and extraction of the detector axes (the true
pitch is rebuilt from the coordinate range). The header name is cross-checked
against the report's `A:` line (`provenance_ok`).

**Raw CSV, format B**: a plain 32x32 numeric matrix. A header row containing
numbers is tolerated; non-matrix rows are dropped by row-width mode.

**`.mcc`**: the parser is deliberately permissive because layouts vary between
myQA versions. Inspect a real file first:

```bash
python src/mcc_parser.py <file.mcc> --inspect
```

It prints the keyword lines and every numeric block with its shape and value
range, so you can tell which block holds the dose matrix.

**`.npy`**: any (N, N) array; shapes divisible by the detector size are
block-averaged.

## Gamma conventions (important)

- The reference distribution always sits on the **coarse** detector grid; the
  evaluation distribution is interpolated onto a finer grid for the search.
  Putting the coarse map on the evaluation side instead systematically lowers
  the pass rate - do not swap them.
- Settings mirror the report pipeline exactly: local_gamma=False,
  interp_fraction=10, max_gamma=2.0, global_normalisation = report Dmax,
  cutoff = declared % of Dmax.
- Orientation of the raw matrix vs the recovered maps is auto-detected per case
  among 8 dihedral transforms (`orientation_used`); results should be
  consistent across cases - outliers need manual checking.
- Units self-check: `unit_ratio_raw_vs_rec` (ground truth max / recovered max)
  should be ~1. `consistent` / `scale_offset_check` / `SCALE_MISMATCH` flags
  mark whether absolute-dose gamma comparisons are meaningful for that case.

## Key output columns

`per_case.csv` (validate.py):
- `gamma_rawA/B/Adet/Bdet`: gamma recomputed with the **raw chamber matrix** as
  reference; evaluation sides are the recovered plan (display grid), the DCM
  plane, and their detector-grid versions - each mirrors one report-side gamma.
- `dev_rawX` = gamma_rawX - declared GPR; `delta_devX` = dev_rawX - v7 devX,
  i.e. the net effect of replacing the recovered reference by the raw matrix.
- `dose_bias_median`, `dose_sd`, `sd_ratio_high_low`, `scale_a/b/r`:
  dose-domain fidelity of the recovered measured map.
- `align_ncc`, `align_y_mm`: DCM registration quality.

`factor3x2.csv`: `refEff_*` (reference-side effects, R1-R3),
`srcEff_*` (evaluation-side effects, plan vs DCM), `interaction`
(a large interaction means the "source effect" is largely driven by the
detail-content ratio between the two sides, not by the source itself).

`c4_calibration.csv`: `delta_dcm`, `delta_plan` - response of both pairings to
undoing the amplitude bias on the recovered reference; both moving together
supports the bias mechanism.

## Example outputs

`src/results_3x2/` and `src/results_c4/` contain example results from the
authors' runs (20 cases, ids anonymised) so the output schema and typical
magnitudes can be inspected without data access.

## Known limitations

- The `.mcc` parser is a skeleton; calibrate it with `--inspect` on a real
  export before trusting `parse_mcc` output.
- RT Dose is assumed to be in standard axial orientation
  (ImageOrientationPatient ~ (1,0,0,0,1,0)); reslice other orientations first.
- `export_for_closedloop.py` is tied to the authors' v7 batch script (not part
  of this package); everything else runs standalone.
- The closed loop demonstrates the fidelity of the PDF recovery chain, not the
  accuracy of the myQA measurement itself - keep manuscript wording consistent
  with that.

## Data privacy

Before distributing the package further, strip patient identifiers from case
lists and data paths. `cases_closedloop.csv` (generated by the export script)
is intentionally not included here.
