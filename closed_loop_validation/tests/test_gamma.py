# -*- coding: utf-8 -*-
"""Unit tests for gamma.py (pure synthetic data; runs standalone or under pytest)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gamma import gamma_analysis  # noqa: E402

PITCH = 7.619
N = 32
MAX_DOSE = 200.0  # cGy


def _ramp32():
    """Linear ramp along columns (0 -> MAX_DOSE), constant along rows."""
    x = np.arange(N) / (N - 1)
    dose = MAX_DOSE * x[None, :] * np.ones((N, N))
    return dose


def _ramp_fine(spacing_mm, shift_mm=0.0):
    n_fine = int(np.floor((N - 1) * PITCH / spacing_mm)) + 1
    x = np.arange(n_fine) * spacing_mm
    xmax = (N - 1) * PITCH
    dose = MAX_DOSE * (x - shift_mm) / xmax
    return np.tile(dose[None, :], (n_fine, 1))


def test_identical_fields():
    ref = _ramp32() + 12.0  # min dose 12 cGy > 5% x 212 = 10.6, so every point is evaluated
    r = gamma_analysis(ref, (PITCH, PITCH), ref, (PITCH, PITCH),
                       dd_percent=3, dta_mm=3, interp_factor=10)
    assert r["pass_rate"] == 100.0
    assert r["mean_gamma"] < 1e-6
    assert r["n_eval"] == N * N  # all points above the 5% cutoff
    print("test_identical_fields OK:", r["pass_rate"], r["mean_gamma"])


def test_global_offset_equals_dd():
    """Flat field + an offset of exactly 1x delta-D -> gamma = 1 everywhere (exact check).

    A ramp field cannot be used here: within the search window gamma trades part
    of the dose difference against distance, so gamma(min) < delta/deltaD
    (correct engine behaviour -- see the analytic check in test_pure_lateral_shift).
    """
    ref = np.full((N, N), 150.0)
    ev = ref + 0.03 * 150.0
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH, PITCH),
                       dd_percent=3, dta_mm=3, interp_factor=10)
    assert r["pass_rate"] == 100.0
    assert abs(r["mean_gamma"] - 1.0) < 1e-6
    print("test_global_offset_equals_dd OK:", r["mean_gamma"])


def test_pure_lateral_shift():
    """Pure 1.5 mm lateral shift of a linear ramp -> gamma matches the analytic value.

    Analytic: the evaluation map is sampled on a 0.7619 mm grid anchored at 0,
    so a sample falls exactly at zero distance from each reference point; the
    dose difference there is s*shift (s = ramp slope) and gamma = s*1.5/(0.03*Dmax)
    ~ 0.2016. (The continuous optimum 0.18 is unreachable under this sampling --
    the engine matches the theory.)
    """
    ref = _ramp32() + 10.0
    ev = _ramp_fine(PITCH / 10, shift_mm=1.5) + 10.0
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH / 10,) * 2,
                       dd_percent=3, dta_mm=3, interp_factor=1)
    assert r["pass_rate"] == 100.0
    gm = r["gamma_map"]
    center = gm[4:28, 6:26]
    mg = float(np.nanmean(center))
    s = MAX_DOSE / ((N - 1) * PITCH)
    expected = s * 1.5 / (0.03 * ref.max())
    assert abs(mg - expected) < 0.005, f"mean gamma in the centre region={mg}, expected ~{expected:.4f}"
    print("test_pure_lateral_shift OK:", mg, "expected", expected)


def test_failing_region():
    """Eval dose x0.7 inside a central disk -> partial failures, 0 < pass rate < 100."""
    ref = _ramp32() + 10.0
    ev = ref.copy()
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    disk = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 9
    ev[disk] = ref[disk] * 0.7
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH, PITCH),
                       dd_percent=3, dta_mm=3, interp_factor=10)
    assert 0.0 < r["pass_rate"] < 100.0
    print("test_failing_region OK:", r["pass_rate"])


def test_coarse_ref_fine_eval_consistency():
    """Same smooth field: coarse reference vs fine evaluation -> 100% pass, tiny mean gamma."""
    ev = _ramp_fine(PITCH / 10) + 10.0
    ref = _ramp32() + 10.0
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH / 10,) * 2,
                       dd_percent=3, dta_mm=3, interp_factor=1)
    assert r["pass_rate"] == 100.0
    assert r["mean_gamma"] < 0.05
    print("test_coarse_ref_fine_eval OK:", r["mean_gamma"])


def test_local_norm():
    """Exact check of local normalisation: flat field, eval x(1+3%) -> the per-point
    dose difference equals the local delta-D exactly, so gamma = 1.

    On a ramp, locally normalised gamma would be < 1 (gamma trades dose difference
    for distance via the gradient), hence the gradient-free flat field.
    """
    ref = np.full((N, N), 150.0)
    ev = ref * 1.03
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH, PITCH),
                       dd_percent=3, dta_mm=3, global_norm=False, interp_factor=10)
    assert r["pass_rate"] == 100.0
    assert abs(r["mean_gamma"] - 1.0) < 1e-6
    print("test_local_norm OK:", r["mean_gamma"])


def test_cutoff():
    """Points below the cutoff are excluded from the statistics."""
    ref = _ramp32()
    ev = ref * 2.0  # huge dose difference; any included point must fail
    r = gamma_analysis(ref, (PITCH, PITCH), ev, (PITCH, PITCH),
                       dd_percent=3, dta_mm=3, cutoff_percent=50, interp_factor=10)
    assert r["n_eval"] < N * N  # only the high-dose half participates
    assert r["pass_rate"] == 0.0
    print("test_cutoff OK: n_eval =", r["n_eval"])


if __name__ == "__main__":
    test_identical_fields()
    test_global_offset_equals_dd()
    test_pure_lateral_shift()
    test_failing_region()
    test_coarse_ref_fine_eval_consistency()
    test_local_norm()
    test_cutoff()
    print("\nall gamma tests passed")
