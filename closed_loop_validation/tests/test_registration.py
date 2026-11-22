# -*- coding: utf-8 -*-
"""Unit tests for registration.py (synthetic 3D dose, no pydicom; runs standalone or under pytest)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import registration as rg  # noqa: E402

PITCH = 7.619


def _synthetic_dcm(nz=84, ny=60, nx=84, spacing=3.0):
    """Synthetic RT Dose dict: ellipsoidal dose; the x/z extent covers the
    >=244 mm field of view of the detector grid."""
    z = (np.arange(nz) - nz // 2) * spacing
    y = (np.arange(ny) - ny // 2) * spacing
    x = (np.arange(nx) - nx // 2) * spacing
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    dose = 200.0 * np.exp(-((X ** 2) / (2 * 60.0 ** 2)
                            + (Y ** 2) / (2 * 15.0 ** 2)
                            + (Z ** 2) / (2 * 70.0 ** 2))) + 2.0
    return {
        "dose": dose,
        "row_spacing": spacing,
        "col_spacing": spacing,
        "position": [float(-nx // 2 * spacing), float(-ny // 2 * spacing), float(-nz // 2 * spacing)],
        # DICOM: GridFrameOffsetVector is relative to ImagePositionPatient[2]
        "offsets": (z - z[0]).copy(),
        "units": "GY",
        "for_uid": "synthetic",
    }, x, y, z


def test_coronal_slice_and_det_grid():
    dcm, x, y, z = _synthetic_dcm()
    plane, zz, xx = rg.sample_coronal_slice(dcm, y_mm=0.0)
    assert plane.shape == (84, 84)
    assert np.allclose(zz, z) and np.allclose(xx, x)
    det, center = rg.sample_to_detector_grid(plane, zz, xx, pitch=PITCH, n=32, ss=8)
    assert det.shape == (32, 32)
    assert np.isfinite(det).sum() > 32 * 32 * 0.9  # field of view essentially covered
    print("test_coronal_slice_and_det_grid OK; det max =", np.nanmax(det))


def test_align_recovers_slice_and_shift():
    dcm, x, y, z = _synthetic_dcm()
    # ground truth: the y=0 plane with the detector-grid centre shifted by (dz=+2mm, dx=-3mm)
    plane, zz, xx = rg.sample_coronal_slice(dcm, 0.0)
    target, _ = rg.sample_to_detector_grid(
        plane, zz, xx, pitch=PITCH, n=32, center=(0.0 + 2.0, 0.0 - 3.0), ss=8
    )
    y_cand = np.arange(-3.0, 3.0 + 1e-6, 1.0)
    res = rg.align_dcm_to_detector_map(
        dcm, target, pitch=PITCH, n=32, y_candidates=y_cand,
        margin_mm=8.0, ss=4, orientation="normal",
    )
    assert abs(res["best_y_mm"] - 0.0) <= 1.0 + 1e-6
    assert abs(res["best_shift_dz"] - 2.0) <= PITCH / 2 + 1e-6
    assert abs(res["best_shift_dx"] - (-3.0)) <= PITCH / 2 + 1e-6
    assert res["ncc_final"] > 0.999
    print("test_align_recovers_slice_and_shift OK:",
          res["best_y_mm"], res["best_shift_dz"], res["best_shift_dx"], res["ncc_final"])


def test_block_average():
    fine = np.arange(64 * 64, dtype=float).reshape(64, 64)
    coarse = rg.block_average(fine, 2)
    assert coarse.shape == (32, 32)
    assert abs(coarse[0, 0] - np.mean(fine[:2, :2])) < 1e-9
    print("test_block_average OK")


def test_fit_dose_scale():
    pred = np.random.default_rng(0).normal(100, 20, (32, 32))
    meas = 1.02 * pred - 0.5
    fit = rg.fit_dose_scale(meas, pred)
    assert abs(fit["a"] - 1.02) < 1e-6 and abs(fit["b"] + 0.5) < 1e-6
    assert fit["r"] > 0.999
    print("test_fit_dose_scale OK:", fit)


def test_orientation():
    a = np.arange(9, dtype=float).reshape(3, 3)
    assert np.allclose(rg._apply_orientation(a, "T"), a.T)
    assert np.allclose(rg._apply_orientation(a, "UD"), a[::-1, :])
    assert np.allclose(rg._apply_orientation(a, "LR"), a[:, ::-1])
    assert np.allclose(rg._apply_orientation(a, "T+UD"), a.T[::-1, :])
    print("test_orientation OK")


if __name__ == "__main__":
    test_coronal_slice_and_det_grid()
    test_align_recovers_slice_and_shift()
    test_block_average()
    test_fit_dose_scale()
    test_orientation()
    print("\nall registration tests passed")
