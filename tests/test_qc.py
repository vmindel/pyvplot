"""Tests for QC metrics: insert-size distribution, FRiP, phasogram, and NRL estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvplot.qc import frip, insert_size_distribution, phasogram, qc_summary
from pyvplot.store import FragmentStore


def test_insert_size_distribution_and_qc_summary():
    # ATAC-like bimodal distribution: NFR (~60 bp) and mononucleosome (~200 bp)
    rng = np.random.default_rng(7)
    nfr_lens = rng.integers(40, 80, size=300, dtype=np.int32)
    nuc_lens = rng.integers(190, 220, size=200, dtype=np.int32)
    lens = np.concatenate([nfr_lens, nuc_lens])
    starts = np.arange(1000, 1000 + len(lens) * 50, 50, dtype=np.int32)
    ends = starts + lens

    store = FragmentStore({"chr1": starts}, {"chr1": ends}, assay="atac")
    dist = insert_size_distribution(store, min_len=30, max_len=250)
    assert list(dist.columns) == ["length", "count", "fraction"]
    assert int(dist["count"].sum()) == 500
    assert pytest.approx(float(dist["fraction"].sum()), rel=1e-9) == 1.0

    summary = qc_summary(store)
    assert summary["n_fragments"] == 500
    assert pytest.approx(summary["size_band_fractions"]["nfr"], rel=1e-6) == 0.6
    assert pytest.approx(summary["size_band_fractions"]["mononucleosome"], rel=1e-6) == 0.4


def test_frip_midpoint_overlap_and_merged_peaks():
    # 4 fragments on chr1:
    # f0: [100, 200) mid=150 (in peak 1 [120, 180) AND overlapping peak 2 [140, 220))
    # f1: [250, 350) mid=300 (overlaps peak 3 [320, 400), midpoint outside peak 3!)
    # f2: [500, 600) mid=550 (completely outside peaks)
    # f3: [800, 900) mid=850 (inside peak 4 [800, 900))
    store = FragmentStore(
        starts={"chr1": np.array([100, 250, 500, 800], dtype=np.int32)},
        ends={"chr1": np.array([200, 350, 600, 900], dtype=np.int32)},
        assay="atac",
    )
    peaks = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [120, 140, 320, 800],
            "end": [180, 220, 400, 900],
        }
    )

    # Midpoint mode: f0 and f3 are in peaks (2/4 = 0.5); overlapping peaks 1 & 2 must NOT double-count f0!
    frip_mid, per_peak_mid = frip(store, peaks, mode="midpoint", return_per_peak=True)
    assert pytest.approx(frip_mid, rel=1e-9) == 0.5
    np.testing.assert_array_equal(per_peak_mid, [1, 1, 0, 1])

    # Overlap mode: f0, f1, f3 overlap peaks (3/4 = 0.75)
    frip_ov, per_peak_ov = frip(store, peaks, mode="overlap", return_per_peak=True)
    assert pytest.approx(frip_ov, rel=1e-9) == 0.75
    np.testing.assert_array_equal(per_peak_ov, [1, 1, 1, 1])


def test_phasogram_recovers_true_nrl_and_numba_parity():
    """Simulate a phased nucleosome array with exact NRL = 185 bp and verify recovery."""
    rng = np.random.default_rng(2026)
    true_nrl = 185.0
    starts_list: list[int] = []
    ends_list: list[int] = []

    # 40 well-phased nucleosome arrays of 8 nucleosomes each
    for array_idx in range(40):
        base = 5_000 + array_idx * 5_000
        for nuc_k in range(8):
            dyad_center = base + int(round(nuc_k * true_nrl))
            for _ in range(12):
                dyad = int(round(dyad_center + rng.normal(0.0, 8.0)))
                fl = int(rng.integers(145, 155))
                s = dyad - fl // 2
                starts_list.append(s)
                ends_list.append(s + fl)

    store = FragmentStore(
        starts={"chr1": np.asarray(starts_list, dtype=np.int32)},
        ends={"chr1": np.asarray(ends_list, dtype=np.int32)},
        assay="mnase",
    )

    res_nb = phasogram(store, max_lag=900, smooth_sigma=6.0, use_numba=True)
    res_np = phasogram(store, max_lag=900, smooth_sigma=6.0, use_numba=False)

    # Exact parity between Numba and NumPy phasogram counters
    np.testing.assert_array_equal(res_nb.counts, res_np.counts)

    # Both peak-fit and FFT NRL estimates must accurately recover 185 bp
    assert res_nb.nrl is not None
    assert abs(res_nb.nrl - true_nrl) <= 3.0
    assert res_nb.nrl_peak_fit is not None
    assert abs(res_nb.nrl_peak_fit - true_nrl) <= 3.0
    assert res_nb.nrl_fft is not None
    assert abs(res_nb.nrl_fft - true_nrl) <= 4.0
    assert res_nb.r_squared is not None and res_nb.r_squared > 0.99
    assert len(res_nb.peak_positions) >= 3


@pytest.mark.parametrize("mode", ["midpoint", "overlap"])
def test_frip_per_peak_with_non_default_index(mode: str):
    store = FragmentStore(
        starts={"chr1": np.array([100, 500, 900, 5000], dtype=np.int32)},
        ends={"chr1": np.array([150, 550, 950, 5050], dtype=np.int32)},
        assay="generic",
    )
    peaks = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "start": [0, 450, 880, 4900],
            "end": [200, 600, 1000, 5100],
            "score": [1, 9, 9, 9],
        }
    )
    sub = peaks[peaks["score"] > 5]  # index 1, 2, 3
    score, per_peak = frip(store, sub, mode=mode, return_per_peak=True)
    assert per_peak.tolist() == [1, 1, 1]
    assert score == pytest.approx(3 / 4)
    # same answer as the unfiltered frame restricted to those rows
    _, full = frip(store, peaks, mode=mode, return_per_peak=True)
    assert full[1:].tolist() == per_peak.tolist()
