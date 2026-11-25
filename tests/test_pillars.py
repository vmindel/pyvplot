"""Tests for multi-length cut-pillar and de-novo motif footprint scanning and pygenomic_tracks DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyvplot as vp
import pyvplot.pillars as pillars_mod


def _make_synthetic_chec_store() -> vp.FragmentStore:
    """Create a synthetic paired-end ChEC-seq locus with two protected 17-bp motifs
    at chr1:1199-1216 and chr1:1259-1276 flanked by multi-length cleavage pillars
    at cut boundaries 1199, 1216, 1259, and 1276.
    """
    rng = np.random.default_rng(42)
    starts = []
    ends = []

    # Motif 1 at [1199, 1216) -> left pillar boundary at 1199, right pillar boundary at 1216
    # Motif 2 at [1259, 1276) -> left pillar boundary at 1259, right pillar boundary at 1276
    pillar_positions = [1199, 1216, 1259, 1276]

    # 1. Spanning short TF fragments across Motif 1 ([1199, 1216)) and Motif 2 ([1259, 1276))
    for _ in range(80):
        starts.append(1199 + int(rng.integers(-1, 2)))
        ends.append(1259 + int(rng.integers(-1, 2)))
    for _ in range(80):
        starts.append(1216 + int(rng.integers(-1, 2)))
        ends.append(1276 + int(rng.integers(-1, 2)))
    for _ in range(60):
        starts.append(1199 + int(rng.integers(-1, 2)))
        ends.append(1276 + int(rng.integers(-1, 2)))

    # 2. Multi-length cuts anchored at each pillar boundary across lengths 25..220 bp
    for p in pillar_positions:
        for fl in range(25, 225, 10):
            for _ in range(4):
                if p in (1199, 1259):
                    starts.append(p - fl + int(rng.integers(-1, 2)))
                    ends.append(p + int(rng.integers(-1, 2)))
                else:
                    starts.append(p + int(rng.integers(-1, 2)))
                    ends.append(p + fl + int(rng.integers(-1, 2)))

    return vp.FragmentStore.from_arrays(
        {"chr1": np.array(starts, dtype=np.int32)},
        {"chr1": np.array(ends, dtype=np.int32)},
        assay="chec",
    )


def test_scan_locus_pillars_and_track_dfs():
    store = _make_synthetic_chec_store()
    scan = store.scan_locus_pillars("chr1:1100-1400")

    assert scan.chrom == "chr1"
    assert scan.start == 1100
    assert scan.end == 1400
    assert len(scan.pos) == 300
    assert scan.protection_2d.shape == (len(scan.footprint_widths), 300)
    assert "_store" not in scan.metadata

    # Check called pillars match the 4 synthetic pillar boundaries
    assert len(scan.pillars) == 4
    assert {"chrom", "start", "end", "pos", "score", "strand", "pillar_score", "n_len_bins"}.issubset(
        scan.pillars.columns
    )
    np.testing.assert_array_equal(scan.pillars["pos"].to_numpy(), np.array([1199, 1216, 1259, 1276]))

    # Check called motifs match exact [l, r) intervals [1199, 1216) and [1259, 1276) with no shadow calls
    assert len(scan.motifs) == 2
    assert {"chrom", "start", "end", "center", "width", "name", "score", "strand", "protection_score"}.issubset(
        scan.motifs.columns
    )
    np.testing.assert_array_equal(scan.motifs["start"].to_numpy(), np.array([1199, 1259]))
    np.testing.assert_array_equal(scan.motifs["end"].to_numpy(), np.array([1216, 1276]))
    np.testing.assert_array_equal(scan.motifs["width"].to_numpy(), np.array([17, 17]))

    # Check pygenomic_tracks DataFrame outputs
    tracks = scan.to_pygenomic_tracks(bin_size=1)
    for key in ("pillar_score", "protection_score", "raw_log2_protection", "tf_coverage", "cuts", "pillars", "motifs"):
        df = tracks[key]
        assert isinstance(df, pd.DataFrame)
        assert {"chrom", "start", "end"}.issubset(df.columns)
        assert (df["end"] > df["start"]).all()

    binned = scan.to_track_df("protection_score", bin_size=10, drop_zeros=True)
    assert len(binned) > 0
    assert list(binned.columns) == ["chrom", "start", "end", "value", "score"]
    assert (binned["value"] != 0.0).all()


def test_shadow_regression_single_footprint_with_background():
    """A single planted footprint [2000, 2016) surrounded by background cuts on either side
    must produce exactly one called motif [2000, 2016) and zero shadow calls.
    """
    rng = np.random.default_rng(17)
    starts = []
    ends = []

    # Background fragments across [1500, 2500)
    for _ in range(300):
        s = int(rng.integers(1500, 2500))
        fl = int(rng.integers(25, 220))
        starts.append(s)
        ends.append(s + fl)

    # Planted footprint [2000, 2016)
    l_b, r_b = 2000, 2016
    for fl in range(25, 225, 10):
        for _ in range(3):
            starts.append(l_b - fl + int(rng.integers(-1, 2)))
            ends.append(l_b + int(rng.integers(-1, 2)))
            s0 = r_b + int(rng.integers(-1, 2))
            starts.append(s0)
            ends.append(s0 + fl)
    for _ in range(40):
        starts.append(l_b + int(rng.integers(-1, 1)))
        ends.append(r_b + int(rng.integers(20, 70)))

    store = vp.FragmentStore.from_arrays(
        {"chr1": np.array(starts, dtype=np.int32)},
        {"chr1": np.array(ends, dtype=np.int32)},
        assay="chec",
    )
    scan = store.scan_locus_pillars("chr1:1600-2400")
    assert len(scan.motifs) == 1
    assert int(scan.motifs["start"].iloc[0]) == 2000
    assert int(scan.motifs["end"].iloc[0]) == 2016


def test_numba_python_backend_equivalence():
    """Numba and pure-Python backends must return identical intervals and profiles."""
    store = _make_synthetic_chec_store()

    orig_numba = pillars_mod.HAS_NUMBA
    try:
        pillars_mod.HAS_NUMBA = True
        scan_nb = store.scan_locus_pillars("chr1:1100-1400")
        motifs_nb, pil_nb = store.scan_genome_pillars(return_pillars=True, min_bin_frags=5)

        pillars_mod.HAS_NUMBA = False
        scan_py = store.scan_locus_pillars("chr1:1100-1400")
        motifs_py, pil_py = store.scan_genome_pillars(return_pillars=True, min_bin_frags=5)
    finally:
        pillars_mod.HAS_NUMBA = orig_numba

    np.testing.assert_allclose(scan_nb.pillar_score, scan_py.pillar_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scan_nb.protection_score, scan_py.protection_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(
        scan_nb.motifs[["start", "end", "center"]].to_numpy(),
        scan_py.motifs[["start", "end", "center"]].to_numpy(),
    )
    np.testing.assert_array_equal(
        motifs_nb[["start", "end", "center"]].to_numpy(),
        motifs_py[["start", "end", "center"]].to_numpy(),
    )
    np.testing.assert_array_equal(
        pil_nb[["start", "end", "pos"]].to_numpy(),
        pil_py[["start", "end", "pos"]].to_numpy(),
    )


def test_truncation_regression_wide_active_window():
    """A single 60-kb active window holding 2,000 footprints every 30 bp over 10-70 kb
    must never truncate calls in scan_genome_pillars and must match between Numba and Python.
    """
    starts = []
    ends = []
    lengths = np.linspace(25, 220, 30, dtype=np.int32)
    for b in range(10_000, 70_000, 30):
        for fl in lengths:
            starts.append(b - int(fl))
            ends.append(b)
            starts.append(b + 16)
            ends.append(b + 16 + int(fl))
        for _ in range(20):
            starts.append(b)
            ends.append(b + 46)

    store = vp.FragmentStore.from_arrays(
        {"chr1": np.array(starts, dtype=np.int32)},
        {"chr1": np.array(ends, dtype=np.int32)},
        assay="chec",
    )

    orig_numba = pillars_mod.HAS_NUMBA
    try:
        pillars_mod.HAS_NUMBA = True
        motifs_nb, pil_nb = store.scan_genome_pillars(return_pillars=True)

        pillars_mod.HAS_NUMBA = False
        motifs_py, pil_py = store.scan_genome_pillars(return_pillars=True)
    finally:
        pillars_mod.HAS_NUMBA = orig_numba

    assert len(motifs_nb) >= 2000
    assert int(motifs_nb["end"].max()) >= 69_900
    # Every planted footprint comes back exactly: 1-bp cut stacks must not drift to a plateau edge.
    planted = {(b, b + 16) for b in range(10_000, 70_000, 30)}
    assert planted <= set(zip(motifs_nb["start"].tolist(), motifs_nb["end"].tolist(), strict=True))
    np.testing.assert_array_equal(
        motifs_nb[["start", "end", "center"]].to_numpy(),
        motifs_py[["start", "end", "center"]].to_numpy(),
    )
    np.testing.assert_array_equal(
        pil_nb[["start", "end", "pos"]].to_numpy(),
        pil_py[["start", "end", "pos"]].to_numpy(),
    )


def test_simulation_precision_recall_and_zero_boundary_offset():
    """On 120 planted footprints in 60,000 background fragments, pair-based calling
    achieves precision >= 0.95, recall >= 108/120, and median start/end offset == 0 bp.
    """
    rng = np.random.default_rng(5)
    S, E, planted = [], [], []
    for _ in range(60_000):
        s = int(rng.integers(0, 200_000))
        fl = int(rng.integers(20, 250))
        S.append(s)
        E.append(s + fl)
    for c in rng.choice(np.arange(5_000, 195_000, 400), 120, replace=False):
        w = int(rng.integers(10, 25))
        l_b, r_b = int(c - w // 2), int(c - w // 2 + w)
        planted.append((l_b, r_b))
        for _ in range(int(rng.integers(15, 60))):
            fl = int(rng.integers(25, 220))
            S.append(l_b - fl + int(rng.integers(-1, 2)))
            E.append(l_b + int(rng.integers(-1, 2)))
            fl = int(rng.integers(25, 220))
            s0 = r_b + int(rng.integers(-1, 2))
            S.append(s0)
            E.append(s0 + fl)
        for _ in range(int(rng.integers(10, 40))):
            S.append(l_b + int(rng.integers(-2, 1)))
            E.append(r_b + int(rng.integers(20, 80)))

    o = np.argsort(S)
    S_arr = np.array(S, dtype=np.int32)[o]
    E_arr = np.array(E, dtype=np.int32)[o]
    keep = S_arr >= 0
    store = vp.FragmentStore.from_arrays({"chr1": S_arr[keep]}, {"chr1": E_arr[keep]}, assay="chec")

    motifs_df = store.scan_genome_pillars()
    planted_arr = np.array(planted, dtype=np.int64)
    p_centers = 0.5 * (planted_arr[:, 0] + planted_arr[:, 1])

    cs = motifs_df["start"].to_numpy()
    ce = motifs_df["end"].to_numpy()
    cc = 0.5 * (cs + ce)

    tp = 0
    start_offs = []
    end_offs = []
    for s_i, e_i, c_i in zip(cs, ce, cc, strict=True):
        dists = np.abs(p_centers - c_i)
        idx = int(np.argmin(dists))
        if dists[idx] <= 3.0:
            tp += 1
            start_offs.append(int(s_i - planted_arr[idx, 0]))
            end_offs.append(int(e_i - planted_arr[idx, 1]))

    precision = tp / len(motifs_df)
    recall_count = sum(np.min(np.abs(cc - pc_i)) <= 3.0 for pc_i in p_centers)

    assert precision >= 0.95
    assert recall_count >= 108
    assert np.median(start_offs) == 0.0
    assert np.median(end_offs) == 0.0


@pytest.mark.parametrize("jitter", [0, 1])
def test_pillars_land_on_sharp_and_jittered_cut_stacks(jitter):
    """pillar_score is box-filtered, so a 1-bp cut stack is a flat plateau whose peak position
    background noise decides. Called pillars and motif edges must sit on the stack itself,
    whether it is 1 bp wide or spread over +-1 bp, for every background realisation.
    """
    l_b, r_b = 5000, 5016
    for seed in range(12):
        rng = np.random.default_rng(seed)
        starts, ends = [], []
        for _ in range(3000):
            s = int(rng.integers(4000, 6000))
            starts.append(s)
            ends.append(s + int(rng.integers(20, 250)))
        for _ in range(80):
            fl = int(rng.integers(25, 220))
            starts.append(l_b - fl)
            ends.append(l_b + int(rng.integers(-jitter, jitter + 1)))
            s0 = r_b + int(rng.integers(-jitter, jitter + 1))
            starts.append(s0)
            ends.append(s0 + int(rng.integers(25, 220)))
        for _ in range(40):
            starts.append(l_b)
            ends.append(r_b + int(rng.integers(20, 80)))
        order = np.argsort(starts)
        store = vp.FragmentStore.from_arrays(
            {"chr1": np.array(starts)[order].astype(np.int32)},
            {"chr1": np.array(ends)[order].astype(np.int32)},
            assay="chec",
        )
        scan = store.scan_locus_pillars("chr1:4800-5200")
        assert {l_b, r_b} <= set(scan.pillars["pos"].tolist()), seed
        hits = scan.motifs[(scan.motifs["center"] - 5008).abs() < 8]
        assert hits[["start", "end"]].values.tolist() == [[l_b, r_b]], seed
