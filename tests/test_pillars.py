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
    scan = store.scan_locus_pillars("chr1:1100-1400", caller="pairs")

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
    scan = store.scan_locus_pillars("chr1:1600-2400", caller="pairs")
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
        motifs_nb, pil_nb = store.scan_genome_pillars(return_pillars=True, caller="pairs")

        pillars_mod.HAS_NUMBA = False
        motifs_py, pil_py = store.scan_genome_pillars(return_pillars=True, caller="pairs")
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

    motifs_df = store.scan_genome_pillars(caller="pairs")
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
        scan = store.scan_locus_pillars("chr1:4800-5200", caller="pairs")
        assert {l_b, r_b} <= set(scan.pillars["pos"].tolist()), seed
        hits = scan.motifs[(scan.motifs["center"] - 5008).abs() < 8]
        assert hits[["start", "end"]].values.tolist() == [[l_b, r_b]], seed


def _single_footprint_store(l_b: int, r_b: int, seed: int = 3) -> vp.FragmentStore:
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for _ in range(3000):
        s = int(rng.integers(l_b - 1000, l_b + 1000))
        starts.append(s)
        ends.append(s + int(rng.integers(20, 250)))
    for _ in range(80):
        fl = int(rng.integers(25, 220))
        starts.append(l_b - fl)
        ends.append(l_b)
        starts.append(r_b)
        ends.append(r_b + int(rng.integers(25, 220)))
    for _ in range(40):
        starts.append(l_b)
        ends.append(r_b + int(rng.integers(20, 80)))
    order = np.argsort(starts)
    return vp.FragmentStore.from_arrays(
        {"chr1": np.array(starts)[order].astype(np.int32)},
        {"chr1": np.array(ends)[order].astype(np.int32)},
        assay="chec",
    )


def _called(scan_or_df, l_b, r_b):
    m = scan_or_df.motifs if hasattr(scan_or_df, "motifs") else scan_or_df
    return [l_b, r_b] in m[["start", "end"]].values.tolist()


def test_motif_width_range_is_exact_and_inclusive():
    store = _single_footprint_store(5000, 5025)  # r - l = 25
    assert not _called(store.scan_locus_pillars("chr1:4800-5200", motif_width_range=(10, 24), caller="pairs"), 5000, 5025)
    assert _called(store.scan_locus_pillars("chr1:4800-5200", motif_width_range=(10, 25), caller="pairs"), 5000, 5025)
    store = _single_footprint_store(5000, 5010)  # lower bound, r - l = 10
    assert _called(store.scan_locus_pillars("chr1:4800-5200", motif_width_range=(10, 24), caller="pairs"), 5000, 5010)
    assert not _called(store.scan_locus_pillars("chr1:4800-5200", motif_width_range=(11, 24), caller="pairs"), 5000, 5010)


@pytest.mark.parametrize("caller", ["pairs", "profile"])
def test_expected_cuts_scale_free_validated_and_kept_by_call(caller):
    store = _single_footprint_store(5000, 5016)
    region, W = "chr1:4800-5200", 400
    rng = np.random.default_rng(0)
    exp = rng.uniform(0.5, 2.0, W)
    a = store.scan_locus_pillars(region, expected_cuts=exp, caller=caller)
    b = store.scan_locus_pillars(region, expected_cuts=exp * 1e-3, caller=caller)
    pd.testing.assert_frame_equal(a.motifs, b.motifs)
    # .call() reuses the scan's array unless given another
    pd.testing.assert_frame_equal(a.call().motifs, a.motifs)
    plain = store.scan_locus_pillars(region, caller=caller)
    pd.testing.assert_frame_equal(plain.call(expected_cuts=exp).motifs, a.motifs)
    # a core expected to be rarely cut explains the depletion away
    gc_core = np.ones(W)
    gc_core[5003 - 4800 : 5013 - 4800] = 0.15
    def at_footprint(m):
        if caller == "pairs":
            return m[(m.start == 5000) & (m.end == 5016)]
        return m[(m.center - 5008).abs() <= 3]

    hit = at_footprint(plain.motifs)
    hit_bias = at_footprint(store.scan_locus_pillars(region, expected_cuts=gc_core, caller=caller).motifs)
    assert len(hit) == 1
    assert len(hit_bias) == 0
    with pytest.raises(ValueError, match="one value per bp"):
        store.scan_locus_pillars(region, expected_cuts=np.ones(W - 1))
    with pytest.raises(ValueError, match="one value per bp"):
        plain.call(expected_cuts=np.ones(W + 1))
    with pytest.raises(ValueError, match="non-negative"):
        store.scan_locus_pillars(region, expected_cuts=-np.ones(W))


def test_genome_expected_cuts_callable_matches_locus_scan_per_window():
    store = _single_footprint_store(5000, 5016)
    rng = np.random.default_rng(1)
    genome_exp = rng.uniform(0.5, 2.0, 20_000)
    seen = []

    def f(chrom, start, end):
        seen.append((chrom, start, end))
        return genome_exp[start:end]

    motifs = store.scan_genome_pillars(expected_cuts=f)
    wins = pillars_mod.find_active_windows(store.starts["chr1"], store.ends["chr1"])
    assert seen == [("chr1", int(w0), int(w1)) for w0, w1 in wins]
    per_window = pd.concat(
        [store.scan_locus_pillars(("chr1", w0, w1), expected_cuts=genome_exp[w0:w1]).motifs for w0, w1 in wins],
        ignore_index=True,
    )
    key = ["start", "end", "protection_score"]
    pd.testing.assert_frame_equal(
        motifs[key].sort_values(key).reset_index(drop=True), per_window[key].sort_values(key).reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="one value per bp"):
        store.scan_genome_pillars(expected_cuts=lambda c, s, e: np.ones(e - s + 1))


@pytest.mark.parametrize("k,off", [(4, 2), (5, 2), (6, 3), (5, 0), (5, 4)])
def test_pillar_expected_cuts_matches_hand_computed_kmer_weights(k, off, tmp_path):
    """Boundary b: forward k-mer with base b at `off` (fragment starting at b) plus the reverse
    complement k-mer with base b - 1 at `off` in the RC frame (fragment ending at b), the two
    frames KmerBiasTable.from_cuts estimates. Out-of-chromosome bases score 1.0."""
    from pyvplot.bias import reverse_complement

    rng = np.random.default_rng(k * 10 + off)
    g = "".join(rng.choice(list("ACGT"), 400))
    table = vp.KmerBiasTable(k=k, weights=rng.uniform(0.2, 5.0, 4**k), cut_offset=off)

    def w(kmer: str) -> float:
        return table[kmer] if len(kmer) == k and "N" not in kmer else 1.0

    padded = "N" * 20 + g + "N" * 20

    def hand(b: int) -> float:
        q = b + 20
        return w(padded[q - off : q - off + k]) + w(reverse_complement(padded[q - k + off : q + off]))

    f = vp.pillar_expected_cuts(table, {"chr1": g})
    for start, end in [(100, 160), (0, 30), (370, 400)]:
        got = f("chr1", start, end)
        assert got.shape == (end - start,)
        interior = [b for b in range(start, end) if b - k >= 0 and b + k <= len(g)]
        np.testing.assert_allclose(got[np.array(interior) - start], [hand(b) for b in interior])
    fasta = tmp_path / "g.fa"
    fasta.write_text(">chr1\n" + "\n".join(g[i : i + 60] for i in range(0, len(g), 60)) + "\n")
    np.testing.assert_allclose(vp.pillar_expected_cuts(table, fasta)("chr1", 100, 160), f("chr1", 100, 160))
    np.testing.assert_array_equal(f("chrX", 0, 10), np.full(10, 2.0))


def test_pillar_expected_cuts_feeds_both_scans():
    store = _single_footprint_store(5000, 5016)
    rng = np.random.default_rng(2)
    g = "".join(rng.choice(list("ACGT"), 20_000))
    table = vp.KmerBiasTable(k=4, weights=rng.uniform(0.5, 2.0, 256))
    f = vp.pillar_expected_cuts(table, {"chr1": g})
    via_fn = store.scan_locus_pillars("chr1:4800-5200", expected_cuts=f)
    via_arr = store.scan_locus_pillars("chr1:4800-5200", expected_cuts=f("chr1", 4800, 5200))
    pd.testing.assert_frame_equal(via_fn.motifs, via_arr.motifs)
    np.testing.assert_array_equal(via_fn.expected_cuts, f("chr1", 4800, 5200))
    assert len(store.scan_genome_pillars(expected_cuts=f)) > 0


def test_plot_draws_calls_at_pair_score_and_threshold_on_them():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = _single_footprint_store(5000, 5016)
    scan = store.scan_locus_pillars("chr1:4800-5200")
    assert len(scan.motifs) >= 1

    expected = sorted(
        (int(s), int(e), round(float(p), 6))
        for s, e, p in scan.motifs[["start", "end", "protection_score"]].itertuples(index=False)
    )
    for kwargs, n_axes, prot_idx in (({"store": store}, 4, 1), ({}, 4, 1), ({"mode": "multiscale"}, 4, 2)):
        fig, axes = scan.plot(**kwargs)
        assert len(axes) == n_axes
        for a in axes:
            assert tuple(round(v) for v in a.get_xlim()) == (4800, 5200)
        if kwargs.get("mode") != "multiscale":  # multiscale adds colourbar axes
            assert len(fig.axes) == len(axes)  # no twin y-axes
        ax_p = axes[prot_idx]
        bars = sorted(
            (round(b.get_x()), round(b.get_x() + b.get_width()), round(b.get_height(), 6))
            for c in ax_p.containers
            for b in c
        )
        assert bars == expected
        thresholds = [ln.get_ydata()[0] for ln in ax_p.get_lines() if ln.get_linestyle() == "--"]
        assert thresholds == [scan.metadata["min_protection_score"]]
        plt.close(fig)
    fig, ax = plt.subplots()
    scan.plot(ax=ax)
    assert len(ax.containers) == len(scan.motifs)
    plt.close(fig)


def test_profile_caller_calls_sit_on_profile_peaks():
    store = _make_synthetic_chec_store()
    scan = store.scan_locus_pillars("chr1:1100-1400")
    assert scan.metadata["caller"] == "profile"
    m = scan.motifs
    assert {"chrom", "start", "end", "center", "width", "protection_score"}.issubset(m.columns)
    assert len(m) >= 2
    for r in m.itertuples():
        i = int(r.center) - scan.start
        p = scan.protection_score
        assert p[i] == pytest.approx(r.protection_score)
        assert p[i] >= scan.metadata["min_protection_score"]
        assert p[i] > p[i - 1] and p[i] >= p[i + 1]
        w, fp = int(scan.best_widths[i]), scan.metadata["flank_pad"]
        assert abs(int(r.start) - (int(r.center) - w // 2)) <= fp
        assert abs(int(r.end) - (int(r.center) + w - w // 2)) <= fp
        assert int(r.width) == int(r.end) - int(r.start)
        assert r.start < r.center < r.end
    assert (np.diff(m["center"].to_numpy()) >= scan.metadata["min_motif_distance"]).all()
    # both planted 17-bp footprints are found at their centres
    for c in (1207.5, 1267.5):
        assert (m["center"] - c).abs().min() <= 2


def test_call_switches_caller_without_rescanning():
    store = _make_synthetic_chec_store()
    prof = store.scan_locus_pillars("chr1:1100-1400")
    pairs = store.scan_locus_pillars("chr1:1100-1400", caller="pairs")
    assert pairs.metadata["min_protection_score"] == 0.75
    assert prof.metadata["min_protection_score"] == 1.0
    pd.testing.assert_frame_equal(prof.call(caller="pairs").motifs, pairs.motifs)
    pd.testing.assert_frame_equal(pairs.call(caller="profile").motifs, prof.motifs)
    pd.testing.assert_frame_equal(prof.call().motifs, prof.motifs)
    with pytest.raises(ValueError, match="caller"):
        store.scan_locus_pillars("chr1:1100-1400", caller="peaks")
    with pytest.raises(ValueError, match="caller"):
        store.scan_genome_pillars(caller="peaks")


def _occupancy_brute(starts, ends, L, R, tf=(20, 120), inner=5, outer=60, bg=(40, 150)):
    """Direct per-boundary count of bracketing fragments, and the mean count at c +- bg."""
    s, e = np.asarray(starts, np.int64), np.asarray(ends, np.int64)
    fl = e - s
    k = (fl >= tf[0]) & (fl <= tf[1])
    s, e = s[k], e[k]
    lo = L - bg[1]
    cnt = np.array(
        [np.sum((s >= c - outer) & (s <= c - inner) & (e >= c + inner) & (e <= c + outer)) for c in range(lo, R + bg[1])],
        dtype=float,
    )
    offs = np.r_[np.arange(-bg[1], -bg[0] + 1), np.arange(bg[0], bg[1] + 1)]
    obs = np.array([cnt[c - lo] for c in range(L, R)])
    exp = np.array([cnt[c - lo + offs].mean() for c in range(L, R)])
    return obs, exp


def _make_occupancy_store():
    rng = np.random.default_rng(7)
    s = rng.integers(0, 3000, size=4000)
    e = s + rng.integers(15, 260, size=4000)
    # a bound site at boundary 1500: short fragments with an end 5-40 bp to either side
    s2 = 1500 - rng.integers(5, 40, size=300)
    e2 = 1500 + rng.integers(5, 40, size=300)
    starts, ends = np.r_[s, s2], np.r_[e, e2]
    o = np.argsort(starts, kind="stable")
    return vp.FragmentStore.from_arrays(
        {"chr1": starts[o].astype(np.int32)}, {"chr1": ends[o].astype(np.int32)}, assay="chec"
    )


def test_occupancy_profile_matches_direct_count_and_peaks_at_site():
    store = _make_occupancy_store()
    L, R = 1300, 1700
    obs, exp = _occupancy_brute(store.starts["chr1"], store.ends["chr1"], L, R)
    counts = vp.occupancy_profile(store, f"chr1:{L}-{R}", score="counts")
    np.testing.assert_array_equal(counts, obs)
    llr = vp.occupancy_profile(store, ("chr1", L, R))
    assert llr.shape == (R - L,)
    np.testing.assert_allclose(llr, pillars_mod._signed_poisson_llr(obs, exp), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        vp.occupancy_profile(store, ("chr1", L, R), score="log2"), np.log2((obs + 1) / (exp + 1)), rtol=1e-9
    )
    assert abs(int(np.argmax(llr)) + L - 1500) <= 3
    assert llr.max() > 50
    assert not vp.occupancy_profile(store, ("chr2", 0, 100)).any()
    with pytest.raises(ValueError, match="inner"):
        vp.occupancy_profile(store, ("chr1", L, R), inner=60, outer=60)
    with pytest.raises(ValueError, match="score"):
        vp.occupancy_profile(store, ("chr1", L, R), score="z")


def test_scan_carries_occupancy_track_and_motif_column():
    store = _make_synthetic_chec_store()
    for caller in ("profile", "pairs"):
        scan = store.scan_locus_pillars("chr1:1100-1400", caller=caller)
        np.testing.assert_array_equal(scan.occupancy_score, vp.occupancy_profile(store, "chr1:1100-1400"))
        m = scan.motifs
        assert len(m) > 0
        idx = np.floor(m["center"].to_numpy(dtype=float)).astype(np.int64) - scan.start
        np.testing.assert_allclose(m["occupancy_score"].to_numpy(dtype=float), scan.occupancy_score[idx])
        recalled = scan.call(caller="pairs" if caller == "profile" else "profile")
        assert "occupancy_score" in recalled.motifs.columns
        tr = scan.to_track_df("occupancy_score")
        np.testing.assert_array_equal(tr["value"].to_numpy(), scan.occupancy_score)
        assert "occupancy_score" in scan.to_pygenomic_tracks()
        assert scan.metadata["occupancy"]["inner"] == 5
    genome = store.scan_genome_pillars(min_bin_frags=1)
    assert "occupancy_score" in genome.columns
    assert genome["occupancy_score"].notna().all()


def test_plot_occupancy_panel():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = _make_synthetic_chec_store()
    scan = store.scan_locus_pillars("chr1:1100-1400")
    fig, ax = plt.subplots()
    scan.plot(ax=ax, what="occupancy")
    assert ax.get_ylabel().startswith("Occupancy")
    plt.close(fig)
