"""Exact correctness tests verifying fast Numba & NumPy kernels against naive brute-force reference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pyvplot.extract import HAS_NUMBA, accumulate_matrix, accumulate_vplot, extract_events
from pyvplot.store import FragmentStore


def _naive_vplot(
    store: FragmentStore,
    anchors: pd.DataFrame,
    upstream: int,
    downstream: int,
    min_len: int,
    max_len: int,
    pos_bin: int,
    len_bin: int,
    feature: str,
) -> np.ndarray:
    """Independent O(N_anchors * N_frags) brute-force Python reference for V-plot accumulation."""
    n_pos = (upstream + downstream) // pos_bin
    n_len = (max_len - min_len) // len_bin
    grid = np.zeros((n_len, n_pos), dtype=np.float64)

    for _, row in anchors.iterrows():
        chrom = str(row["chrom"])
        if chrom not in store.starts:
            continue
        c = float(row["center"])
        a_st = -1 if str(row.get("strand", "+")) == "-" else 1
        starts = store.starts[chrom]
        ends = store.ends[chrom]
        strands = store.strands[chrom] if store.strands is not None else None

        for j in range(len(starts)):
            s = int(starts[j])
            e = int(ends[j])
            fl = e - s
            if not (min_len <= fl < max_len):
                continue
            li = (fl - min_len) // len_bin

            rels: list[float] = []
            if feature == "midpoint":
                mid = 0.5 * (s + e)
                rels.append((mid - c) if a_st >= 0 else (c - mid))
            elif feature == "cuts":
                if strands is not None:
                    rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    if a_st >= 0:
                        rels.extend([s - c, e - c])
                    else:
                        rels.extend([c - e, c - s])
            elif feature == "cuts_plus":
                if strands is not None:
                    if int(strands[j]) * a_st > 0:
                        rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    rels.append((s - c) if a_st >= 0 else (c - e))
            elif feature == "cuts_minus":
                if strands is not None:
                    if int(strands[j]) * a_st < 0:
                        rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    rels.append((e - c) if a_st >= 0 else (c - s))

            for rel in rels:
                pi = int(math.floor((rel + upstream) / pos_bin))
                if 0 <= pi < n_pos:
                    grid[li, pi] += 1.0

    return grid


def _naive_matrix(
    store: FragmentStore,
    anchors: pd.DataFrame,
    upstream: int,
    downstream: int,
    bin_size: int,
    feature: str,
    len_range: tuple[int, int] | None,
) -> np.ndarray:
    """Independent O(N_anchors * N_frags) brute-force Python reference for matrix extraction."""
    n_bins = (upstream + downstream) // bin_size
    out = np.zeros((len(anchors), n_bins), dtype=np.float64)

    for i, (_, row) in enumerate(anchors.iterrows()):
        chrom = str(row["chrom"])
        if chrom not in store.starts:
            continue
        c = float(row["center"])
        a_st = -1 if str(row.get("strand", "+")) == "-" else 1
        starts = store.starts[chrom]
        ends = store.ends[chrom]
        strands = store.strands[chrom] if store.strands is not None else None

        for j in range(len(starts)):
            s = int(starts[j])
            e = int(ends[j])
            fl = e - s
            if len_range is not None and not (len_range[0] <= fl <= len_range[1]):
                continue

            if feature == "coverage":
                r_start = (s - c) if a_st >= 0 else (c - e)
                r_end = (e - c) if a_st >= 0 else (c - s)
                for b in range(n_bins):
                    bin_lo = -upstream + b * bin_size
                    bin_hi = bin_lo + bin_size
                    ov = min(r_end, bin_hi) - max(r_start, bin_lo)
                    if ov > 0.0:
                        out[i, b] += ov / bin_size
                continue

            rels: list[float] = []
            if feature == "midpoint":
                mid = 0.5 * (s + e)
                rels.append((mid - c) if a_st >= 0 else (c - mid))
            elif feature == "cuts":
                if strands is not None:
                    rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    if a_st >= 0:
                        rels.extend([s - c, e - c])
                    else:
                        rels.extend([c - e, c - s])
            elif feature == "cuts_plus":
                if strands is not None:
                    if int(strands[j]) * a_st > 0:
                        rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    rels.append((s - c) if a_st >= 0 else (c - e))
            elif feature == "cuts_minus":
                if strands is not None:
                    if int(strands[j]) * a_st < 0:
                        rels.append((s - c) if a_st >= 0 else (c - s))
                else:
                    rels.append((e - c) if a_st >= 0 else (c - s))

            for rel in rels:
                bi = int(math.floor((rel + upstream) / bin_size))
                if 0 <= bi < n_bins:
                    out[i, bi] += 1.0

    return out


def _make_random_store(seed: int = 42, n_frags: int = 800) -> FragmentStore:
    rng = np.random.default_rng(seed)
    starts = {}
    ends = {}
    for chrom in ("chr1", "chr2"):
        s = rng.integers(500, 5000, size=n_frags, dtype=np.int32)
        fl = rng.integers(15, 300, size=n_frags, dtype=np.int32)
        starts[chrom] = s
        ends[chrom] = s + fl
    return FragmentStore(starts=starts, ends=ends, assay="chec")


@pytest.mark.parametrize("feature", ["midpoint", "cuts", "cuts_plus", "cuts_minus"])
@pytest.mark.parametrize("window", [(120, 120, 4), (150, 90, 6), (60, 60, 1)])
def test_vplot_exact_match_vs_naive_and_numba_parity(feature: str, window: tuple[int, int, int]):
    upstream, downstream, pos_bin = window
    store = _make_random_store(seed=101, n_frags=400)
    rng = np.random.default_rng(202)
    anchors = pd.DataFrame(
        {
            "chrom": rng.choice(["chr1", "chr2", "chr_missing"], size=40),
            "center": rng.integers(600, 4800, size=40),
            "strand": rng.choice(["+", "-"], size=40),
        }
    )

    ref = _naive_vplot(
        store, anchors, upstream=upstream, downstream=downstream,
        min_len=20, max_len=260, pos_bin=pos_bin, len_bin=5, feature=feature,
    )
    grid_np, _ = accumulate_vplot(
        store, anchors, upstream=upstream, downstream=downstream,
        min_len=20, max_len=260, pos_bin=pos_bin, len_bin=5, feature=feature, use_numba=False,
    )
    np.testing.assert_allclose(grid_np, ref, rtol=0, atol=1e-12)

    if HAS_NUMBA:
        grid_nb, _ = accumulate_vplot(
            store, anchors, upstream=upstream, downstream=downstream,
            min_len=20, max_len=260, pos_bin=pos_bin, len_bin=5, feature=feature, use_numba=True,
        )
        np.testing.assert_allclose(grid_nb, ref, rtol=0, atol=1e-12)


def test_vplot_parallel_chunked_kernel_parity():
    """Exercise >= 256 anchors on a single chrom so _vplot_parallel_kernel runs."""
    store = _make_random_store(seed=77, n_frags=1000)
    rng = np.random.default_rng(88)
    anchors = pd.DataFrame(
        {
            "chrom": ["chr1"] * 350,
            "center": rng.integers(800, 4500, size=350),
            "strand": rng.choice(["+", "-"], size=350),
        }
    )
    grid_np, _ = accumulate_vplot(
        store, anchors, upstream=100, downstream=100,
        min_len=20, max_len=220, pos_bin=2, len_bin=2, feature="midpoint", use_numba=False,
    )
    grid_nb, _ = accumulate_vplot(
        store, anchors, upstream=100, downstream=100,
        min_len=20, max_len=220, pos_bin=2, len_bin=2, feature="midpoint", use_numba=True,
    )
    np.testing.assert_allclose(grid_nb, grid_np, rtol=0, atol=1e-12)


@pytest.mark.parametrize("feature", ["midpoint", "cuts", "cuts_plus", "cuts_minus", "coverage"])
@pytest.mark.parametrize("window", [(100, 100, 10), (120, 60, 5), (40, 40, 1)])
def test_matrix_exact_match_vs_naive_and_numba_parity(feature: str, window: tuple[int, int, int]):
    upstream, downstream, bin_size = window
    store = _make_random_store(seed=303, n_frags=500)
    rng = np.random.default_rng(404)
    anchors = pd.DataFrame(
        {
            "chrom": rng.choice(["chr1", "chr2", "chr_empty"], size=35),
            "center": rng.integers(700, 4600, size=35) + rng.choice([0.0, 0.5], size=35),
            "strand": rng.choice(["+", "-"], size=35),
        }
    )

    ref = _naive_matrix(
        store, anchors, upstream=upstream, downstream=downstream,
        bin_size=bin_size, feature=feature, len_range=(30, 180),
    )
    mat_np, _ = accumulate_matrix(
        store, anchors, upstream=upstream, downstream=downstream,
        bin_size=bin_size, feature=feature, len_range=(30, 180), use_numba=False,
    )
    np.testing.assert_allclose(mat_np, ref, rtol=0, atol=1e-12)

    if HAS_NUMBA:
        mat_nb, _ = accumulate_matrix(
            store, anchors, upstream=upstream, downstream=downstream,
            bin_size=bin_size, feature=feature, len_range=(30, 180), use_numba=True,
        )
        np.testing.assert_allclose(mat_nb, ref, rtol=0, atol=1e-12)


def test_asymmetric_motif_strand_flip_invariance():
    """Reflecting fragments around center and flipping anchor strand must yield identical signal."""
    center = 2000
    # Deliberately asymmetric fragment set around a '+' motif (e.g., CTCF / directional TSS):
    # - Upstream short TF fragment: [1940, 1985)  (rel left=-60, right=-15, mid=-37.5)
    # - Spanning fragment: [1970, 2060)           (rel left=-30, right=+60, mid=+15)
    # - Downstream +1 nucleosome: [2040, 2195)    (rel left=+40, right=+195, mid=+117.5)
    s_plus = np.array([1940, 1970, 2040], dtype=np.int32)
    e_plus = np.array([1985, 2060, 2195], dtype=np.int32)
    store_plus = FragmentStore({"chr1": s_plus}, {"chr1": e_plus}, assay="chec")
    anchor_plus = pd.DataFrame({"chrom": ["chr1"], "center": [center], "strand": ["+"]})

    # Reflect every interval [s, e) -> [2*center - e, 2*center - s) for a '-' strand motif!
    s_minus = (2 * center - e_plus).astype(np.int32)
    e_minus = (2 * center - s_plus).astype(np.int32)
    store_minus = FragmentStore({"chr1": s_minus}, {"chr1": e_minus}, assay="chec")
    anchor_minus = pd.DataFrame({"chrom": ["chr1"], "center": [center], "strand": ["-"]})

    for feat in ("midpoint", "cuts", "cuts_plus", "cuts_minus", "coverage"):
        m_p, _ = accumulate_matrix(store_plus, anchor_plus, upstream=250, downstream=250, bin_size=1, feature=feat)
        m_m, _ = accumulate_matrix(store_minus, anchor_minus, upstream=250, downstream=250, bin_size=1, feature=feat)
        np.testing.assert_allclose(m_p, m_m, rtol=0, atol=1e-12)

    for feat in ("midpoint", "cuts", "cuts_plus", "cuts_minus"):
        v_p, _ = accumulate_vplot(
            store_plus, anchor_plus, upstream=250, downstream=250,
            min_len=20, max_len=220, pos_bin=1, len_bin=1, feature=feat,
        )
        v_m, _ = accumulate_vplot(
            store_minus, anchor_minus, upstream=250, downstream=250,
            min_len=20, max_len=220, pos_bin=1, len_bin=1, feature=feat,
        )
        np.testing.assert_allclose(v_p, v_m, rtol=0, atol=1e-12)

    # Events must also match identically
    ev_p = extract_events(store_plus, anchor_plus, flank=250, overlap_mode="any").sort_values("rel_left").reset_index(drop=True)
    ev_m = extract_events(store_minus, anchor_minus, flank=250, overlap_mode="any").sort_values("rel_left").reset_index(drop=True)
    pd.testing.assert_frame_equal(ev_p, ev_m)


def test_chip_exo_stranded_extraction_and_strand_flip():
    """Verify ChIP-exo 5' exonuclease stop sites flip properly on '-' motifs."""
    center = 1000
    # '+' motif: upstream '+' read stops at -15 (985), downstream '-' read stops at +12 (1012)
    store_p = FragmentStore(
        starts={"chr1": np.array([985, 1012], dtype=np.int32)},
        ends={"chr1": np.array([986, 1013], dtype=np.int32)},
        strands={"chr1": np.array([1, -1], dtype=np.int8)},
        assay="chip_exo",
    )
    anchor_p = pd.DataFrame({"chrom": ["chr1"], "center": [center], "strand": ["+"]})

    # Reflected '-' motif: upstream read is genomic '-' at 1015, downstream is genomic '+' at 988
    store_m = FragmentStore(
        starts={"chr1": np.array([988, 1015], dtype=np.int32)},
        ends={"chr1": np.array([989, 1016], dtype=np.int32)},
        strands={"chr1": np.array([1, -1], dtype=np.int8)},
        assay="chip_exo",
    )
    anchor_m = pd.DataFrame({"chrom": ["chr1"], "center": [center], "strand": ["-"]})

    for feat in ("cuts", "cuts_plus", "cuts_minus"):
        mp_nb, _ = accumulate_matrix(store_p, anchor_p, upstream=50, downstream=50, bin_size=1, feature=feat, use_numba=True)
        mm_nb, _ = accumulate_matrix(store_m, anchor_m, upstream=50, downstream=50, bin_size=1, feature=feat, use_numba=True)
        mp_np, _ = accumulate_matrix(store_p, anchor_p, upstream=50, downstream=50, bin_size=1, feature=feat, use_numba=False)
        mm_np, _ = accumulate_matrix(store_m, anchor_m, upstream=50, downstream=50, bin_size=1, feature=feat, use_numba=False)
        np.testing.assert_allclose(mp_nb, mp_np)
        np.testing.assert_allclose(mm_nb, mm_np)
        np.testing.assert_allclose(mp_nb, mm_nb)

    # Specifically verify cuts_plus landed at rel=-15 (bin 35) and cuts_minus at rel=+12 (bin 62)
    mp_plus, _ = accumulate_matrix(store_p, anchor_p, upstream=50, downstream=50, bin_size=1, feature="cuts_plus")
    mp_minus, _ = accumulate_matrix(store_p, anchor_p, upstream=50, downstream=50, bin_size=1, feature="cuts_minus")
    assert mp_plus[0, 35] == 1.0 and np.sum(mp_plus) == 1.0
    assert mp_minus[0, 62] == 1.0 and np.sum(mp_minus) == 1.0


def test_boundary_and_spanning_fragments():
    """Verify fragments strictly spanning the entire anchor window are never missed by binary search."""
    # Fragment [100, 900) has length 800 and spans across anchor window [450, 550)
    store = FragmentStore(
        starts={"chr1": np.array([100, 450, 500], dtype=np.int32)},
        ends={"chr1": np.array([900, 470, 550], dtype=np.int32)},
        assay="generic",
    )
    anchors = pd.DataFrame({"chrom": ["chr1"], "center": [500], "strand": ["+"]})
    ref = _naive_matrix(store, anchors, upstream=50, downstream=50, bin_size=10, feature="coverage", len_range=None)
    mat_nb, _ = accumulate_matrix(store, anchors, upstream=50, downstream=50, bin_size=10, feature="coverage", use_numba=True)
    mat_np, _ = accumulate_matrix(store, anchors, upstream=50, downstream=50, bin_size=10, feature="coverage", use_numba=False)
    np.testing.assert_allclose(mat_nb, ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mat_np, ref, rtol=0, atol=1e-12)
    # Every bin in [-50, 50) must have >= 1.0 coverage from the spanning fragment [100, 900)
    assert np.all(mat_nb >= 1.0)
