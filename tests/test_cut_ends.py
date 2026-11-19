"""Proximal/distal cut features and the motif_overlap filter, checked against a brute-force reference."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import pyvplot as vp
from pyvplot.extract import HAS_NUMBA, accumulate_matrix, accumulate_vplot, extract_events
from pyvplot.store import FragmentStore

OVERLAPS = [None, "any", "full", "none"]
BACKENDS = [False, True] if HAS_NUMBA else [False]


def _passes(s: int, e: int, m_lo: float, m_hi: float, mode: str | None) -> bool:
    hit = s < m_hi and e > m_lo
    if mode is None:
        return True
    if mode == "any":
        return hit
    if mode == "full":
        return s <= m_lo and e >= m_hi
    return not hit


def _end_positions(s: int, e: int, c: float, a_st: int, feature: str) -> list[float]:
    """Anchor-oriented positions of the requested end(s), from genomic distances."""
    # Genomic ends and their upstream/downstream order in anchor orientation.
    up_end, dn_end = (s, e) if a_st >= 0 else (e, s)
    rel = (lambda x: x - c) if a_st >= 0 else (lambda x: c - x)
    d_up, d_dn = abs(up_end - c), abs(dn_end - c)
    near, far = (up_end, dn_end) if d_up <= d_dn else (dn_end, up_end)
    if feature == "cuts":
        return [rel(s), rel(e)]
    if feature == "midpoint":
        return [rel(0.5 * (s + e))]
    if feature == "cuts_proximal":
        return [rel(near)]
    if feature == "cuts_distal":
        return [rel(far)]
    raise AssertionError(feature)


def _anchor_rows(anchors: pd.DataFrame):
    for _, row in anchors.iterrows():
        a_st = -1 if row["strand"] == "-" else 1
        if "start" in row:
            m_lo, m_hi = float(row["start"]), float(row["end"])
            c = float(row["center"]) if "center" in row else 0.5 * (m_lo + m_hi)
        else:
            c = float(row["center"])
            m_lo, m_hi = c - 0.5, c + 0.5
        yield str(row["chrom"]), c, a_st, m_lo, m_hi


def _naive_matrix(store, anchors, up, dn, bin_size, feature, overlap, len_range=(0, 10**9)):
    n_bins = (up + dn) // bin_size
    out = np.zeros((len(anchors), n_bins))
    for i, (chrom, c, a_st, m_lo, m_hi) in enumerate(_anchor_rows(anchors)):
        if chrom not in store.starts:
            continue
        for s, e in zip(store.starts[chrom].tolist(), store.ends[chrom].tolist(), strict=True):
            if not (len_range[0] <= e - s <= len_range[1]) or not _passes(s, e, m_lo, m_hi, overlap):
                continue
            for r in _end_positions(s, e, c, a_st, feature):
                b = math.floor((r + up) / bin_size)
                if 0 <= b < n_bins:
                    out[i, b] += 1.0
    return out


def _naive_vplot(store, anchors, up, dn, pos_bin, min_len, max_len, len_bin, feature, overlap):
    n_pos, n_len = (up + dn) // pos_bin, (max_len - min_len) // len_bin
    grid = np.zeros((n_len, n_pos))
    for chrom, c, a_st, m_lo, m_hi in _anchor_rows(anchors):
        if chrom not in store.starts:
            continue
        for s, e in zip(store.starts[chrom].tolist(), store.ends[chrom].tolist(), strict=True):
            fl = e - s
            if not (min_len <= fl < max_len) or not _passes(s, e, m_lo, m_hi, overlap):
                continue
            li = (fl - min_len) // len_bin
            for r in _end_positions(s, e, c, a_st, feature):
                pi = math.floor((r + up) / pos_bin)
                if 0 <= pi < n_pos:
                    grid[li, pi] += 1.0
    return grid


def _store(seed: int = 7, n: int = 600) -> FragmentStore:
    rng = np.random.default_rng(seed)
    starts, ends = {}, {}
    for chrom in ("chr1", "chr2"):
        s = rng.integers(500, 3000, size=n).astype(np.int32)
        starts[chrom] = s
        ends[chrom] = (s + rng.integers(15, 260, size=n)).astype(np.int32)
    return FragmentStore(starts=starts, ends=ends, assay="chec")


def _interval_anchors(seed: int = 8, n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = rng.integers(700, 2800, size=n)
    return pd.DataFrame(
        {
            "chrom": rng.choice(["chr1", "chr2"], size=n),
            "start": start,
            "end": start + rng.integers(1, 20, size=n),
            "strand": rng.choice(["+", "-"], size=n),
        }
    )


def _point_anchors(seed: int = 9, n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "chrom": rng.choice(["chr1", "chr2"], size=n),
            "center": rng.integers(700, 2800, size=n) + rng.choice([0.0, 0.5], size=n),
            "strand": rng.choice(["+", "-"], size=n),
        }
    )


@pytest.mark.parametrize("anchor_kind", ["interval", "point"])
@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("feature", ["midpoint", "cuts", "cuts_proximal", "cuts_distal"])
@pytest.mark.parametrize("use_numba", BACKENDS)
def test_matrix_matches_naive(anchor_kind, overlap, feature, use_numba):
    store = _store()
    anchors = _interval_anchors() if anchor_kind == "interval" else _point_anchors()
    ref = _naive_matrix(store, anchors, 150, 90, 3, feature, overlap, len_range=(20, 240))
    got, _ = accumulate_matrix(
        store, anchors, upstream=150, downstream=90, bin_size=3, feature=feature,
        len_range=(20, 240), motif_overlap=overlap, use_numba=use_numba,
    )
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("feature", ["midpoint", "cuts", "cuts_proximal", "cuts_distal"])
@pytest.mark.parametrize("use_numba", BACKENDS)
def test_vplot_matches_naive(overlap, feature, use_numba):
    store = _store()
    anchors = _interval_anchors()
    ref = _naive_vplot(store, anchors, 120, 120, 4, 20, 260, 5, feature, overlap)
    got, _ = accumulate_vplot(
        store, anchors, upstream=120, downstream=120, min_len=20, max_len=260,
        pos_bin=4, len_bin=5, feature=feature, motif_overlap=overlap, use_numba=use_numba,
    )
    np.testing.assert_array_equal(got, ref)


def test_vplot_parallel_kernel_with_filter():
    """>= 256 anchors on one chromosome runs the chunked Numba kernel."""
    if not HAS_NUMBA:
        pytest.skip("numba not installed")
    store = _store(seed=3, n=1500)
    rng = np.random.default_rng(4)
    start = rng.integers(700, 2800, size=300)
    anchors = pd.DataFrame(
        {"chrom": "chr1", "start": start, "end": start + 12, "strand": rng.choice(["+", "-"], size=300)}
    )
    kw = dict(upstream=100, downstream=100, min_len=20, max_len=220, pos_bin=2, len_bin=2,
              feature="cuts_proximal", motif_overlap="any")
    g_nb, _ = accumulate_vplot(store, anchors, use_numba=True, **kw)
    g_np, _ = accumulate_vplot(store, anchors, use_numba=False, **kw)
    np.testing.assert_array_equal(g_nb, g_np)


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_proximal_plus_distal_equals_cuts(overlap):
    store = _store()
    anchors = _interval_anchors()
    kw = dict(upstream=100, downstream=100, bin_size=1, motif_overlap=overlap)
    prox, _ = accumulate_matrix(store, anchors, feature="cuts_proximal", **kw)
    dist, _ = accumulate_matrix(store, anchors, feature="cuts_distal", **kw)
    cuts, _ = accumulate_matrix(store, anchors, feature="cuts", **kw)
    np.testing.assert_array_equal(prox + dist, cuts)


@pytest.mark.parametrize("feature", ["midpoint", "cuts", "cuts_proximal"])
def test_overlap_partitions(feature):
    store = _store()
    anchors = _interval_anchors()
    kw = dict(upstream=100, downstream=100, bin_size=2, feature=feature)
    m = {ov: accumulate_matrix(store, anchors, motif_overlap=ov, **kw)[0] for ov in OVERLAPS}
    np.testing.assert_array_equal(m["any"] + m["none"], m[None])
    assert np.all(m["full"] <= m["any"])


def test_proximal_ends_bracket_spanning_fragments():
    """For fragments containing the motif, the proximal end lies on the near side of each flank."""
    store = FragmentStore({"chr1": np.array([950, 990, 1003])}, {"chr1": np.array([1030, 1100, 1060])})
    anchors = pd.DataFrame({"chrom": ["chr1"], "start": [998], "end": [1002], "strand": ["+"]})
    ev = extract_events(store, anchors, flank=200, overlap_mode="any")
    ev = ev.sort_values("rel_left").reset_index(drop=True)
    # Fragments: [950,1030) mid 990 -> near end 1030 (+30); [990,1100) -> near 990 (-10);
    # [1003,1060) starts right of the motif -> near end 1003 (+3), no overlap.
    np.testing.assert_array_equal(ev["rel_proximal"], [30.0, -10.0, 3.0])
    np.testing.assert_array_equal(ev["rel_distal"], [-50.0, 100.0, 60.0])
    assert ev["motif_overlap"].tolist() == ["full", "full", "none"]


def test_tie_goes_to_upstream_end_and_flips_with_strand():
    # Fragment [980, 1020) centred exactly on 1000.
    store = FragmentStore({"chr1": np.array([980])}, {"chr1": np.array([1020])})
    for strand in ("+", "-"):
        anchors = pd.DataFrame({"chrom": ["chr1"], "center": [1000.0], "strand": [strand]})
        prox, _ = accumulate_matrix(store, anchors, 50, 50, bin_size=1, feature="cuts_proximal")
        dist, _ = accumulate_matrix(store, anchors, 50, 50, bin_size=1, feature="cuts_distal")
        assert prox[0, 50 - 20] == 1.0 and prox.sum() == 1.0
        assert dist[0, 50 + 20] == 1.0 and dist.sum() == 1.0


def test_strand_flip_invariance_with_intervals():
    center = 2000
    s = np.array([1940, 1970, 1995, 2040], dtype=np.int32)
    e = np.array([1985, 2060, 2003, 2195], dtype=np.int32)
    st_p = FragmentStore({"chr1": s}, {"chr1": e})
    st_m = FragmentStore({"chr1": (2 * center - e).astype(np.int32)}, {"chr1": (2 * center - s).astype(np.int32)})
    an_p = pd.DataFrame({"chrom": ["chr1"], "start": [1996], "end": [2001], "strand": ["+"]})
    an_m = pd.DataFrame({"chrom": ["chr1"], "start": [1999], "end": [2004], "strand": ["-"]})
    # The two motifs mirror each other around 2000 only if their centres match the mirror.
    an_p["center"] = 1998.5
    an_m["center"] = 2001.5
    for feat in ("cuts_proximal", "cuts_distal", "midpoint"):
        for ov in OVERLAPS:
            mp, _ = accumulate_matrix(st_p, an_p, 250, 250, bin_size=1, feature=feat, motif_overlap=ov)
            mm, _ = accumulate_matrix(st_m, an_m, 250, 250, bin_size=1, feature=feat, motif_overlap=ov)
            np.testing.assert_array_equal(mp, mm)


def test_events_filter_matches_matrix_counts():
    store = _store()
    anchors = _interval_anchors()
    for ov in OVERLAPS:
        ev = extract_events(store, anchors, upstream=100, downstream=100, overlap_mode="midpoint", motif_overlap=ov)
        mat, _ = accumulate_matrix(store, anchors, 100, 100, bin_size=1, feature="midpoint", motif_overlap=ov)
        per_anchor = ev.groupby("anchor_idx").size().reindex(range(len(anchors)), fill_value=0).to_numpy()
        np.testing.assert_array_equal(per_anchor, mat.sum(axis=1))
        if ov is not None:
            expected = {"any": {"full", "partial"}, "full": {"full"}, "none": {"none"}}[ov]
            assert set(ev["motif_overlap"].astype(str)) <= expected


def test_bed_file_keeps_motif_interval(tmp_path):
    store = FragmentStore({"chr1": np.array([900, 1004])}, {"chr1": np.array([1003, 1100])})
    bed = tmp_path / "m.bed"
    bed.write_text("chr1\t995\t1005\tm1\t0\t+\n")
    # Both fragments overlap [995, 1005); neither covers the 1-bp centre [999.5, 1000.5) only.
    ev_any = extract_events(store, bed, flank=200, overlap_mode="any", motif_overlap="any")
    assert len(ev_any) == 2
    ev_full = extract_events(store, bed, flank=200, overlap_mode="any", motif_overlap="full")
    assert len(ev_full) == 0


def test_stranded_store_rejects_proximity_and_overlap():
    store = FragmentStore(
        starts={"chr1": np.array([985])}, ends={"chr1": np.array([986])},
        strands={"chr1": np.array([1], dtype=np.int8)}, assay="chip_exo",
    )
    anchors = pd.DataFrame({"chrom": ["chr1"], "center": [1000], "strand": ["+"]})
    with pytest.raises(ValueError, match="stranded"):
        store.to_matrix(anchors, flank=50, bin_size=1, feature="cuts_proximal")
    with pytest.raises(ValueError, match="stranded"):
        store.vplot(anchors, flank=50, feature="cuts", motif_overlap="any")


def test_invalid_motif_overlap():
    store, anchors = vp.datasets.simulate_chec()
    with pytest.raises(ValueError, match="motif_overlap"):
        store.to_matrix(anchors, motif_overlap="spans")


def test_genome_tracks_reject_proximity_features():
    store, _ = vp.datasets.simulate_chec()
    with pytest.raises(ValueError, match="Unsupported feature"):
        vp.to_binned_track(store, feature="cuts_proximal")


def test_cut_site_footprint_proximity_split():
    store, anchors = vp.datasets.simulate_chec()
    fp_s = store.cut_site_footprint(anchors, flank=60, len_range="tf")
    fp_p = store.cut_site_footprint(anchors, flank=60, len_range="tf", split_by="proximity")
    np.testing.assert_allclose(fp_p.combined, fp_s.combined)
    np.testing.assert_allclose(fp_p.proximal + fp_p.distal, fp_p.combined)
    assert fp_p.plus is None and fp_p.split_by == "proximity"
    df = fp_p.to_dataframe()
    assert set(df["end"]) == {"proximal", "distal"}
    fp_any = store.cut_site_footprint(anchors, flank=60, len_range="tf", split_by="proximity", motif_overlap="any")
    assert fp_any.motif_overlap == "any" and fp_any.matrix_proximal.motif_overlap == "any"
    assert fp_any.combined.sum() < fp_p.combined.sum()


def test_vplot_records_filter_and_guards_diff_and_normalize():
    store, anchors = vp.datasets.simulate_chec()
    kw = dict(flank=200, pos_bin=5, len_bin=5)
    v_all = store.vplot(anchors, **kw)
    v_any = store.vplot(anchors, motif_overlap="any", **kw)
    assert v_any.motif_overlap == "any" and "motif_overlap='any'" in repr(v_any)
    assert v_any.normalize("cpm").motif_overlap == "any"
    with pytest.raises(ValueError, match="motif_overlap"):
        v_any.diff(v_all)
    with pytest.raises(ValueError, match="feature"):
        store.vplot(anchors, feature="cuts_proximal", **kw).diff(v_all)
    with pytest.warns(UserWarning, match="depleted by construction"):
        v_any.normalize("expected")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.vplot(anchors, motif_overlap="none", **kw).normalize("cpm")
