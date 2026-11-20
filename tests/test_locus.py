"""Single-locus fragment view, checked against a brute-force reference."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pyvplot.store import FragmentStore


@pytest.fixture(scope="module")
def store() -> FragmentStore:
    rng = np.random.default_rng(7)
    s = rng.integers(0, 5000, 400)
    e = s + rng.integers(20, 250, 400)
    return FragmentStore.from_arrays({"chr1": s, "chr2": s[:50]}, {"chr1": e, "chr2": e[:50]}, assay="chec")


def _frags(store: FragmentStore):
    return list(zip(store.starts["chr1"].tolist(), store.ends["chr1"].tolist(), strict=True))


REGION = (1000, 3000)
MOTIF = (1980, 2000)  # centre 1990


def test_midpoint_genomic_matches_reference(store):
    lv = store.locus_vplot("chr1:1000-3000")
    exp = sorted((0.5 * (s + e), e - s) for s, e in _frags(store) if 1000 <= 0.5 * (s + e) < 3000)
    got = sorted(zip(lv.points["x"], lv.points["frag_len"], strict=True))
    assert got == exp
    assert lv.coords == "genomic" and lv.xlim == (1000.0, 3000.0)
    assert (lv.points["x"] == lv.points["pos"]).all()


@pytest.mark.parametrize("strand", ["+", "-"])
def test_motif_cuts_split_proximal_distal(store, strand):
    lv = store.locus_vplot(("chr1", *REGION), motif=("chr1", *MOTIF, strand), feature="cuts")
    c = 0.5 * sum(MOTIF)
    sgn = 1 if strand == "+" else -1
    exp_prox, exp_dist = [], []
    for s, e in _frags(store):
        near, far = (s, e) if abs(s - c) <= abs(e - c) else (e, s)
        for kind, g in (("p", near), ("d", far)):
            if REGION[0] <= g < REGION[1]:
                (exp_prox if kind == "p" else exp_dist).append(sgn * (g - c))
    pts = lv.points
    assert sorted(pts.loc[pts["end"] == "proximal", "x"]) == pytest.approx(sorted(exp_prox))
    assert sorted(pts.loc[pts["end"] == "distal", "x"]) == pytest.approx(sorted(exp_dist))
    # genomic position of each point is recoverable regardless of orientation
    assert np.allclose(pts["pos"], c + sgn * pts["x"])


def test_single_end_features_partition_cuts(store):
    kw = dict(motif="chr1:1980-2000")
    both = store.locus_vplot("chr1:1000-3000", feature="cuts", **kw)
    prox = store.locus_vplot("chr1:1000-3000", feature="cuts_proximal", **kw)
    dist = store.locus_vplot("chr1:1000-3000", feature="cuts_distal", **kw)
    assert len(both.points) == len(prox.points) + len(dist.points)
    assert (prox.points["end"] == "proximal").all() and (dist.points["end"] == "distal").all()
    left = store.locus_vplot("chr1:1000-3000", feature="cuts_plus")
    right = store.locus_vplot("chr1:1000-3000", feature="cuts_minus")
    assert (left.points["end"] == "left").all() and (right.points["end"] == "right").all()


@pytest.mark.parametrize("mode", ["any", "full", "none"])
def test_motif_overlap_filter(store, mode):
    lv = store.locus_vplot("chr1:1000-3000", motif="chr1:1980-2000", feature="midpoint", motif_overlap=mode)
    lo, hi = MOTIF
    def ok(s, e):
        hit = s < hi and e > lo
        return {"any": hit, "full": s <= lo and e >= hi, "none": not hit}[mode]
    exp = sorted(0.5 * (s + e) - 1990 for s, e in _frags(store) if ok(s, e) and 1000 <= 0.5 * (s + e) < 3000)
    assert sorted(lv.points["x"]) == pytest.approx(exp)


def test_matches_aggregate_vplot_for_single_anchor(store):
    # even-length motif: integer centre, so both bin grids start on the same edge
    lv = store.locus_vplot("chr1:1501-2501", motif="chr1:1990-1992", feature="cuts_proximal")
    v = store.vplot(
        {"chrom": ["chr1"], "center": [1991], "start": [1990], "end": [1992]},
        upstream=490, downstream=510, min_len=0, max_len=500, pos_bin=1, len_bin=1, feature="cuts_proximal",
    )
    assert v.counts.sum() == len(lv.points)
    b = lv.to_vplot(pos_bin=1, len_bin=1, min_len=0, max_len=500)
    assert np.array_equal(b.counts, v.counts)


def test_len_range_and_empty(store):
    lv = store.locus_vplot("chr1:1000-3000", len_range=(50, 80))
    assert lv.points["frag_len"].between(50, 80).all()
    empty = store.locus_vplot("chr1:100000-100500")
    assert len(empty.points) == 0
    empty.plot()


def test_errors(store):
    with pytest.raises(ValueError, match="relative to a motif"):
        store.locus_vplot("chr1:1000-3000", feature="cuts_proximal")
    with pytest.raises(ValueError, match="needs a motif"):
        store.locus_vplot("chr1:1000-3000", motif_overlap="any")
    with pytest.raises(ValueError, match="outside the region"):
        store.locus_vplot("chr1:1000-3000", motif="chr1:4000-4010")
    with pytest.raises(ValueError, match="Cannot parse"):
        store.locus_vplot("chr1-1000-3000")
    with pytest.raises(ValueError, match="Unsupported feature"):
        store.locus_vplot("chr1:1000-3000", feature="coverage")
    with pytest.raises(ValueError, match="coords"):
        store.locus_vplot("chr1:1000-3000", coords="bad")


def test_plots_and_vplot(store):
    lv = store.locus_vplot("chr1:1,000-3,000", motif="chr1:1980-2000:-", feature="cuts")
    assert lv.motif_strand == "-" and lv.motif_span == (-10.0, 10.0)
    fig, ax = lv.plot(color_by="motif_overlap")
    assert ax.get_xlim() == pytest.approx(lv.xlim)
    lv.plot(color_by=None)
    assert lv.plot_plotnine() is not None
    lv.to_vplot(pos_bin=10, len_bin=10).plot()
    g = store.locus_vplot("chr1:1000-3000", motif="chr1:1980-2000", coords="genomic")
    assert g.motif_span == (1980.0, 2000.0)
    g.to_vplot().plot(center_line=False, xlabel="chr1")
    assert "LocusVPlot" in repr(lv)


def test_missing_chrom_gives_empty(store):
    with pytest.warns(UserWarning):
        lv = store.locus_vplot("chrZ:0-100")
    assert len(lv.points) == 0
