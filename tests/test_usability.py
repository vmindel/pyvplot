"""Guard rails that turn silent failure modes into errors or warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pyvplot as vp


@pytest.fixture
def store() -> vp.FragmentStore:
    starts = np.array([900, 950, 1000, 1040, 1100], dtype=np.int32)
    lens = np.array([60, 150, 70, 155, 50], dtype=np.int32)
    return vp.FragmentStore({"chr1": starts}, {"chr1": starts + lens}, assay="chec")


@pytest.fixture
def anchors() -> pd.DataFrame:
    return pd.DataFrame({"chrom": ["chr1", "chr1"], "center": [1000, 1100], "strand": ["+", "-"]})


def test_warns_when_no_anchor_chrom_matches(store):
    ens = pd.DataFrame({"chrom": ["1", "1"], "center": [1000, 1100]})
    for call in (
        lambda: store.vplot(ens, flank=100),
        lambda: store.to_matrix(ens, flank=100),
        lambda: store.extract_events(ens, flank=100),
    ):
        with pytest.warns(UserWarning, match="chromosome naming"):
            call()


def test_warns_on_partial_chrom_mismatch(store):
    mixed = pd.DataFrame({"chrom": ["chr1", "chrX"], "center": [1000, 1000]})
    with pytest.warns(UserWarning, match="1 of 2 anchors"):
        store.to_matrix(mixed, flank=100)


def test_no_warning_when_chroms_match(store, anchors):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        store.vplot(anchors, flank=100)
        store.to_matrix(anchors, flank=100)


def test_named_size_bins_accepted_wherever_length_is_a_filter(store, anchors):
    lo, hi = store.assay.size_bins["tf"]
    ev_named = store.extract_events(anchors, flank=200, len_range="tf")
    ev_tuple = store.extract_events(anchors, flank=200, len_range=(lo, hi))
    pd.testing.assert_frame_equal(ev_named, ev_tuple)
    assert ev_named["frag_len"].between(lo, hi).all()

    ph_named = store.phasogram(max_lag=200, len_range="mononucleosome")
    ph_tuple = store.phasogram(max_lag=200, len_range=store.assay.size_bins["mononucleosome"])
    np.testing.assert_array_equal(ph_named.counts, ph_tuple.counts)

    v = store.vplot(anchors, flank=100, min_len=20, max_len=200)
    np.testing.assert_allclose(
        v.marginal_position(len_range="tf")["value"],
        v.marginal_position(len_range=(lo, hi))["value"],
    )


def test_unknown_size_bin_lists_available(store, anchors):
    with pytest.raises(ValueError, match="Available size bins: mononucleosome, tf"):
        store.to_matrix(anchors, len_range="nfr")


def test_vplot_rejects_named_bin_with_pointer(store, anchors):
    with pytest.raises(TypeError, match="len_range="):
        store.vplot(anchors, min_len="tf")  # type: ignore[arg-type]


def test_cuts_stranded_feature_is_accepted():
    df = pd.DataFrame(
        {"chrom": ["chr1"] * 3, "start": [990, 1000, 1010], "end": [991, 1001, 1011], "strand": ["+", "-", "+"]}
    )
    exo = vp.FragmentStore.from_dataframe(df, assay="chip_exo", strand_col="strand")
    a = pd.DataFrame({"chrom": ["chr1"], "center": [1000]})
    m_default = exo.to_matrix(a, flank=20, bin_size=1)
    m_explicit = exo.to_matrix(a, flank=20, bin_size=1, feature="cuts_stranded")
    np.testing.assert_array_equal(m_default.values, m_explicit.values)
    v = exo.vplot(a, flank=20, min_len=1, max_len=2, feature="cuts_stranded")
    assert v.counts.sum() == 3


def test_kmer_bias_from_fasta_path(tmp_path):
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 5_000))
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1 description\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n")
    starts = np.array([i + 2 for i in range(10, 4_800) if seq[i : i + 4] == "GATC"] * 10, dtype=np.int32)
    store = vp.FragmentStore({"chr1": starts}, {"chr1": starts + 80}, assay="chec")

    from_path = vp.KmerBiasTable.from_cuts(store, fa, k=4, cut_offset=2)
    from_dict = vp.KmerBiasTable.from_cuts(store, {"chr1": seq}, k=4, cut_offset=2)
    np.testing.assert_allclose(from_path.weights, from_dict.weights)
    assert from_path["GATC"] > 5.0

    from pyvplot.bias import _read_fasta

    assert _read_fasta(fa) == {"chr1": seq}


def test_kmer_bias_raises_when_no_chrom_matches(store):
    with pytest.raises(ValueError, match="chromosome naming"):
        vp.KmerBiasTable.from_cuts(store, {"1": "ACGT" * 100}, k=2)


def test_reprs_are_short(store, anchors):
    objs = [
        store.vplot(anchors, flank=100),
        store.to_matrix(anchors, flank=100),
        store.cut_site_footprint(anchors, flank=50),
        store.phasogram(max_lag=200),
        vp.VPlotCollection().add(store.vplot(anchors, flank=100), condition="WT"),
    ]
    for o in objs:
        r = repr(o)
        assert len(r) < 300 and "array(" not in r, r


def test_expected_normalization_warns_on_empty_flanks(store, anchors):
    v = store.vplot(anchors, flank=100, min_len=20, max_len=200)
    with pytest.warns(UserWarning, match="zero expected"):
        v.normalize("expected")


def test_simulate_chec_shapes_and_unbound_condition():
    wt, anchors = vp.datasets.simulate_chec(n_sites=10, seed=3)
    ko, anchors_ko = vp.datasets.simulate_chec(n_sites=10, tf_fragments=0, seed=3)
    assert list(anchors.columns) == ["chrom", "center", "strand"]
    assert len(anchors) == 10 and anchors.index[0] == "site_0"
    pd.testing.assert_frame_equal(anchors, anchors_ko)
    assert wt.n_fragments == 10 * (30 + 80 + 100)
    tf_wt = wt.to_matrix(anchors, flank=20, bin_size=40, feature="midpoint", len_range="tf").values.sum()
    tf_ko = ko.to_matrix(anchors, flank=20, bin_size=40, feature="midpoint", len_range="tf").values.sum()
    assert tf_wt > 5 * max(tf_ko, 1)


def test_diff_log2fc_does_not_depend_on_off_target_depth():
    """Scaling both libraries' depth with off-target reads must not shrink the default log2FC.

    Each store is deepened in proportion to its size so the library-size ratio,
    which CPM legitimately uses, stays the same.
    """
    wt, anchors = vp.datasets.simulate_chec(n_sites=30, seed=1)
    ko, _ = vp.datasets.simulate_chec(n_sites=30, tf_fragments=0, seed=2)

    def deepen(store: vp.FragmentStore, n: int) -> vp.FragmentStore:
        s = np.arange(10_000_000, 10_000_000 + 1_000 * n, 1_000, dtype=np.int64)
        extra = vp.FragmentStore.from_arrays({"chr1": s}, {"chr1": s + 100}, assay="chec")
        return vp.FragmentStore.merge([store, extra])

    kw = dict(flank=250, min_len=20, max_len=200, pos_bin=10, len_bin=10)
    shallow = ko.vplot(anchors, **kw).diff(wt.vplot(anchors, **kw))
    deep = deepen(ko, 30 * ko.n_fragments).vplot(anchors, **kw).diff(
        deepen(wt, 30 * wt.n_fragments).vplot(anchors, **kw)
    )
    np.testing.assert_allclose(deep.counts, shallow.counts, atol=1e-9)
    assert shallow.counts.min() < -3


def test_diff_cells_empty_in_both_samples_are_zero_at_unequal_depth():
    """A shared pseudocount: no signal in either sample means log2FC 0, whatever the depths."""
    wt, anchors = vp.datasets.simulate_chec(n_sites=30, seed=1)
    ko, _ = vp.datasets.simulate_chec(n_sites=30, tf_fragments=0, seed=2)
    s = np.arange(10_000_000, 10_000_000 + 1_000 * 50_000, 1_000, dtype=np.int64)
    extra = vp.FragmentStore.from_arrays({"chr1": s}, {"chr1": s + 100}, assay="chec")
    deep_ko = vp.FragmentStore.merge([ko, extra])  # ~10x deeper, all off-target

    kw = dict(flank=250, min_len=20, max_len=400, pos_bin=10, len_bin=10)
    a, b = deep_ko.vplot(anchors, **kw), wt.vplot(anchors, **kw)
    both_empty = (a.counts == 0) & (b.counts == 0)
    assert both_empty.sum() > 10
    d = a.diff(b)
    np.testing.assert_array_equal(d.counts[both_empty], 0.0)
