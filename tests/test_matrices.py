"""Tests for SignalMatrix, cut-site footprints, and footprint scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvplot.matrices import cut_site_footprint, footprint_score
from pyvplot.store import FragmentStore


def _make_protected_motif_store() -> tuple[FragmentStore, pd.DataFrame]:
    """Create two anchors: anchor 0 has a strong TF footprint (protected center, high flanks); anchor 1 is flat."""
    anchors = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "center": [2000, 6000],
            "strand": ["+", "-"],
        },
        index=["motif_bound", "motif_unbound"],
    )

    starts: list[int] = []
    ends: list[int] = []

    # Bound motif at 2000: cuts piled up at [-25..-12] and [+12..+25], ZERO cuts in [-10, +10)
    for offset in range(12, 26):
        # Fragments spanning across the motif with left cut at 2000 - offset, right cut at 2000 + offset
        for _ in range(5):
            starts.append(2000 - offset)
            ends.append(2000 + offset)

    # Unbound motif at 6000: uniform cuts everywhere across [-35, +35)
    for pos in range(5965, 6035):
        starts.append(pos)
        ends.append(pos + 60)

    store = FragmentStore(
        starts={"chr1": np.asarray(starts, dtype=np.int32)},
        ends={"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )
    return store, anchors


def test_to_matrix_size_bins_and_normalizations():
    store = FragmentStore(
        starts={"chr1": np.array([980, 930, 1010], dtype=np.int32)},
        ends={"chr1": np.array([1040, 1080, 1060], dtype=np.int32)},
        assay="chec",
    )
    anchors = pd.DataFrame({"chrom": ["chr1"], "center": [1000], "strand": ["+"]}, index=["a1"])

    # Fragment lengths are 60 (tf), 150 (mononucleosome), 50 (tf)
    m_tf = store.to_matrix(anchors, flank=100, bin_size=10, feature="midpoint", len_range="tf")
    m_nuc = store.to_matrix(anchors, flank=100, bin_size=10, feature="midpoint", len_range="mononucleosome")

    assert m_tf.shape == (1, 20)
    assert np.sum(m_tf.values) == 2.0
    assert np.sum(m_nuc.values) == 1.0
    assert list(m_tf.region_index) == ["a1"]

    # Duck-typed dict access & __array__
    assert m_tf["upstream"] == 100
    assert m_tf["downstream"] == 100
    assert m_tf["bin_size"] == 10
    np.testing.assert_array_equal(np.asarray(m_tf), m_tf.values)

    d = m_tf.to_dict()
    assert set(d.keys()) == {"values", "region_index", "upstream", "downstream", "bin_size"}

    # Normalizations
    m_cpm = m_tf.normalize("cpm")
    np.testing.assert_allclose(m_cpm.values, m_tf.values / (3 / 1e6))
    m_rpkm = m_tf.normalize("rpkm")
    np.testing.assert_allclose(m_rpkm.values, m_cpm.values / (10 / 1000.0))
    m_rs = m_tf.normalize("row_sum")
    assert pytest.approx(float(np.sum(m_rs.values[0])), rel=1e-9) == 1.0

    # mean_profile & to_dataframe
    prof = m_tf.mean_profile(ci=0.95)
    assert {"pos", "mean", "std", "sem", "ci_low", "ci_high"}.issubset(prof.columns)
    long_df = m_tf.to_dataframe()
    assert len(long_df) == 20


def test_cut_site_footprint_and_protection_score():
    store, anchors = _make_protected_motif_store()

    fp = cut_site_footprint(store, anchors, flank=40, bin_size=1, split_strands=True, agg="sum")
    assert fp.plus is not None and fp.minus is not None
    np.testing.assert_allclose(fp.plus + fp.minus, fp.combined)
    df = fp.to_dataframe()
    assert set(df["strand"].unique()) == {"+", "-"}

    scores = footprint_score(store, anchors, motif_half_width=10, flank_width=25, feature="cuts")
    assert list(scores.index) == ["motif_bound", "motif_unbound"]

    # Bound motif has zero center cuts and high flank cuts -> strongly positive protection_score, negative depth
    assert scores.loc["motif_bound", "center_cuts"] == 0.0
    assert scores.loc["motif_bound", "flank_cuts"] > 50.0
    assert scores.loc["motif_bound", "protection_score"] > 2.0
    assert scores.loc["motif_bound", "footprint_depth"] < -1.0

    # Bound motif must score substantially higher protection than flat unbound motif
    assert scores.loc["motif_bound", "protection_score"] > scores.loc["motif_unbound", "protection_score"]


def test_signal_matrix_normalize_is_not_compounding():
    from pyvplot.matrices import SignalMatrix

    raw = SignalMatrix(
        values=np.ones((2, 4)), region_index=np.array([0, 1]), upstream=20, downstream=20,
        bin_size=10, n_fragments=2_000_000,
    )
    cpm = raw.normalize("cpm")
    with pytest.raises(ValueError, match="already normalised"):
        cpm.normalize("rpkm")
    assert raw.raw_count_scale() == 1.0
    assert cpm.raw_count_scale() == pytest.approx(0.5)
    assert raw.normalize("rpkm").raw_count_scale() == pytest.approx(0.5 / 0.01)
    assert raw.normalize("zscore").raw_count_scale() is None


def test_footprint_score_pseudocount_is_scale_aware_and_bed_duplicates_are_unique(tmp_path):
    import pyvplot as vp

    store, anchors = _make_protected_motif_store()
    sc_raw = store.footprint_score(anchors, motif_half_width=10, flank_width=25)
    sc_cpm = store.footprint_score(anchors, motif_half_width=10, flank_width=25, normalize="cpm")
    sc_rpkm = store.footprint_score(anchors, motif_half_width=10, flank_width=25, normalize="rpkm")
    # Protection score is a log2 ratio of flank vs center density and must be invariant to CPM/RPKM scaling
    np.testing.assert_allclose(sc_cpm["protection_score"], sc_raw["protection_score"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sc_rpkm["protection_score"], sc_raw["protection_score"], rtol=1e-9, atol=1e-9)

    # BED file with identical column-4 TF name on every row gets unique anchor IDs
    bed = tmp_path / "ctcf.bed"
    bed.write_text("chr1\t1990\t2010\tCTCF\t100\t+\nchr1\t5990\t6010\tCTCF\t80\t-\n")
    sc_bed = store.footprint_score(bed, motif_half_width=10, flank_width=25)
    assert sc_bed.index.tolist() == ["CTCF_0", "CTCF_1"]
    mat_bed = store.to_matrix(bed, flank=50, bin_size=10)
    assert mat_bed.region_index.tolist() == ["CTCF_0", "CTCF_1"]
    assert mat_bed.to_dataframe()["anchor"].nunique() == 2
