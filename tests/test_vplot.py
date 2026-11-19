"""Tests for VPlot, normalizations, smoothing, differential V-plots, and VPlotCollection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvplot.store import FragmentStore
from pyvplot.vplot import VPlotCollection, vplot


def _make_tf_and_nucleosome_store(
    n_anchors: int = 20,
    tf_strength: int = 30,
    nuc_strength: int = 80,
    seed: int = 123,
) -> tuple[FragmentStore, pd.DataFrame]:
    """Generate synthetic TF-bound fragments (~60 bp at center) + abundant flanking nucleosomes (~150 bp)."""
    rng = np.random.default_rng(seed)
    centers = np.arange(5_000, 5_000 + n_anchors * 2_000, 2_000)
    anchors = pd.DataFrame(
        {
            "chrom": ["chr1"] * n_anchors,
            "center": centers,
            "strand": rng.choice(["+", "-"], size=n_anchors),
        }
    )

    starts: list[int] = []
    ends: list[int] = []
    for c in centers:
        # Short TF footprints centered near anchor (length ~50-70 bp)
        for _ in range(tf_strength):
            fl = int(rng.integers(50, 70))
            mid = int(c + rng.normal(0, 5))
            s = mid - fl // 2
            starts.append(s)
            ends.append(s + fl)
        # Background uniform 60 bp fragments across flanks
        for _ in range(10):
            fl = int(rng.integers(50, 70))
            mid = int(c + rng.integers(-290, 290))
            s = mid - fl // 2
            starts.append(s)
            ends.append(s + fl)
        # Abundant nucleosomes (145-160 bp) uniformly across window + phased at +/-130 bp
        for _ in range(nuc_strength):
            fl = int(rng.integers(145, 160))
            mid = int(c + rng.integers(-290, 290))
            s = mid - fl // 2
            starts.append(s)
            ends.append(s + fl)

    store = FragmentStore(
        starts={"chr1": np.asarray(starts, dtype=np.int32)},
        ends={"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )
    return store, anchors


def test_vplot_basic_and_normalizations():
    store, anchors = _make_tf_and_nucleosome_store(tf_strength=30, nuc_strength=100)
    vp = vplot(store, anchors, flank=300, min_len=20, max_len=220, pos_bin=5, len_bin=5)

    assert vp.shape == (40, 120)
    assert len(vp.pos_centers) == 120
    assert len(vp.len_centers) == 40
    assert vp.normalization == "raw"
    assert vp.is_diff is False

    # Check TF enrichment at center (pos near 0, len near 60)
    vp_cpm = vp.normalize("cpm")
    assert vp_cpm.normalization == "cpm"
    expected_scale = (store.n_fragments / 1e6) * len(anchors)
    np.testing.assert_allclose(vp_cpm.counts, vp.counts / expected_scale)

    vp_density = vp.normalize("density")
    assert pytest.approx(float(np.sum(vp_density.counts)), rel=1e-9) == 1.0

    # Expected/background normalization should elevate the 60 bp TF center well above 1.0
    # even though 150 bp nucleosomes are more abundant overall!
    vp_exp_flank = vp.normalize("expected", expected_from="flank")
    vp_exp_genome = vp.normalize("expected", expected_from="genome")
    tf_row = int(np.argmin(np.abs(vp.len_centers - 60)))
    nuc_row = int(np.argmin(np.abs(vp.len_centers - 152)))
    center_col = int(np.argmin(np.abs(vp.pos_centers - 0)))

    assert vp_exp_flank.counts[tf_row, center_col] > 3.0
    assert vp_exp_flank.counts[tf_row, center_col] > vp_exp_flank.counts[nuc_row, center_col]
    assert vp_exp_genome.counts[tf_row, center_col] > 3.0

    # log2_oe and row_zscore
    vp_oe = vp.normalize("log2_oe", expected_from="flank", pseudocount=0.1)
    assert vp_oe.counts[tf_row, center_col] > 1.5
    vp_z = vp.normalize("row_zscore")
    assert vp_z.counts[tf_row, center_col] > 2.0


def test_vplot_smoothing_and_differential():
    wt_store, anchors = _make_tf_and_nucleosome_store(tf_strength=40, nuc_strength=60, seed=10)
    # Mutant loses TF binding (tf_strength=0) while keeping nucleosomes
    mut_store, _ = _make_tf_and_nucleosome_store(tf_strength=0, nuc_strength=60, seed=11)

    vp_wt = vplot(wt_store, anchors, flank=200, min_len=20, max_len=200, pos_bin=4, len_bin=4)
    vp_mut = vplot(mut_store, anchors, flank=200, min_len=20, max_len=200, pos_bin=4, len_bin=4)

    # Smoothing preserves total mass approximately and reduces variance
    vp_sm = vp_wt.smooth(sigma=(1.5, 1.5))
    assert vp_sm.shape == vp_wt.shape
    assert np.var(vp_sm.counts) < np.var(vp_wt.counts)

    # Differential V-plot should show strong negative log2FC at the 60 bp TF center
    vp_diff = vp_mut.diff(vp_wt, mode="log2fc", pseudocount=0.5, smooth_sigma=(1.5, 1.5))
    assert vp_diff.is_diff is True
    assert vp_diff.normalization == "diff:log2fc"

    tf_row = int(np.argmin(np.abs(vp_diff.len_centers - 60)))
    center_col = int(np.argmin(np.abs(vp_diff.pos_centers - 0)))
    assert vp_diff.counts[tf_row, center_col] < -1.0

    # Other diff modes
    vp_sub = vp_mut.diff(vp_wt, mode="subtract")
    assert vp_sub.counts[tf_row, center_col] < 0.0
    vp_rat = vp_mut.diff(vp_wt, mode="ratio", pseudocount=0.5)
    assert vp_rat.counts[tf_row, center_col] < 1.0
    vp_z = vp_mut.diff(vp_wt, mode="zscore")
    assert abs(float(np.mean(vp_z.counts))) < 1e-6


def test_vplot_marginals_tidy_and_collection():
    store, anchors = _make_tf_and_nucleosome_store(n_anchors=10, tf_strength=20, nuc_strength=20)
    vp1 = store.vplot(anchors, flank=100, min_len=20, max_len=200, pos_bin=10, len_bin=10)
    vp2 = vp1.normalize("cpm")

    m_pos = vp1.marginal_position(len_range=(40, 80))
    assert list(m_pos.columns) == ["pos", "value"]
    assert len(m_pos) == 20
    # Peak of 40-80 bp marginal should be near 0
    peak_pos = float(m_pos.loc[m_pos["value"].idxmax(), "pos"])
    assert abs(peak_pos) <= 20.0

    m_len = vp1.marginal_length(pos_range=(-20, 20))
    assert list(m_len.columns) == ["length", "value"]
    assert len(m_len) == 18

    coll = VPlotCollection()
    coll.add(vp2, tf="CTCF", condition="WT")
    coll.add(vp2, tf="CTCF", condition="Mut")
    df = coll.to_dataframe()
    assert len(df) == 2 * vp2.counts.size
    assert {"pos", "length", "value", "tf", "condition"}.issubset(df.columns)
