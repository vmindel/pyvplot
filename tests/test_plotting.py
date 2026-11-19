"""Smoke and rendering tests for matplotlib and plotnine plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9

from pyvplot.plotting import (
    plot_cut_footprint_matplotlib,
    plot_cut_footprint_plotnine,
    plot_insert_size_matplotlib,
    plot_insert_size_plotnine,
    plot_phasogram_matplotlib,
    plot_phasogram_plotnine,
    plot_vplot_collection_plotnine,
    plot_vplot_matplotlib,
    plot_vplot_plotnine,
)
from pyvplot.store import FragmentStore
from pyvplot.vplot import VPlotCollection, vplot


def _make_plot_fixtures() -> tuple[FragmentStore, pd.DataFrame]:
    rng = np.random.default_rng(99)
    centers = np.array([1000, 3000, 5000])
    anchors = pd.DataFrame({"chrom": ["chr1"] * 3, "center": centers, "strand": ["+", "-", "+"]})
    starts: list[int] = []
    ends: list[int] = []
    for c in centers:
        for _ in range(40):
            fl = int(rng.integers(40, 180))
            mid = int(c + rng.normal(0, 40))
            starts.append(mid - fl // 2)
            ends.append(mid - fl // 2 + fl)
    store = FragmentStore(
        {"chr1": np.asarray(starts, dtype=np.int32)},
        {"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )
    return store, anchors


def test_matplotlib_renderers():
    store, anchors = _make_plot_fixtures()
    vp = vplot(store, anchors, flank=100, min_len=30, max_len=190, pos_bin=5, len_bin=5)
    vp_diff = vp.diff(vp, mode="log2fc", smooth_sigma=(1.0, 1.0))

    fig1, _ = plot_vplot_matplotlib(vp.normalize("cpm"), title="WT V-plot")
    fig2, _ = plot_vplot_matplotlib(vp_diff, title="Diff V-plot")
    fig3, _ = plot_insert_size_matplotlib(store, log_y=False)
    fig4, _ = plot_insert_size_matplotlib(store, log_y=True)
    ph = store.phasogram(max_lag=400)
    fig5, _ = plot_phasogram_matplotlib(ph)
    fp = store.cut_site_footprint(anchors, flank=50, bin_size=1)
    fig6, _ = plot_cut_footprint_matplotlib(fp)

    for fig in (fig1, fig2, fig3, fig4, fig5, fig6):
        fig.canvas.draw()
        plt.close(fig)


def test_plotnine_renderers():
    store, anchors = _make_plot_fixtures()
    vp = vplot(store, anchors, flank=80, min_len=40, max_len=160, pos_bin=10, len_bin=10)
    vp_norm = vp.normalize("cpm")
    vp_diff = vp.diff(vp, mode="log2fc")

    p1 = plot_vplot_plotnine(vp_norm, cmap="magma")
    p2 = plot_vplot_plotnine(vp_diff)
    coll = VPlotCollection().add(vp_norm, tf="TF1", condition="WT").add(vp_norm, tf="TF1", condition="Mut")
    p3 = plot_vplot_collection_plotnine(coll, facet="tf ~ condition")
    p4 = plot_insert_size_plotnine(store, log_y=True)
    p5 = plot_phasogram_plotnine(store.phasogram(max_lag=300))
    p6 = plot_cut_footprint_plotnine(store.cut_site_footprint(anchors, flank=40, bin_size=2))

    for p in (p1, p2, p3, p4, p5, p6):
        assert isinstance(p, p9.ggplot)
        fig = p.draw()
        plt.close(fig)
