"""matplotlib and plotnine helpers for V-plots, per-anchor heatmaps, cut footprints and QC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyvplot.matrices import CutSiteFootprint, SignalMatrix
    from pyvplot.qc import PhasogramResult
    from pyvplot.store import FragmentStore
    from pyvplot.vplot import VPlot, VPlotCollection


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib plotting requires `matplotlib`. Install it with `pip install pyvplot[plot]`."
        ) from exc
    return plt


def _require_plotnine() -> Any:
    try:
        import plotnine as p9
    except ImportError as exc:
        raise ImportError(
            "plotnine plotting requires `plotnine`. Install it with `pip install pyvplot[plot]`."
        ) from exc
    return p9


def _auto_vplot_limits(
    counts: np.ndarray,
    is_diff: bool,
    vmin: float | None,
    vmax: float | None,
    quantile_clip: float = 0.995,
) -> tuple[float, float]:
    finite = counts[np.isfinite(counts)]
    if len(finite) == 0:
        return (-1.0, 1.0) if is_diff else (0.0, 1.0)
    if is_diff:
        bound = float(np.quantile(np.abs(finite), quantile_clip))
        if bound == 0.0:
            bound = float(np.max(np.abs(finite))) or 1.0
        lo = -bound if vmin is None else float(vmin)
        hi = bound if vmax is None else float(vmax)
        return lo, hi
    lo = float(np.min(finite)) if vmin is None else float(vmin)
    hi = float(np.quantile(finite, quantile_clip)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def plot_vplot_matplotlib(
    vp: VPlot,
    ax: Any = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    quantile_clip: float = 0.995,
    title: str | None = None,
    xlabel: str = "Distance from anchor center (bp)",
    ylabel: str = "Fragment length (bp)",
    cbar_label: str | None = None,
    colorbar: bool = True,
    center_line: bool = True,
    figsize: tuple[float, float] = (6.0, 4.5),
) -> tuple[Any, Any]:
    """Render a [`VPlot`][pyvplot.VPlot] (or differential V-plot) using ``matplotlib``.

    Returns ``(fig, ax)``.
    """
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    use_cmap = cmap if cmap is not None else ("RdBu_r" if vp.is_diff else "magma")
    lo, hi = _auto_vplot_limits(vp.counts, vp.is_diff, vmin, vmax, quantile_clip=quantile_clip)

    extent = [
        float(vp.pos_edges[0]),
        float(vp.pos_edges[-1]),
        float(vp.len_edges[0]),
        float(vp.len_edges[-1]),
    ]
    im = ax.imshow(
        vp.counts,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=use_cmap,
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    if center_line:
        ax.axvline(0.0, color="white" if not vp.is_diff else "black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if colorbar:
        label = cbar_label if cbar_label is not None else vp.normalization
        fig.colorbar(im, ax=ax, label=label)

    return fig, ax


def plot_vplot_plotnine(
    vp: VPlot,
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
    quantile_clip: float = 0.995,
    title: str | None = None,
    xlabel: str = "Distance from anchor center (bp)",
    ylabel: str = "Fragment length (bp)",
    fill_label: str | None = None,
    center_line: bool = True,
) -> Any:
    """Render a [`VPlot`][pyvplot.VPlot] as a customizable ``plotnine`` ``ggplot`` object."""
    p9 = _require_plotnine()
    df = vp.to_dataframe()
    lo, hi = (
        limits
        if limits is not None
        else _auto_vplot_limits(vp.counts, vp.is_diff, None, None, quantile_clip=quantile_clip)
    )
    df["value_clipped"] = np.clip(df["value"].to_numpy(dtype=np.float64), lo, hi)

    fill_name = fill_label if fill_label is not None else vp.normalization
    p = (
        p9.ggplot(df, p9.aes(x="pos", y="length", fill="value_clipped"))
        + p9.geom_raster()
        + p9.labs(x=xlabel, y=ylabel, fill=fill_name, title=title or "")
        + p9.theme_minimal()
    )

    if vp.is_diff and cmap is None:
        p = p + p9.scale_fill_gradient2(
            low="#2166ac",
            mid="#f7f7f7",
            high="#b2182b",
            midpoint=0.0,
            limits=(lo, hi),
        )
    else:
        use_cmap = cmap if cmap is not None else "magma"
        p = p + p9.scale_fill_cmap(cmap_name=use_cmap, limits=(lo, hi))

    if center_line:
        p = p + p9.geom_vline(
            xintercept=0,
            linetype="dashed",
            color="black" if vp.is_diff else "white",
            alpha=0.6,
            size=0.5,
        )

    return p


def plot_vplot_collection_plotnine(
    collection: VPlotCollection,
    facet: str | None = None,
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
    quantile_clip: float = 0.995,
    xlabel: str = "Distance from anchor center (bp)",
    ylabel: str = "Fragment length (bp)",
    fill_label: str = "Signal",
    center_line: bool = True,
) -> Any:
    """Render a [`VPlotCollection`][pyvplot.VPlotCollection] as a faceted ``plotnine`` plot."""
    p9 = _require_plotnine()
    df = collection.to_dataframe()
    if len(df) == 0:
        raise ValueError("Cannot plot an empty VPlotCollection.")

    is_diff = all(vp.is_diff for vp, _ in collection.items)
    vals = df["value"].to_numpy(dtype=np.float64)
    lo, hi = (
        limits
        if limits is not None
        else _auto_vplot_limits(vals, is_diff, None, None, quantile_clip=quantile_clip)
    )
    df["value_clipped"] = np.clip(vals, lo, hi)

    p = (
        p9.ggplot(df, p9.aes(x="pos", y="length", fill="value_clipped"))
        + p9.geom_raster()
        + p9.labs(x=xlabel, y=ylabel, fill=fill_label)
        + p9.theme_minimal()
    )

    if is_diff and cmap is None:
        p = p + p9.scale_fill_gradient2(
            low="#2166ac",
            mid="#f7f7f7",
            high="#b2182b",
            midpoint=0.0,
            limits=(lo, hi),
        )
    else:
        p = p + p9.scale_fill_cmap(cmap_name=cmap or "magma", limits=(lo, hi))

    if center_line:
        p = p + p9.geom_vline(xintercept=0, linetype="dashed", color="white", alpha=0.5, size=0.4)

    if facet is not None:
        p = p + (p9.facet_grid(facet) if "~" in facet else p9.facet_wrap(f"~ {facet}"))
    else:
        meta_cols = [c for c in df.columns if c not in {"pos", "length", "value", "value_clipped"}]
        if len(meta_cols) == 1:
            p = p + p9.facet_wrap(f"~ {meta_cols[0]}")
        elif len(meta_cols) >= 2:
            p = p + p9.facet_grid(f"{meta_cols[0]} ~ {meta_cols[1]}")

    return p


def plot_insert_size_matplotlib(
    store: FragmentStore,
    ax: Any = None,
    min_len: int | None = None,
    max_len: int | None = None,
    log_y: bool = False,
    show_bands: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> tuple[Any, Any]:
    """Plot fragment insert-size distribution with optional assay size-band shading."""
    plt = _require_matplotlib()
    df = store.insert_size_distribution(min_len=min_len, max_len=max_len, normalize=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if len(df) > 0:
        y_vals = df["fraction"].to_numpy(dtype=np.float64)
        ax.plot(df["length"], y_vals, color="#1f4e79", linewidth=1.4)
        ax.fill_between(df["length"], 0.0, y_vals, color="#1f4e79", alpha=0.18)

    if show_bands and store.assay.size_bins:
        palette = ["#2b8cbe", "#fdae6b", "#74c476", "#9e9ac8"]
        for i, (band_name, (lo, hi)) in enumerate(store.assay.size_bins.items()):
            color = palette[i % len(palette)]
            ax.axvspan(lo, hi, color=color, alpha=0.15, label=f"{band_name} ({lo}-{hi} bp)")
        ax.legend(frameon=False, fontsize=8)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Fraction of fragments")
    ax.set_title(title if title is not None else f"Insert-size distribution ({store.assay.name})")
    return fig, ax


def plot_insert_size_plotnine(
    store: FragmentStore,
    min_len: int | None = None,
    max_len: int | None = None,
    log_y: bool = False,
    title: str | None = None,
) -> Any:
    """Plot fragment insert-size distribution with ``plotnine``."""
    p9 = _require_plotnine()
    df = store.insert_size_distribution(min_len=min_len, max_len=max_len, normalize=True)
    if log_y:
        df = df[df["fraction"] > 0].copy()
    p = (
        p9.ggplot(df, p9.aes(x="length", y="fraction"))
        + p9.geom_line(color="#1f4e79", size=0.9)
        + p9.labs(
            x="Fragment length (bp)",
            y="Fraction of fragments",
            title=title if title is not None else f"Insert-size distribution ({store.assay.name})",
        )
        + p9.theme_minimal()
    )
    if log_y:
        p = p + p9.scale_y_log10()
    return p


def plot_phasogram_matplotlib(
    result: PhasogramResult,
    ax: Any = None,
    show_raw: bool = True,
    show_peaks: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> tuple[Any, Any]:
    """Plot a nucleosome phasogram and detected NRL peaks with ``matplotlib``."""
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if show_raw:
        ax.plot(result.lags, result.counts, color="#cccccc", linewidth=0.7, label="Raw")
    ax.plot(result.lags, result.smoothed, color="#2c3e50", linewidth=1.6, label="Smoothed")

    if show_peaks and len(result.peak_positions) > 0:
        for idx, pk in enumerate(result.peak_positions):
            ax.axvline(
                float(pk),
                color="#d95f02",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="Detected peaks" if idx == 0 else None,
            )

    default_title = (
        f"Nucleosome Phasogram (NRL = {result.nrl:.1f} bp)"
        if result.nrl is not None
        else "Nucleosome Phasogram"
    )
    ax.set_title(title if title is not None else default_title)
    ax.set_xlabel("Pair distance lag (bp)")
    ax.set_ylabel("Pair count")
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def plot_phasogram_plotnine(
    result: PhasogramResult,
    title: str | None = None,
) -> Any:
    """Plot a nucleosome phasogram with ``plotnine``."""
    p9 = _require_plotnine()
    df = result.to_dataframe()
    default_title = (
        f"Nucleosome Phasogram (NRL = {result.nrl:.1f} bp)"
        if result.nrl is not None
        else "Nucleosome Phasogram"
    )
    p = (
        p9.ggplot(df, p9.aes(x="lag", y="smoothed"))
        + p9.geom_line(color="#2c3e50", size=0.9)
        + p9.labs(
            x="Pair distance lag (bp)",
            y="Smoothed pair count",
            title=title if title is not None else default_title,
        )
        + p9.theme_minimal()
    )
    if len(result.peak_positions) > 0:
        for pk in result.peak_positions:
            p = p + p9.geom_vline(xintercept=float(pk), linetype="dashed", color="#d95f02", alpha=0.8)
    return p


def plot_cut_footprint_matplotlib(
    footprint: CutSiteFootprint,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 3.8),
) -> tuple[Any, Any]:
    """Plot 1-bp cleavage cut-site footprint (split by end, or combined) with ``matplotlib``."""
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if footprint.proximal is not None and footprint.distal is not None:
        ax.plot(footprint.pos, footprint.proximal, color="#1b7837", linewidth=1.2, label="Proximal cuts")
        ax.plot(footprint.pos, footprint.distal, color="#762a83", linewidth=1.2, label="Distal cuts")
        ax.legend(frameon=False, fontsize=8)
    elif footprint.plus is not None and footprint.minus is not None:
        ax.plot(footprint.pos, footprint.plus, color="#d73027", linewidth=1.2, label="+ strand cuts")
        ax.plot(footprint.pos, footprint.minus, color="#4575b4", linewidth=1.2, label="- strand cuts")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.plot(footprint.pos, footprint.combined, color="#2c3e50", linewidth=1.3, label="All cuts")

    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Distance from anchor center (bp)")
    ax.set_ylabel(f"Cut signal ({footprint.agg})")
    ax.set_title(title if title is not None else "Cleavage Cut-Site Footprint")
    return fig, ax


def plot_cut_footprint_plotnine(
    footprint: CutSiteFootprint,
    title: str | None = None,
) -> Any:
    """Plot 1-bp cleavage cut-site footprint with ``plotnine``."""
    p9 = _require_plotnine()
    df = footprint.to_dataframe()
    p = (
        p9.ggplot(df, p9.aes(x="pos", y="signal", color="end" if "end" in df.columns else "strand"))
        + p9.geom_line(size=0.8)
        + p9.geom_vline(xintercept=0, linetype="dashed", color="#666666", alpha=0.6)
        + p9.labs(
            x="Distance from anchor center (bp)",
            y=f"Cut signal ({footprint.agg})",
            title=title if title is not None else "Cleavage Cut-Site Footprint",
        )
        + p9.theme_minimal()
    )
    return p


def _row_order(values: np.ndarray, order: Any) -> np.ndarray:
    n = values.shape[0]
    if order is None:
        return np.arange(n)
    if isinstance(order, str):
        if order not in {"mean", "max"}:
            raise ValueError(f"Unknown order {order!r}; expected 'mean', 'max', None, or row positions.")
        key = values.mean(axis=1) if order == "mean" else values.max(axis=1)
        return np.argsort(-key, kind="stable")
    idx = np.asarray(order)
    if idx.dtype == bool or idx.ndim != 1:
        raise ValueError("`order` must be 'mean', 'max', None, or a 1D array of row positions.")
    return idx.astype(np.int64)


def _heatmap_limits(values: np.ndarray, vmin: float | None, vmax: float | None, quantile_clip: float) -> tuple[float, float, bool]:
    finite = values[np.isfinite(values)]
    diverging = finite.size > 0 and float(finite.min()) < 0.0
    if diverging:
        lim = float(np.quantile(np.abs(finite), quantile_clip)) if finite.size else 1.0
        lim = lim if lim > 0 else 1.0
        return (-lim if vmin is None else vmin), (lim if vmax is None else vmax), True
    hi = float(np.quantile(finite, quantile_clip)) if finite.size else 1.0
    if hi <= 0.0:
        hi = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    return (0.0 if vmin is None else vmin), (hi if vmax is None else vmax), False


def plot_signal_matrix_matplotlib(
    mat: SignalMatrix,
    order: str | Any | None = "mean",
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    quantile_clip: float = 0.99,
    profile: bool = True,
    ci: float | None = 0.95,
    title: str | None = None,
    xlabel: str = "Distance from anchor center (bp)",
    cbar_label: str | None = None,
    figsize: tuple[float, float] = (4.0, 6.0),
) -> tuple[Any, tuple[Any, Any]]:
    """Plot a [`SignalMatrix`][pyvplot.SignalMatrix] as a heatmap, with its mean profile on top.

    Parameters
    ----------
    mat:
        Matrix to plot; one heatmap row per anchor.
    order:
        Row order, top to bottom: ``"mean"`` (default, highest row mean first),
        ``"max"``, ``None`` (input order), or an array of row positions, e.g.
        ``np.argsort(-scores["protection_score"].to_numpy())``. Positions may
        select a subset of rows.
    cmap:
        Colour map. Defaults to ``"magma"``, or ``"RdBu_r"`` centred on 0 when
        the matrix has negative values (z-scores, log ratios).
    vmin, vmax:
        Colour limits. By default ``0`` to the ``quantile_clip`` quantile, so a
        few very deep anchors do not wash out the rest.
    quantile_clip:
        Quantile used for the default upper limit.
    profile:
        Draw the mean profile (over the plotted rows) above the heatmap.
    ci:
        Confidence level of the shaded band around the profile; ``None`` for no band.
    title, xlabel, cbar_label:
        Labels; ``cbar_label`` defaults to the matrix's normalisation.
    figsize:
        Figure size in inches.

    Returns
    -------
    (fig, (ax_profile, ax_heatmap))
        ``ax_profile`` is ``None`` when ``profile=False``.
    """
    plt = _require_matplotlib()
    rows = _row_order(mat.values, order)
    vals = mat.values[rows]
    lo, hi, diverging = _heatmap_limits(vals, vmin, vmax, quantile_clip)
    use_cmap = cmap if cmap is not None else ("RdBu_r" if diverging else "magma")

    fig = plt.figure(figsize=figsize)
    if profile:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 0.05], hspace=0.05, wspace=0.05)
        ax_prof = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_prof)
        cax = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.05)
        ax_prof = None
        ax_heat = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

    n = vals.shape[0]
    im = ax_heat.imshow(
        vals,
        aspect="auto",
        cmap=use_cmap,
        vmin=lo,
        vmax=hi,
        extent=[-mat.upstream, mat.downstream, n, 0],
    )
    ax_heat.axvline(0.0, color="white" if not diverging else "black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_heat.set_xlabel(xlabel)
    ax_heat.set_ylabel(f"{n:,} anchors")
    ax_heat.set_yticks([])
    fig.colorbar(im, cax=cax, label=cbar_label if cbar_label is not None else mat.normalization)

    if ax_prof is not None:
        x = mat.bin_centers
        mean = vals.mean(axis=0) if n else np.zeros(mat.n_bins)
        ax_prof.plot(x, mean, color="#2c3e50", linewidth=1.2)
        if ci is not None and n > 1:
            z = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}.get(round(ci, 2))
            if z is None:
                from scipy.stats import norm as _norm

                z = float(_norm.ppf(0.5 + 0.5 * ci))
            sem = vals.std(axis=0, ddof=1) / np.sqrt(n)
            ax_prof.fill_between(x, mean - z * sem, mean + z * sem, color="#2c3e50", alpha=0.2, linewidth=0)
        ax_prof.axvline(0.0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_prof.set_ylabel("Mean")
        ax_prof.tick_params(labelbottom=False)
        ax_prof.spines[["top", "right"]].set_visible(False)

    top = ax_prof if ax_prof is not None else ax_heat
    lr = "all lengths" if mat.len_range is None else f"{mat.len_range[0]}-{mat.len_range[1]} bp"
    top.set_title(title if title is not None else f"{mat.feature}, {lr}")
    return fig, (ax_prof, ax_heat)


def plot_signal_matrix_plotnine(
    mat: SignalMatrix,
    order: str | Any | None = "mean",
    cmap: str | None = None,
    limits: tuple[float, float] | None = None,
    quantile_clip: float = 0.99,
    title: str | None = None,
    xlabel: str = "Distance from anchor center (bp)",
    fill_label: str | None = None,
) -> Any:
    """Plot a [`SignalMatrix`][pyvplot.SignalMatrix] as a ``plotnine`` heatmap (``geom_raster``).

    Rows are ordered as in [`plot_signal_matrix_matplotlib`][pyvplot.plotting.plot_signal_matrix_matplotlib].
    For the mean profile, plot ``mat.mean_profile(ci=0.95)`` separately. Each
    cell is one data-frame row, so for matrices above a few hundred thousand
    cells the matplotlib version is much faster.
    """
    p9 = _require_plotnine()
    import pandas as pd

    rows = _row_order(mat.values, order)
    vals = mat.values[rows]
    lo, hi, diverging = _heatmap_limits(vals, None, None, quantile_clip)
    lims = limits if limits is not None else (lo, hi)
    n_rows, n_bins = vals.shape
    df = pd.DataFrame(
        {
            "pos": np.tile(mat.bin_centers, n_rows),
            "row": np.repeat(np.arange(n_rows), n_bins),
            "value": np.clip(vals.ravel(), lims[0], lims[1]),
        }
    )
    fill = (
        p9.scale_fill_gradient2(low="#2166ac", mid="white", high="#b2182b", midpoint=0, limits=lims)
        if (cmap is None and diverging)
        else p9.scale_fill_cmap(cmap_name=cmap or "magma", limits=lims)
    )
    lr = "all lengths" if mat.len_range is None else f"{mat.len_range[0]}-{mat.len_range[1]} bp"
    return (
        p9.ggplot(df, p9.aes(x="pos", y="row", fill="value"))
        + p9.geom_raster()
        + p9.geom_vline(xintercept=0, linetype="dashed", color="white", alpha=0.6)
        + fill
        + p9.scale_y_reverse(expand=(0, 0))
        + p9.scale_x_continuous(expand=(0, 0))
        + p9.labs(
            x=xlabel,
            y=f"{n_rows:,} anchors",
            fill=fill_label if fill_label is not None else mat.normalization,
            title=title if title is not None else f"{mat.feature}, {lr}",
        )
        + p9.theme_minimal()
        + p9.theme(axis_text_y=p9.element_blank(), panel_grid=p9.element_blank())
    )
