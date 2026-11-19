"""Publication-ready matplotlib and plotnine helpers for V-plots, cut footprints, and QC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyvplot.matrices import CutSiteFootprint
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
    """Plot 1-bp cleavage cut-site footprint (strand-separated or combined) with ``matplotlib``."""
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if footprint.plus is not None and footprint.minus is not None:
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
        p9.ggplot(df, p9.aes(x="pos", y="signal", color="strand"))
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
