"""V-plot 2D fragment geometry histograms, normalizations, smoothing, and differential V-plots."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.assays import resolve_len_range
from pyvplot.extract import accumulate_vplot
from pyvplot.store import FragmentStore


def _gaussian_filter_2d(
    arr: np.ndarray,
    sigma: float | tuple[float, float],
    mode: str = "nearest",
) -> np.ndarray:
    """Apply 2D Gaussian smoothing via SciPy, with a pure-NumPy separable fallback."""
    if isinstance(sigma, (int, float)):
        sy = sx = float(sigma)
    else:
        sy, sx = float(sigma[0]), float(sigma[1])

    if sy <= 0.0 and sx <= 0.0:
        return arr.astype(np.float64, copy=True)

    try:
        from scipy.ndimage import gaussian_filter

        return np.asarray(gaussian_filter(arr.astype(np.float64, copy=False), sigma=(sy, sx), mode=mode), dtype=np.float64)
    except ImportError:
        out = arr.astype(np.float64, copy=True)
        for axis, s in ((0, sy), (1, sx)):
            if s <= 0.0:
                continue
            radius = max(1, int(np.ceil(4.0 * s)))
            x = np.arange(-radius, radius + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (x / s) ** 2)
            kernel /= kernel.sum()
            pad_mode = "edge" if mode == "nearest" else "reflect"
            pad_width = [(radius, radius), (0, 0)] if axis == 0 else [(0, 0), (radius, radius)]
            padded = np.pad(out, pad_width, mode=pad_mode)
            if axis == 0:
                out = np.apply_along_axis(lambda v, k=kernel: np.convolve(v, k, mode="valid"), 0, padded)
            else:
                out = np.apply_along_axis(lambda v, k=kernel: np.convolve(v, k, mode="valid"), 1, padded)
        return out


@dataclass(repr=False)
class VPlot:
    """2D fragment-length vs. relative-position histogram around genomic anchors.

    Attributes
    ----------
    counts:
        2D ``float64`` array of shape ``(n_len, n_pos)``. Row 0 corresponds to
        the shortest fragment length bin; column 0 corresponds to the leftmost
        (most upstream) position bin.
    pos_edges:
        1D array of length ``n_pos + 1`` giving bin edges along the relative
        position axis (bp).
    len_edges:
        1D array of length ``n_len + 1`` giving bin edges along the fragment
        length axis (bp).
    n_anchors:
        Number of anchors aggregated into this V-plot.
    n_fragments:
        Total number of fragments in the source :class:`~pyvplot.store.FragmentStore`
        (used for CPM normalization).
    assay:
        Name of the assay preset (e.g., ``"chec"``, ``"atac"``).
    feature:
        Geometric feature accumulated (``"midpoint"``, ``"cuts"``, etc.).
    normalization:
        Name of normalization applied (``"raw"``, ``"cpm"``, ``"expected"``,
        ``"log2fc"``, etc.).
    is_diff:
        Whether this object represents a differential V-plot (e.g., mutant vs. WT).
    genome_len_counts:
        Optional 1D array of length ``n_len`` giving genome-wide fragment counts
        per length bin from the source store (enables ``expected_from="genome"``).
    metadata:
        Arbitrary metadata dictionary.
    """

    counts: np.ndarray
    pos_edges: np.ndarray
    len_edges: np.ndarray
    n_anchors: int
    n_fragments: int
    assay: str = "generic"
    feature: str = "midpoint"
    normalization: str = "raw"
    is_diff: bool = False
    genome_len_counts: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"VPlot(pos=[{self.pos_edges[0]:g}, {self.pos_edges[-1]:g}) bp step {self.pos_bin:g}, "
            f"length=[{self.len_edges[0]:g}, {self.len_edges[-1]:g}) bp step {self.len_bin:g}, "
            f"shape={self.shape}, n_anchors={self.n_anchors}, feature={self.feature!r}, "
            f"normalization={self.normalization!r}, assay={self.assay!r})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.counts.shape[0]), int(self.counts.shape[1]))

    @property
    def pos_centers(self) -> np.ndarray:
        """Bin centers along the relative position axis (bp)."""
        return 0.5 * (self.pos_edges[:-1] + self.pos_edges[1:])

    @property
    def len_centers(self) -> np.ndarray:
        """Bin centers along the fragment length axis (bp)."""
        return 0.5 * (self.len_edges[:-1] + self.len_edges[1:])

    @property
    def pos_bin(self) -> float:
        return float(self.pos_edges[1] - self.pos_edges[0])

    @property
    def len_bin(self) -> float:
        return float(self.len_edges[1] - self.len_edges[0])

    def _clone(
        self,
        counts: np.ndarray,
        normalization: str | None = None,
        is_diff: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VPlot:
        return VPlot(
            counts=np.ascontiguousarray(counts, dtype=np.float64),
            pos_edges=self.pos_edges.copy(),
            len_edges=self.len_edges.copy(),
            n_anchors=self.n_anchors,
            n_fragments=self.n_fragments,
            assay=self.assay,
            feature=self.feature,
            normalization=self.normalization if normalization is None else normalization,
            is_diff=self.is_diff if is_diff is None else is_diff,
            genome_len_counts=None if self.genome_len_counts is None else self.genome_len_counts.copy(),
            metadata=dict(self.metadata) if metadata is None else metadata,
        )

    def _expected_row_rates(
        self,
        expected_from: str = "flank",
        flank_frac: float = 0.2,
    ) -> np.ndarray:
        """Compute expected per-bin background level for each fragment-length row."""
        n_len, n_pos = self.shape
        if expected_from == "flank":
            if not (0.0 < flank_frac <= 0.5):
                raise ValueError("`flank_frac` must be in (0, 0.5].")
            edge_bins = max(1, int(round(n_pos * flank_frac)))
            flank_vals = np.concatenate(
                [self.counts[:, :edge_bins], self.counts[:, n_pos - edge_bins :]],
                axis=1,
            )
            return np.mean(flank_vals, axis=1)

        if expected_from == "row_mean":
            return np.mean(self.counts, axis=1)

        if expected_from == "genome":
            if self.genome_len_counts is None or len(self.genome_len_counts) != n_len:
                raise ValueError(
                    "Genome-wide length counts are unavailable on this VPlot. "
                    "Use expected_from='flank' or 'row_mean'."
                )
            total_g = float(np.sum(self.genome_len_counts))
            if total_g <= 0.0:
                return np.zeros(n_len, dtype=np.float64)
            row_probs = self.genome_len_counts / total_g
            total_obs = float(np.sum(self.counts))
            return (row_probs * total_obs) / float(n_pos)

        raise ValueError(
            f"Unknown expected_from {expected_from!r}; expected 'flank', 'genome', or 'row_mean'."
        )

    def normalize(
        self,
        method: str = "cpm",
        expected_from: str = "flank",
        flank_frac: float = 0.2,
        pseudocount: float = 1e-6,
    ) -> VPlot:
        """Return a normalized copy of this V-plot.

        Parameters
        ----------
        method:
            Normalization strategy:
            - ``"cpm"``: counts per million fragments per anchor.
            - ``"cpm_total"``: counts per million total fragments (summed across anchors).
            - ``"per_anchor"``: mean counts per anchor.
            - ``"density"``: normalize grid so total sum equals 1.
            - ``"expected"``: divide each fragment-length row by its expected
              background rate (``expected_from``), preventing abundant
              nucleosomal fragments from drowning out short TF V-signatures.
            - ``"log2_oe"``: ``log2((observed + pseudocount) / (expected + pseudocount))``.
            - ``"row_zscore"``: z-score each fragment-length row across positions.
        expected_from:
            Background estimator for ``"expected"`` / ``"log2_oe"``:
            ``"flank"`` (outer window edges), ``"genome"`` (store-wide insert-size
            frequency), or ``"row_mean"``.
        flank_frac:
            Fraction of positions at each end used when ``expected_from="flank"``.
        pseudocount:
            Small constant preventing division by zero in ``"expected"`` and ``"log2_oe"``.
        """
        m = method.lower()
        arr = self.counts

        if m == "cpm":
            frag_scale = float(self.n_fragments) / 1e6 if self.n_fragments > 0 else 1.0
            anchor_scale = float(self.n_anchors) if self.n_anchors > 0 else 1.0
            return self._clone(arr / (frag_scale * anchor_scale), normalization="cpm")

        if m == "cpm_total":
            frag_scale = float(self.n_fragments) / 1e6 if self.n_fragments > 0 else 1.0
            return self._clone(arr / frag_scale, normalization="cpm_total")

        if m == "per_anchor":
            anchor_scale = float(self.n_anchors) if self.n_anchors > 0 else 1.0
            return self._clone(arr / anchor_scale, normalization="per_anchor")

        if m == "density":
            total = float(np.sum(arr))
            out = arr / total if total > 0.0 else np.zeros_like(arr)
            return self._clone(out, normalization="density")

        if m in {"expected", "log2_oe"}:
            exp = self._expected_row_rates(expected_from=expected_from, flank_frac=flank_frac)
            starved = (exp == 0.0) & (np.sum(arr, axis=1) > 0.0)
            if np.any(starved):
                warnings.warn(
                    f"{int(starved.sum())} of {len(exp)} length rows have counts but zero expected "
                    f"background (expected_from={expected_from!r}); they are divided by "
                    f"pseudocount={pseudocount:g} and will dominate the plot. Use a wider flank, "
                    "coarser len_bin, more anchors, or expected_from='genome'.",
                    UserWarning,
                    stacklevel=2,
                )

        if m == "expected":
            out = arr / (exp[:, None] + pseudocount)
            # Rows with zero expected AND zero observed remain 0
            zero_rows = (exp == 0.0) & (np.sum(arr, axis=1) == 0.0)
            out[zero_rows, :] = 0.0
            return self._clone(out, normalization=f"expected:{expected_from}")

        if m == "log2_oe":
            out = np.log2((arr + pseudocount) / (exp[:, None] + pseudocount))
            return self._clone(out, normalization=f"log2_oe:{expected_from}")

        if m == "row_zscore":
            mu = np.mean(arr, axis=1, keepdims=True)
            sd = np.std(arr, axis=1, keepdims=True)
            out = np.where(sd > 0.0, (arr - mu) / np.where(sd > 0.0, sd, 1.0), 0.0)
            return self._clone(out, normalization="row_zscore")

        raise ValueError(
            f"Unknown normalization method {method!r}. Expected one of: "
            "'cpm', 'cpm_total', 'per_anchor', 'density', 'expected', 'log2_oe', 'row_zscore'."
        )

    def raw_count_scale(self) -> float | None:
        """Value of one raw count in this V-plot's current units, or ``None`` if unknown.

        Known for ``raw``, ``cpm``, ``cpm_total`` and ``per_anchor`` (and smoothed
        versions of them); ``None`` for shape-changing normalisations such as
        ``expected`` or ``density``.
        """
        n_frag = float(self.n_fragments) / 1e6 if self.n_fragments > 0 else 1.0
        n_anc = float(self.n_anchors) if self.n_anchors > 0 else 1.0
        return {
            "raw": 1.0,
            "cpm": 1.0 / (n_frag * n_anc),
            "cpm_total": 1.0 / n_frag,
            "per_anchor": 1.0 / n_anc,
        }.get(self.normalization)

    def smooth(
        self,
        sigma: float | tuple[float, float] = (1.5, 1.5),
        mode: str = "nearest",
    ) -> VPlot:
        """Apply 2D Gaussian smoothing (``sigma=(sigma_len, sigma_pos)`` in bins)."""
        smoothed = _gaussian_filter_2d(self.counts, sigma=sigma, mode=mode)
        meta = dict(self.metadata)
        meta["smooth_sigma"] = sigma
        return self._clone(smoothed, metadata=meta)

    def diff(
        self,
        other: VPlot,
        mode: str = "log2fc",
        pseudocount: float | None = None,
        smooth_sigma: float | tuple[float, float] | None = None,
        normalize_first: str | None = "cpm",
        smooth_before: bool = True,
    ) -> VPlot:
        """Compute a differential V-plot comparing ``self`` (e.g., mutant) vs. ``other`` (e.g., WT).

        Parameters
        ----------
        other:
            Reference :class:`VPlot` with identical position and length axes.
        mode:
            Comparison metric: ``"log2fc"``, ``"subtract"``, ``"ratio"``, or ``"zscore"``.
        pseudocount:
            Added to numerator and denominator for ``"log2fc"`` and ``"ratio"``.
            ``None`` (default) adds half a raw count per cell, converted into
            each V-plot's own units (see :meth:`raw_count_scale`), so the result
            does not shrink towards 0 as sequencing depth grows. A number is
            added as-is, in the V-plots' current units. For normalisations
            without a known count scale (``expected``, ``density``, ...) the
            default falls back to ``0.5`` in those units.
        smooth_sigma:
            Optional 2D Gaussian smoothing ``(sigma_len, sigma_pos)``.
        normalize_first:
            If both V-plots are ``"raw"`` and ``normalize_first`` is not ``None``,
            normalize both using this method before differencing (default ``"cpm"``).
        smooth_before:
            If ``True`` (default), smooth both V-plots before computing the
            differential; if ``False``, smooth the resulting difference map.
        """
        if not (
            np.allclose(self.pos_edges, other.pos_edges)
            and np.allclose(self.len_edges, other.len_edges)
        ):
            raise ValueError("Cannot diff VPlots with different position or length bins.")

        a = self
        b = other
        if normalize_first is not None and a.normalization == "raw" and b.normalization == "raw":
            a = a.normalize(normalize_first)
            b = b.normalize(normalize_first)

        if smooth_sigma is not None and smooth_before:
            a = a.smooth(smooth_sigma)
            b = b.smooth(smooth_sigma)

        if pseudocount is None:
            sa, sb = a.raw_count_scale(), b.raw_count_scale()
            pc_a = 0.5 * sa if sa is not None else 0.5
            pc_b = 0.5 * sb if sb is not None else 0.5
        else:
            pc_a = pc_b = float(pseudocount)

        m = mode.lower()
        if m == "log2fc":
            vals = np.log2((a.counts + pc_a) / (b.counts + pc_b))
        elif m == "subtract":
            vals = a.counts - b.counts
        elif m == "ratio":
            vals = (a.counts + pc_a) / (b.counts + pc_b)
        elif m == "zscore":
            delta = a.counts - b.counts
            sd = float(np.std(delta))
            vals = (delta - float(np.mean(delta))) / sd if sd > 0.0 else np.zeros_like(delta)
        else:
            raise ValueError(
                f"Unknown diff mode {mode!r}; expected 'log2fc', 'subtract', 'ratio', or 'zscore'."
            )

        if smooth_sigma is not None and not smooth_before:
            vals = _gaussian_filter_2d(vals, sigma=smooth_sigma)

        return VPlot(
            counts=np.ascontiguousarray(vals, dtype=np.float64),
            pos_edges=self.pos_edges.copy(),
            len_edges=self.len_edges.copy(),
            n_anchors=self.n_anchors,
            n_fragments=self.n_fragments,
            assay=self.assay,
            feature=self.feature,
            normalization=f"diff:{m}",
            is_diff=True,
            genome_len_counts=None,
            metadata={"diff_mode": m, "pseudocount": (pc_a, pc_b)},
        )

    def marginal_position(
        self,
        len_range: tuple[float, float] | str | None = None,
        agg: str = "sum",
    ) -> pd.DataFrame:
        """Collapse the V-plot along fragment length to a 1D positional profile.

        ``len_range`` keeps length rows whose bin centre lies in the inclusive
        ``(min_len, max_len)`` range; a size-bin name of this V-plot's assay
        (``"tf"``, ``"nfr"``, ...) also works. Returns columns ``pos`` and ``value``.
        """
        if isinstance(len_range, str):
            len_range = resolve_len_range(self.assay, len_range)
        if len_range is None:
            sub = self.counts
        else:
            lc = self.len_centers
            mask = (lc >= len_range[0]) & (lc <= len_range[1])
            sub = self.counts[mask, :]

        if sub.shape[0] == 0:
            vals = np.zeros(self.shape[1], dtype=np.float64)
        elif agg == "sum":
            vals = np.sum(sub, axis=0)
        elif agg == "mean":
            vals = np.mean(sub, axis=0)
        else:
            raise ValueError(f"Unknown agg {agg!r}; expected 'sum' or 'mean'.")

        return pd.DataFrame({"pos": self.pos_centers, "value": vals})

    def marginal_length(
        self,
        pos_range: tuple[float, float] | None = None,
        agg: str = "sum",
    ) -> pd.DataFrame:
        """Collapse the V-plot along position to a 1D fragment-length profile.

        ``pos_range`` keeps position columns whose bin centre lies in the inclusive
        ``(min_pos, max_pos)`` range (bp from the anchor). Returns columns ``length``
        and ``value``.
        """
        if pos_range is None:
            sub = self.counts
        else:
            pc = self.pos_centers
            mask = (pc >= pos_range[0]) & (pc <= pos_range[1])
            sub = self.counts[:, mask]

        if sub.shape[1] == 0:
            vals = np.zeros(self.shape[0], dtype=np.float64)
        elif agg == "sum":
            vals = np.sum(sub, axis=1)
        elif agg == "mean":
            vals = np.mean(sub, axis=1)
        else:
            raise ValueError(f"Unknown agg {agg!r}; expected 'sum' or 'mean'.")

        return pd.DataFrame({"length": self.len_centers, "value": vals})

    def to_dataframe(self, label: str | None = None, **extra_cols: Any) -> pd.DataFrame:
        """Convert V-plot to a tidy DataFrame (``pos``, ``length``, ``value``) for ``plotnine``."""
        n_len, n_pos = self.shape
        pos_grid = np.tile(self.pos_centers, n_len)
        len_grid = np.repeat(self.len_centers, n_pos)
        data: dict[str, Any] = {
            "pos": pos_grid,
            "length": len_grid,
            "value": self.counts.ravel(),
        }
        if label is not None:
            data["label"] = label
        for k, v in extra_cols.items():
            data[k] = v
        return pd.DataFrame(data)

    def plot(self, **kwargs: Any) -> Any:
        """Render V-plot with ``matplotlib``."""
        from pyvplot.plotting import plot_vplot_matplotlib

        return plot_vplot_matplotlib(self, **kwargs)

    def plot_plotnine(self, **kwargs: Any) -> Any:
        """Render V-plot as a ``plotnine`` ``ggplot`` object."""
        from pyvplot.plotting import plot_vplot_plotnine

        return plot_vplot_plotnine(self, **kwargs)


class VPlotCollection:
    """Collection of labeled :class:`VPlot` objects for multi-condition faceting."""

    def __init__(self, items: list[tuple[VPlot, dict[str, Any]]] | None = None) -> None:
        self._items: list[tuple[VPlot, dict[str, Any]]] = list(items) if items is not None else []

    def add(self, vp: VPlot, **metadata: Any) -> VPlotCollection:
        """Add a :class:`VPlot` with condition/factor metadata (e.g., ``tf="CTCF", condition="WT"``)."""
        self._items.append((vp, dict(metadata)))
        return self

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        labels = [", ".join(f"{k}={v}" for k, v in meta.items()) for _, meta in self._items]
        return f"VPlotCollection({len(self)} V-plots: {'; '.join(labels) or 'empty'})"

    @property
    def items(self) -> list[tuple[VPlot, dict[str, Any]]]:
        return list(self._items)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all V-plots into a single tidy DataFrame ready for ``plotnine`` ``facet_grid``."""
        if not self._items:
            return pd.DataFrame(columns=["pos", "length", "value"])
        frames = [vp.to_dataframe(**meta) for vp, meta in self._items]
        return pd.concat(frames, ignore_index=True)

    def plot_plotnine(self, facet: str | None = None, **kwargs: Any) -> Any:
        """Render faceted V-plots with ``plotnine``."""
        from pyvplot.plotting import plot_vplot_collection_plotnine

        return plot_vplot_collection_plotnine(self, facet=facet, **kwargs)


def vplot(
    store: FragmentStore,
    anchors: Any,
    flank: int = 300,
    upstream: int | None = None,
    downstream: int | None = None,
    min_len: int = 20,
    max_len: int = 250,
    pos_bin: int = 1,
    len_bin: int = 1,
    feature: str = "midpoint",
    use_numba: bool | None = None,
) -> VPlot:
    """Count fragments by (position relative to anchor, fragment length) around ``anchors``.

    Each fragment within the window adds 1 to the cell given by its ``feature``
    position (relative to the anchor centre, in the anchor's 5'->3' orientation)
    and its length. Summed over anchors, protected TF sites give a V whose
    vertex sits at the anchor; nucleosomes give a broad band near 150 bp.

    Parameters
    ----------
    store:
        :class:`~pyvplot.store.FragmentStore` to query.
    anchors:
        Anchor table: DataFrame (``chrom`` + ``center``, or ``chrom`` + ``start``
        + ``end``; optional ``strand``), dict of columns, or a BED file path.
        ``-`` strand anchors are mirrored so upstream is always on the left.
    flank:
        Half-window (bp) on each side of the anchor centre.
    upstream, downstream:
        Asymmetric window (bp), relative to anchor orientation. Override ``flank``.
    min_len, max_len:
        Length axis, half-open ``[min_len, max_len)`` bp. ``max_len - min_len``
        must be a multiple of ``len_bin``.
    pos_bin:
        Bin width along position (bp). ``upstream + downstream`` must be a
        multiple of it.
    len_bin:
        Bin width along fragment length (bp).
    feature:
        Which point of each fragment is placed on the position axis:
        ``"midpoint"`` (default; the classic V-plot), ``"cuts"`` (both ends,
        each counted once), ``"cuts_plus"`` (left end in anchor orientation),
        or ``"cuts_minus"`` (right end).
    use_numba:
        ``None`` uses Numba when installed; ``False`` forces the NumPy path.

    Returns
    -------
    VPlot
        Raw counts in ``.counts`` with shape ``(n_len, n_pos)``. Normalise with
        :meth:`VPlot.normalize` before comparing samples.

    Examples
    --------
    >>> v = vp.vplot(store, "motifs.bed", flank=300, min_len=20, max_len=250, pos_bin=5, len_bin=5)
    >>> v.normalize("expected").plot()
    """
    if isinstance(min_len, str) or isinstance(max_len, str):
        raise TypeError(
            "vplot() needs numeric min_len/max_len: they are the edges of the length axis "
            "[min_len, max_len), not a filter. Named size bins such as 'tf' are accepted by "
            "the len_range= argument of to_matrix(), cut_site_footprint(), footprint_score(), "
            "phasogram() and extract_events()."
        )
    up = flank if upstream is None else int(upstream)
    dn = flank if downstream is None else int(downstream)

    grid, norm = accumulate_vplot(
        store=store,
        anchors=anchors,
        upstream=up,
        downstream=dn,
        min_len=min_len,
        max_len=max_len,
        pos_bin=pos_bin,
        len_bin=len_bin,
        feature=feature,
        use_numba=use_numba,
    )

    pos_edges = np.arange(-up, dn + pos_bin, pos_bin, dtype=np.float64)
    len_edges = np.arange(min_len, max_len + len_bin, len_bin, dtype=np.float64)

    # Precompute genome-wide length bin counts for expected_from="genome"
    n_len = len(len_edges) - 1
    genome_len_counts = np.zeros(n_len, dtype=np.float64)
    if store.n_fragments > 0:
        all_lens = store.lengths()
        in_range = (all_lens >= min_len) & (all_lens < max_len)
        if np.any(in_range):
            l_bins = (all_lens[in_range] - min_len) // len_bin
            genome_len_counts = np.bincount(l_bins, minlength=n_len).astype(np.float64)

    return VPlot(
        counts=grid,
        pos_edges=pos_edges,
        len_edges=len_edges,
        n_anchors=norm.n_anchors,
        n_fragments=store.n_fragments,
        assay=store.assay.name,
        feature=feature,
        normalization="raw",
        is_diff=False,
        genome_len_counts=genome_len_counts,
    )
