"""Size-stratified signal matrices, 1-bp cut-site footprints, and motif protection scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.assays import resolve_len_range
from pyvplot.extract import accumulate_matrix
from pyvplot.store import FragmentStore


def _resolve_len_range(
    store: FragmentStore,
    len_range: tuple[int, int] | str | None,
) -> tuple[int, int] | None:
    """Resolve an explicit ``(min_len, max_len)`` tuple or named assay size bin."""
    return resolve_len_range(store.assay, len_range)


@dataclass(repr=False)
class SignalMatrix:
    """Per-anchor binned signal matrix: one row per anchor, one column per position bin.

    Implements duck-typed mapping access (``mat["values"]``, ``mat["upstream"]``,
    etc.) and ``__array__`` so it can be passed directly into downstream tools
    or exported with [`to_dict`][pyvplot.SignalMatrix.to_dict]. Plot with
    [`plot`][pyvplot.SignalMatrix.plot] or [`plot_plotnine`][pyvplot.SignalMatrix.plot_plotnine].

    Attributes
    ----------
    values:
        2D ``float64`` array of shape ``(n_anchors, n_bins)``.
    region_index:
        1D array of anchor identifiers preserving input order.
    upstream:
        Upstream window size from anchor center (bp).
    downstream:
        Downstream window size from anchor center (bp).
    bin_size:
        Width of each genomic bin (bp).
    feature:
        Feature extracted (``"midpoint"``, ``"cuts"``, ``"cuts_plus"``,
        ``"cuts_minus"``, ``"cuts_proximal"``, ``"cuts_distal"`` or ``"coverage"``).
    len_range:
        Optional inclusive ``(min_len, max_len)`` filter used during extraction.
    n_fragments:
        Total fragments in source [`FragmentStore`][pyvplot.FragmentStore].
    assay:
        Source assay name.
    normalization:
        Normalization applied (``"raw"``, ``"cpm"``, ``"rpkm"``, etc.).
    metadata:
        Arbitrary metadata dictionary.
    motif_overlap:
        Motif-overlap filter used at extraction (``None``, ``"any"``,
        ``"full"`` or ``"none"``).
    """

    values: np.ndarray
    region_index: np.ndarray
    upstream: int
    downstream: int
    bin_size: int
    feature: str = "midpoint"
    len_range: tuple[int, int] | None = None
    n_fragments: int = 0
    assay: str = "generic"
    normalization: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)
    motif_overlap: str | None = None

    def __repr__(self) -> str:
        lr = "all" if self.len_range is None else f"{self.len_range[0]}-{self.len_range[1]} bp"
        ov = "" if self.motif_overlap is None else f"motif_overlap={self.motif_overlap!r}, "
        return (
            f"SignalMatrix(shape={self.shape}, window=[-{self.upstream}, +{self.downstream}) bp, "
            f"bin_size={self.bin_size}, feature={self.feature!r}, lengths={lr}, {ov}"
            f"normalization={self.normalization!r}, assay={self.assay!r})"
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def n_anchors(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[1])

    @property
    def bin_edges(self) -> np.ndarray:
        return np.arange(-self.upstream, self.downstream + self.bin_size, self.bin_size, dtype=np.float64)

    @property
    def bin_centers(self) -> np.ndarray:
        edges = self.bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def keys(self) -> tuple[str, ...]:
        return ("values", "region_index", "upstream", "downstream", "bin_size")

    def __getitem__(self, key: str) -> Any:
        if key in self.keys():
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (``values``, ``region_index``, ``upstream``, ``downstream``, ``bin_size``).

        Uses only NumPy and built-in types, so any other tool can consume it
        without pyvplot installed.
        """
        return {
            "values": self.values,
            "region_index": self.region_index,
            "upstream": self.upstream,
            "downstream": self.downstream,
            "bin_size": self.bin_size,
        }

    def plot(self, **kwargs: Any) -> Any:
        """Heatmap (one row per anchor) with the mean profile on top, via ``matplotlib``.

        See [`plot_signal_matrix_matplotlib`][pyvplot.plotting.plot_signal_matrix_matplotlib].
        Returns ``(fig, (ax_profile, ax_heatmap))``.
        """
        from pyvplot.plotting import plot_signal_matrix_matplotlib

        return plot_signal_matrix_matplotlib(self, **kwargs)

    def plot_plotnine(self, **kwargs: Any) -> Any:
        """Heatmap (one row per anchor) as a ``plotnine`` ``ggplot``.

        See [`plot_signal_matrix_plotnine`][pyvplot.plotting.plot_signal_matrix_plotnine].
        """
        from pyvplot.plotting import plot_signal_matrix_plotnine

        return plot_signal_matrix_plotnine(self, **kwargs)

    def normalize(self, method: str = "cpm") -> SignalMatrix:
        """Return a normalized copy of this signal matrix.

        Supported methods:
        - ``"cpm"``: counts per million total fragments (``values / (n_fragments / 1e6)``)
        - ``"rpkm"``: counts per million fragments per kilobase of bin width
        - ``"zscore"``: global z-score across the entire matrix
        - ``"row_zscore"``: z-score each anchor row independently
        - ``"row_sum"``: divide each anchor row by its sum
        """
        m = method.lower()
        arr = self.values

        if not self.normalization.startswith("raw"):
            raise ValueError(
                f"This matrix is already normalised ({self.normalization!r}); normalising it again "
                f"with {method!r} would compound the scaling. Call normalize() on the raw matrix."
            )

        if m == "cpm":
            denom = max(1e-12, float(self.n_fragments) / 1e6) if self.n_fragments > 0 else 1.0
            out = arr / denom
        elif m == "rpkm":
            denom = (
                (max(1e-12, float(self.n_fragments) / 1e6) if self.n_fragments > 0 else 1.0)
                * (float(self.bin_size) / 1000.0)
            )
            out = arr / denom
        elif m == "zscore":
            sd = float(np.std(arr))
            out = (arr - float(np.mean(arr))) / sd if sd > 0.0 else np.zeros_like(arr)
        elif m == "row_zscore":
            mu = np.mean(arr, axis=1, keepdims=True)
            sd = np.std(arr, axis=1, keepdims=True)
            out = np.where(sd > 0.0, (arr - mu) / np.where(sd > 0.0, sd, 1.0), 0.0)
        elif m == "row_sum":
            rs = np.sum(arr, axis=1, keepdims=True)
            out = np.where(rs > 0.0, arr / np.where(rs > 0.0, rs, 1.0), 0.0)
        else:
            raise ValueError(
                f"Unknown normalization method {method!r}; expected 'cpm', 'rpkm', "
                "'zscore', 'row_zscore', or 'row_sum'."
            )

        return SignalMatrix(
            values=np.ascontiguousarray(out, dtype=np.float64),
            region_index=self.region_index.copy(),
            upstream=self.upstream,
            downstream=self.downstream,
            bin_size=self.bin_size,
            feature=self.feature,
            len_range=self.len_range,
            n_fragments=self.n_fragments,
            assay=self.assay,
            normalization=m,
            metadata=dict(self.metadata),
            motif_overlap=self.motif_overlap,
        )

    def raw_count_scale(self) -> float | None:
        """Value of one raw count in this matrix's current units, or ``None`` if unknown.

        Known for raw (including bias-corrected raw), ``cpm`` and ``rpkm`` matrices.
        """
        if self.normalization.startswith("raw"):
            return 1.0
        per_million = float(self.n_fragments) / 1e6 if self.n_fragments > 0 else 1.0
        if self.normalization == "cpm":
            return 1.0 / per_million
        if self.normalization == "rpkm":
            return 1.0 / (per_million * float(self.bin_size) / 1000.0)
        return None

    def subset(self, indices: Any) -> SignalMatrix:
        """Return a new [`SignalMatrix`][pyvplot.SignalMatrix] restricted to selected anchor rows."""
        idx_arr = np.asarray(indices)
        return SignalMatrix(
            values=np.ascontiguousarray(self.values[idx_arr], dtype=np.float64),
            region_index=self.region_index[idx_arr].copy(),
            upstream=self.upstream,
            downstream=self.downstream,
            bin_size=self.bin_size,
            feature=self.feature,
            len_range=self.len_range,
            n_fragments=self.n_fragments,
            assay=self.assay,
            normalization=self.normalization,
            metadata=dict(self.metadata),
            motif_overlap=self.motif_overlap,
        )

    def mean_profile(self, ci: float | None = None) -> pd.DataFrame:
        """Compute aggregate positional profile across anchors.

        Parameters
        ----------
        ci:
            Optional confidence interval level (e.g., ``0.95``) computed via
            normal approximation (``mean +/- z * sem``).
        """
        n = self.n_anchors
        if n == 0:
            cols = ["pos", "mean", "std", "sem"]
            if ci is not None:
                cols.extend(["ci_low", "ci_high"])
            return pd.DataFrame(columns=cols)

        mean = np.mean(self.values, axis=0)
        std = np.std(self.values, axis=0, ddof=1 if n > 1 else 0)
        sem = std / np.sqrt(float(n))
        df = pd.DataFrame(
            {
                "pos": self.bin_centers,
                "mean": mean,
                "std": std,
                "sem": sem,
            }
        )
        if ci is not None:
            if not (0.0 < ci < 1.0):
                raise ValueError("`ci` must be strictly between 0 and 1.")
            try:
                from scipy.stats import norm as _norm

                z = float(_norm.ppf(0.5 + 0.5 * ci))
            except ImportError:
                # Common fallback for 95% / 90% / 99%
                z_map = {0.90: 1.6448536, 0.95: 1.9599640, 0.99: 2.5758293}
                z = z_map.get(round(ci, 2), 1.9599640)
            df["ci_low"] = mean - z * sem
            df["ci_high"] = mean + z * sem
        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Convert matrix to a tidy long-format DataFrame (``anchor``, ``pos``, ``value``)."""
        n_a, n_b = self.shape
        return pd.DataFrame(
            {
                "anchor": np.repeat(self.region_index, n_b),
                "pos": np.tile(self.bin_centers, n_a),
                "value": self.values.ravel(),
            }
        )


def to_matrix(
    store: FragmentStore,
    anchors: Any,
    flank: int = 1000,
    upstream: int | None = None,
    downstream: int | None = None,
    bin_size: int = 10,
    feature: str | None = None,
    len_range: tuple[int, int] | str | None = None,
    normalize: str | None = None,
    motif_overlap: str | None = None,
    use_numba: bool | None = None,
) -> SignalMatrix:
    """Compute a size-stratified ``(n_anchors, n_bins)`` [`SignalMatrix`][pyvplot.SignalMatrix].

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to query.
    anchors:
        Anchor DataFrame, dict, list, or BED file path.
    flank:
        Symmetric half-window (bp) used when ``upstream``/``downstream`` are omitted.
    upstream, downstream:
        Optional asymmetric window bounds (bp) around each anchor center.
    bin_size:
        Bin width (bp). ``upstream + downstream`` must be divisible by ``bin_size``.
    feature:
        Feature to extract: ``"midpoint"``, ``"cuts"``, ``"cuts_plus"``,
        ``"cuts_minus"``, ``"cuts_proximal"`` (end nearer the anchor centre),
        ``"cuts_distal"`` (farther end) or ``"coverage"``. Defaults to the
        store's ``assay.default_feature`` (mapping ``"cuts_stranded"`` to ``"cuts"``).
    len_range:
        Optional inclusive ``(min_len, max_len)`` fragment-length filter, or the
        name of a preset size bin in ``store.assay.size_bins`` (e.g., ``"tf"``,
        ``"nfr"``, ``"mononucleosome"``).
    normalize:
        Optional normalization applied immediately (e.g., ``"cpm"``, ``"rpkm"``).
    motif_overlap:
        Keep only fragments overlapping each anchor's motif interval
        (``"any"``), containing it (``"full"``), or missing it (``"none"``).
        See [`vplot`][pyvplot.vplot] for how the interval is defined.
    use_numba:
        Override automatic Numba dispatch.
    """
    up = flank if upstream is None else int(upstream)
    dn = flank if downstream is None else int(downstream)

    if feature is None:
        feat = "cuts" if store.assay.default_feature == "cuts_stranded" else store.assay.default_feature
    else:
        feat = feature

    resolved_len = _resolve_len_range(store, len_range)

    arr, norm = accumulate_matrix(
        store=store,
        anchors=anchors,
        upstream=up,
        downstream=dn,
        bin_size=bin_size,
        feature=feat,
        len_range=resolved_len,
        use_numba=use_numba,
        motif_overlap=motif_overlap,
    )

    mat = SignalMatrix(
        values=arr,
        region_index=norm.region_index,
        upstream=up,
        downstream=dn,
        bin_size=bin_size,
        feature=feat,
        len_range=resolved_len,
        n_fragments=store.n_fragments,
        assay=store.assay.name,
        normalization="raw",
        motif_overlap=motif_overlap,
    )
    if normalize is not None:
        mat = mat.normalize(normalize)
    return mat


@dataclass(repr=False)
class CutSiteFootprint:
    """1-bp resolution cleavage cut-site footprint around genomic anchors.

    Attributes
    ----------
    pos:
        1D array of bin centers (bp relative to anchor center).
    combined:
        1D aggregate cut-site signal (both ends combined).
    plus:
        Optional 1D aggregate left-end (``cuts_plus``) signal; set when ``split_by="strand"``.
    minus:
        Optional 1D aggregate right-end (``cuts_minus``) signal; set when ``split_by="strand"``.
    matrix_combined:
        Optional per-anchor [`SignalMatrix`][pyvplot.SignalMatrix] for combined cuts.
    matrix_plus:
        Optional per-anchor [`SignalMatrix`][pyvplot.SignalMatrix] for ``+`` cuts.
    matrix_minus:
        Optional per-anchor [`SignalMatrix`][pyvplot.SignalMatrix] for ``-`` cuts.
    n_anchors:
        Number of anchors aggregated.
    agg:
        Aggregation mode (``"mean"`` or ``"sum"``).
    proximal, distal:
        Optional 1D aggregate signal of the end nearer to / farther from the
        anchor centre; set when ``split_by="proximity"``.
    matrix_proximal, matrix_distal:
        Optional per-anchor [`SignalMatrix`][pyvplot.SignalMatrix] objects for
        the proximal and distal ends.
    split_by:
        ``"strand"``, ``"proximity"``, or ``None`` (combined only).
    motif_overlap:
        Motif-overlap filter used at extraction.
    """

    pos: np.ndarray
    combined: np.ndarray
    plus: np.ndarray | None = None
    minus: np.ndarray | None = None
    matrix_combined: SignalMatrix | None = None
    matrix_plus: SignalMatrix | None = None
    matrix_minus: SignalMatrix | None = None
    n_anchors: int = 0
    agg: str = "mean"
    proximal: np.ndarray | None = None
    distal: np.ndarray | None = None
    matrix_proximal: SignalMatrix | None = None
    matrix_distal: SignalMatrix | None = None
    split_by: str | None = None
    motif_overlap: str | None = None

    def __repr__(self) -> str:
        span = f"[{self.pos[0]:g}, {self.pos[-1]:g}]" if len(self.pos) else "[]"
        ov = "" if self.motif_overlap is None else f"motif_overlap={self.motif_overlap!r}, "
        return (
            f"CutSiteFootprint(pos={span} bp, n_bins={len(self.pos)}, n_anchors={self.n_anchors}, "
            f"split_by={self.split_by!r}, {ov}agg={self.agg!r}, "
            f"has_matrices={self.matrix_combined is not None})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export to a tidy DataFrame.

        Columns are ``pos``, ``signal`` and the split label: ``strand``
        (``"+"``/``"-"``, or ``"both"`` when unsplit) for a strand split, or
        ``end`` (``"proximal"``/``"distal"``) for a proximity split.
        """
        if self.proximal is not None and self.distal is not None:
            return pd.concat(
                [
                    pd.DataFrame({"pos": self.pos, "end": "proximal", "signal": self.proximal}),
                    pd.DataFrame({"pos": self.pos, "end": "distal", "signal": self.distal}),
                ],
                ignore_index=True,
            )
        frames: list[pd.DataFrame] = []
        if self.plus is not None and self.minus is not None:
            frames.append(pd.DataFrame({"pos": self.pos, "strand": "+", "signal": self.plus}))
            frames.append(pd.DataFrame({"pos": self.pos, "strand": "-", "signal": self.minus}))
        else:
            frames.append(pd.DataFrame({"pos": self.pos, "strand": "both", "signal": self.combined}))
        return pd.concat(frames, ignore_index=True)

    def plot(self, **kwargs: Any) -> Any:
        """Render cut-site footprint with ``matplotlib``."""
        from pyvplot.plotting import plot_cut_footprint_matplotlib

        return plot_cut_footprint_matplotlib(self, **kwargs)

    def plot_plotnine(self, **kwargs: Any) -> Any:
        """Render cut-site footprint with ``plotnine``."""
        from pyvplot.plotting import plot_cut_footprint_plotnine

        return plot_cut_footprint_plotnine(self, **kwargs)


def cut_site_footprint(
    store: FragmentStore,
    anchors: Any,
    flank: int = 100,
    upstream: int | None = None,
    downstream: int | None = None,
    bin_size: int = 1,
    len_range: tuple[int, int] | str | None = None,
    split_strands: bool = True,
    normalize: str | None = None,
    agg: str = "mean",
    keep_matrices: bool = True,
    split_by: str = "strand",
    motif_overlap: str | None = None,
    use_numba: bool | None = None,
) -> CutSiteFootprint:
    """Compute 1-bp cleavage cut-site profiles, split by fragment end or combined.

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to query.
    anchors:
        Anchor table (DataFrame, dict, or BED path).
    flank:
        Half-window around each anchor center (bp).
    bin_size:
        Resolution in bp (default ``1``).
    len_range:
        Optional fragment-length filter or named assay size bin.
    split_strands:
        If ``True`` (default), extract the two ends separately as well as
        their sum, as chosen by ``split_by`` (despite the name, this covers the
        proximity split too). ``False`` gives the combined profile only and
        cannot be combined with ``split_by="proximity"``.
    normalize:
        Optional normalization (e.g., ``"cpm"``) applied to each matrix prior
        to aggregation.
    agg:
        Aggregation across anchors: ``"mean"`` (default) or ``"sum"``.
    keep_matrices:
        Whether to retain the per-anchor [`SignalMatrix`][pyvplot.SignalMatrix] objects on the
        returned [`CutSiteFootprint`][pyvplot.CutSiteFootprint].
    split_by:
        How the ends are split when ``split_strands=True``: ``"strand"``
        (default; left/right end in anchor orientation, i.e. ``cuts_plus`` /
        ``cuts_minus``, filling ``plus``/``minus``) or ``"proximity"`` (end
        nearer to / farther from the anchor centre, filling
        ``proximal``/``distal``; unstranded stores only).
    motif_overlap:
        Keep only fragments overlapping each anchor's motif interval
        (``"any"``), containing it (``"full"``), or missing it (``"none"``).
        See [`vplot`][pyvplot.vplot].
    """
    if agg not in {"mean", "sum"}:
        raise ValueError(f"Invalid agg {agg!r}; expected 'mean' or 'sum'.")
    if split_by not in {"strand", "proximity"}:
        raise ValueError(f"Invalid split_by {split_by!r}; expected 'strand' or 'proximity'.")

    if not split_strands and split_by != "strand":
        raise ValueError(
            f"split_by={split_by!r} needs split_strands=True; with split_strands=False only "
            "the combined profile is computed."
        )

    reducer = np.mean if agg == "mean" else np.sum

    def _mat(feat: str) -> SignalMatrix:
        return to_matrix(
            store,
            anchors,
            flank=flank,
            upstream=upstream,
            downstream=downstream,
            bin_size=bin_size,
            feature=feat,
            len_range=len_range,
            normalize=normalize,
            motif_overlap=motif_overlap,
            use_numba=use_numba,
        )

    def _profile(m: SignalMatrix) -> np.ndarray:
        if m.n_anchors == 0:
            return np.zeros(m.n_bins, dtype=np.float64)
        return np.asarray(reducer(m.values, axis=0), dtype=np.float64)

    if split_strands:
        feats = ("cuts_plus", "cuts_minus") if split_by == "strand" else ("cuts_proximal", "cuts_distal")
        m_a, m_b = _mat(feats[0]), _mat(feats[1])
        m_comb = SignalMatrix(
            values=m_a.values + m_b.values,
            region_index=m_a.region_index.copy(),
            upstream=m_a.upstream,
            downstream=m_a.downstream,
            bin_size=m_a.bin_size,
            feature="cuts",
            len_range=m_a.len_range,
            n_fragments=m_a.n_fragments,
            assay=m_a.assay,
            normalization=m_a.normalization,
            motif_overlap=motif_overlap,
        )
        p_a, p_b = _profile(m_a), _profile(m_b)
        keep = keep_matrices
        common: dict[str, Any] = {
            "pos": m_a.bin_centers,
            "combined": p_a + p_b,
            "matrix_combined": m_comb if keep else None,
            "n_anchors": m_a.n_anchors,
            "agg": agg,
            "split_by": split_by,
            "motif_overlap": motif_overlap,
        }
        if split_by == "strand":
            return CutSiteFootprint(
                plus=p_a, minus=p_b,
                matrix_plus=m_a if keep else None, matrix_minus=m_b if keep else None,
                **common,
            )
        return CutSiteFootprint(
            proximal=p_a, distal=p_b,
            matrix_proximal=m_a if keep else None, matrix_distal=m_b if keep else None,
            **common,
        )

    m_comb = _mat("cuts")
    return CutSiteFootprint(
        pos=m_comb.bin_centers,
        combined=_profile(m_comb),
        matrix_combined=m_comb if keep_matrices else None,
        n_anchors=m_comb.n_anchors,
        agg=agg,
        split_by=None,
        motif_overlap=motif_overlap,
    )


def footprint_score(
    store: FragmentStore,
    anchors: Any,
    motif_half_width: int = 10,
    flank_width: int = 30,
    len_range: tuple[int, int] | str | None = None,
    feature: str = "cuts",
    pseudocount: float = 0.05,
    normalize: str | None = None,
    use_numba: bool | None = None,
) -> pd.DataFrame:
    """Compute per-anchor TF footprint protection depth and flanking-vs-center ratios.

    Compares cleavage density in the protected motif center
    ``[-motif_half_width, +motif_half_width)`` against the immediately flanking
    shoulders ``[-motif_half_width - flank_width, -motif_half_width)`` and
    ``[+motif_half_width, +motif_half_width + flank_width)``.

    Returns a `pandas.DataFrame` indexed by anchor ID with columns:
    - ``center_cuts``: sum of signal in the motif center window
    - ``flank_cuts``: sum of signal in both flanking windows
    - ``center_density``: per-bp density in motif center
    - ``flank_density``: per-bp density in flanking shoulders
    - ``protection_score``: ``log2((flank_density + pseudocount) / (center_density + pseudocount))``
      (positive values indicate stronger protection at the motif center)
    - ``footprint_depth``: ``center_density - flank_density`` (negative dip = deeper footprint)
    - ``accessibility``: total per-bp density across center + flanks
    """
    if motif_half_width <= 0 or flank_width <= 0:
        raise ValueError("`motif_half_width` and `flank_width` must be positive integers.")

    total_flank = motif_half_width + flank_width
    mat = to_matrix(
        store,
        anchors,
        flank=total_flank,
        bin_size=1,
        feature=feature,
        len_range=len_range,
        normalize=normalize,
        use_numba=use_numba,
    )

    # Bins span [-total_flank, +total_flank):
    # Left flank:  [0, flank_width)
    # Center:      [flank_width, flank_width + 2 * motif_half_width)
    # Right flank: [flank_width + 2 * motif_half_width, 2 * total_flank)
    c_lo = flank_width
    c_hi = flank_width + 2 * motif_half_width

    vals = mat.values
    center_cuts = np.sum(vals[:, c_lo:c_hi], axis=1)
    flank_cuts = np.sum(vals[:, :c_lo], axis=1) + np.sum(vals[:, c_hi:], axis=1)

    center_bp = float(2 * motif_half_width)
    flank_bp = float(2 * flank_width)
    center_density = center_cuts / center_bp
    flank_density = flank_cuts / flank_bp

    protection = np.log2((flank_density + pseudocount) / (center_density + pseudocount))
    depth = center_density - flank_density
    accessibility = (center_cuts + flank_cuts) / (center_bp + flank_bp)

    return pd.DataFrame(
        {
            "center_cuts": center_cuts,
            "flank_cuts": flank_cuts,
            "center_density": center_density,
            "flank_density": flank_density,
            "protection_score": protection,
            "footprint_depth": depth,
            "accessibility": accessibility,
        },
        index=mat.region_index,
    )
