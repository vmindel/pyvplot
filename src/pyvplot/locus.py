"""Single-locus fragment view: one genomic window, optionally anchored on one motif."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyvplot.extract import extract_events

if TYPE_CHECKING:
    from pyvplot.store import FragmentStore
    from pyvplot.vplot import VPlot

_FEATURES = ("midpoint", "cuts", "cuts_plus", "cuts_minus", "cuts_proximal", "cuts_distal")
_NEEDS_MOTIF = ("cuts_proximal", "cuts_distal")

_REGION_RE = re.compile(r"^\s*([^:\s]+):([\d,]+)-([\d,]+)(?::([+-]))?\s*$")


def _parse_region(spec: Any, what: str) -> tuple[str, int, int, str | None]:
    """Parse ``"chr:start-end[:strand]"`` or ``(chrom, start, end[, strand])`` (0-based, half-open)."""
    if isinstance(spec, str):
        m = _REGION_RE.match(spec)
        if m is None:
            raise ValueError(
                f"Cannot parse {what} {spec!r}; expected 'chrom:start-end' or 'chrom:start-end:+' "
                "(0-based, half-open, like BED)."
            )
        chrom, s, e, st = m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", "")), m.group(4)
    elif isinstance(spec, (tuple, list)) and len(spec) in (3, 4):
        chrom, s, e = str(spec[0]), int(spec[1]), int(spec[2])
        st = str(spec[3]) if len(spec) == 4 else None
        if st is not None and st not in {"+", "-"}:
            raise ValueError(f"{what} strand must be '+' or '-', got {st!r}.")
    else:
        raise TypeError(
            f"{what} must be 'chrom:start-end' or a (chrom, start, end[, strand]) tuple, got {type(spec).__name__}."
        )
    if e <= s:
        raise ValueError(f"{what} {spec!r} has end <= start; intervals are [start, end).")
    return chrom, s, e, st


@dataclass(repr=False)
class LocusVPlot:
    """Fragments in one genomic window, as points of (position, fragment length).

    Attributes
    ----------
    points:
        One row per plotted point: ``x`` (plot coordinate), ``pos`` (genomic
        coordinate, 0-based), ``frag_len``, ``end`` (``"midpoint"``; ``"left"`` /
        ``"right"`` without a motif; with a motif ``"proximal"`` / ``"distal"`` for
        ``cuts`` and ``"upstream"`` / ``"downstream"`` for ``cuts_plus`` / ``cuts_minus``,
        in the motif's 5'->3' orientation) and ``motif_overlap``
        (``"full"``, ``"partial"`` or ``"none"``, per fragment; missing without a motif).
    chrom, start, end:
        The locus, 0-based half-open.
    motif:
        ``(start, end)`` of the motif in genomic coordinates, or ``None``.
    motif_strand:
        ``"+"`` or ``"-"``; the motif's orientation (``"+"`` without a motif).
    coords:
        ``"genomic"`` (``x`` is the genomic coordinate) or ``"relative"`` (``x`` is the
        distance from the motif centre, or the locus centre without a motif, in the
        motif's 5'->3' orientation).
    xlim:
        ``(low, high)`` of the window on the ``x`` axis.
    """

    points: pd.DataFrame
    chrom: str
    start: int
    end: int
    motif: tuple[int, int] | None
    motif_strand: str
    feature: str
    coords: str
    xlim: tuple[float, float]
    n_fragments: int
    assay: str = "generic"
    motif_overlap: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    store_n_fragments: int = 0
    orient: bool = True
    # 1-bp histogram of every fragment length in the source store (index = length), for
    # to_vplot()'s genome_len_counts. A histogram, not the lengths: a few kB instead of 4 bytes
    # per fragment held by every LocusVPlot.
    _genome_len_hist: np.ndarray | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        mt = "" if self.motif is None else f", motif={self.motif[0]}-{self.motif[1]}({self.motif_strand})"
        ov = "" if self.motif_overlap is None else f", motif_overlap={self.motif_overlap!r}"
        return (
            f"LocusVPlot({self.chrom}:{self.start}-{self.end}{mt}, feature={self.feature!r}, "
            f"coords={self.coords!r}{ov}, n_fragments={self.n_fragments}, n_points={len(self.points)})"
        )

    @property
    def motif_span(self) -> tuple[float, float] | None:
        """Motif interval on the ``x`` axis, or ``None``."""
        if self.motif is None:
            return None
        lo, hi = float(self.motif[0]), float(self.motif[1])
        if self.coords == "genomic":
            return lo, hi
        c = 0.5 * (lo + hi)
        a, b = lo - c, hi - c
        return (-b, -a) if (self.motif_strand == "-" and self.orient) else (a, b)

    def to_dataframe(self) -> pd.DataFrame:
        """The plotted points, with the locus attached as ``chrom``."""
        out = self.points.copy()
        out.insert(0, "chrom", self.chrom)
        return out

    def to_vplot(
        self,
        pos_bin: int = 5,
        len_bin: int = 5,
        min_len: int | None = None,
        max_len: int | None = None,
    ) -> VPlot:
        """Bin the points into a [`VPlot`][pyvplot.VPlot] over this window (``n_anchors=1``).

        The position axis is ``x``: genomic coordinates when ``coords="genomic"``, so pass
        ``center_line=False`` and a matching ``xlabel`` when plotting it. The V-plot's
        ``n_fragments`` is the whole store's, so ``normalize("cpm")`` is on the same scale as
        the aggregate V-plots of that store.
        """
        from pyvplot.vplot import VPlot

        if pos_bin <= 0 or len_bin <= 0:
            raise ValueError("`pos_bin` and `len_bin` must be positive integers.")
        lens = self.points["frag_len"].to_numpy(dtype=np.float64)
        lo = int(np.floor(lens.min() / len_bin) * len_bin) if min_len is None and len(lens) else (0 if min_len is None else int(min_len))
        hi = int(np.ceil((lens.max() + 1) / len_bin) * len_bin) if max_len is None and len(lens) else (lo + len_bin if max_len is None else int(max_len))
        if hi <= lo:
            raise ValueError(f"max_len ({hi}) must be greater than min_len ({lo}).")
        n_len = int(np.ceil((hi - lo) / len_bin))
        x0, x1 = self.xlim
        n_pos = max(1, int(np.ceil((x1 - x0) / pos_bin)))
        pos_edges = x0 + pos_bin * np.arange(n_pos + 1, dtype=np.float64)
        len_edges = lo + len_bin * np.arange(n_len + 1, dtype=np.float64)
        # Strict half-open binning [lo, hi) x [x0, x1) matching accumulate_vplot
        counts = np.zeros((n_len, n_pos), dtype=np.float64)
        if len(lens) > 0:
            xs = self.points["x"].to_numpy(dtype=np.float64)
            li = np.floor((lens - lo) / len_bin).astype(np.int64)
            pi = np.floor((xs - x0) / pos_bin).astype(np.int64)
            ok = (lens >= lo) & (lens < hi) & (li >= 0) & (li < n_len) & (xs >= x0) & (xs < pos_edges[-1]) & (pi >= 0) & (pi < n_pos)
            if np.any(ok):
                np.add.at(counts, (li[ok], pi[ok]), 1.0)

        genome_len_counts: np.ndarray | None = None
        if self._genome_len_hist is not None:
            # Same [lo, hi) cut as the locus counts and store.vplot; the last bin may be short.
            cum = np.concatenate([[0], np.cumsum(self._genome_len_hist)])
            e = np.clip(np.minimum(lo + len_bin * np.arange(n_len + 1), hi), 0, len(cum) - 1)
            genome_len_counts = (cum[e[1:]] - cum[e[:-1]]).astype(np.float64)

        return VPlot(
            counts=counts,
            pos_edges=pos_edges,
            len_edges=len_edges,
            n_anchors=1,
            n_fragments=self.store_n_fragments or self.n_fragments,
            assay=self.assay,
            feature=self.feature,
            genome_len_counts=genome_len_counts,
            motif_overlap=self.motif_overlap,
            metadata={"locus": f"{self.chrom}:{self.start}-{self.end}", "coords": self.coords},
        )

    def plot(self, **kwargs: Any) -> Any:
        """Scatter of the points with ``matplotlib``; returns ``(fig, ax)``."""
        from pyvplot.plotting import plot_locus_matplotlib

        return plot_locus_matplotlib(self, **kwargs)

    def plot_plotnine(self, **kwargs: Any) -> Any:
        """Scatter of the points with ``plotnine``; returns a ``ggplot``."""
        from pyvplot.plotting import plot_locus_plotnine

        return plot_locus_plotnine(self, **kwargs)


def locus_vplot(
    store: FragmentStore,
    region: Any,
    motif: Any = None,
    feature: str = "midpoint",
    len_range: tuple[int, int] | str | None = None,
    motif_overlap: str | None = None,
    coords: str | None = None,
    orient: bool = True,
) -> LocusVPlot:
    """Fragments in one genomic window as points of (position, fragment length).

    This is the single-locus counterpart of [`vplot`][pyvplot.vplot]: nothing is
    aggregated, so each fragment is one point (``"midpoint"``) or two (``"cuts"``).
    With a ``motif`` the points are relative to it and each end is labelled
    ``proximal`` (nearer the motif centre) or ``distal`` (farther), the split used by
    ``feature="cuts_proximal"`` / ``"cuts_distal"`` in the aggregate functions.

    Parameters
    ----------
    store:
        An unstranded [`FragmentStore`][pyvplot.FragmentStore].
    region:
        The window: ``"chrom:start-end"`` or ``(chrom, start, end)``. 0-based,
        half-open like BED, not UCSC's 1-based inclusive ``chr1:1,001-2,000``.
    motif:
        Optional single motif inside the window: ``"chrom:start-end[:strand]"`` or
        ``(chrom, start, end[, strand])``, same coordinates as ``region``. Without a
        motif the whole window is shown and the proximal/distal features are unavailable.
    feature:
        ``"midpoint"`` (one point per fragment), ``"cuts"`` (both ends), ``"cuts_plus"``
        (left end), ``"cuts_minus"`` (right end; both in the plotted orientation),
        ``"cuts_proximal"`` or ``"cuts_distal"`` (need ``motif``).
    len_range:
        Inclusive ``(min_len, max_len)`` fragment filter or an assay size-bin name.
    motif_overlap:
        ``"any"``, ``"full"`` or ``"none"``: keep fragments that touch, contain or miss
        the motif (needs ``motif``).
    coords:
        ``"genomic"`` puts genomic coordinates on ``x``; ``"relative"`` puts the distance
        from the motif centre (the window centre without a motif). Default: ``"relative"``
        with a motif, ``"genomic"`` without.
    orient:
        With ``coords="relative"``, flip the axis for a ``-`` strand motif so the motif
        reads 5'->3' left to right, as in the aggregate plots. It changes only the axis:
        which fragment ends are selected and how they are labelled follow the motif's
        strand either way.

    Returns
    -------
    LocusVPlot
        Plot with ``.plot()``, or bin with ``.to_vplot()``.

    Examples
    --------
    >>> lv = store.locus_vplot("chr1:10000-10600", motif="chr1:10280-10300:+", feature="cuts")
    >>> fig, ax = lv.plot()
    """
    if feature not in _FEATURES:
        raise ValueError(f"Unsupported feature {feature!r}; expected one of {list(_FEATURES)}.")
    if store.is_stranded:
        raise ValueError("locus_vplot needs fragment intervals; this store is stranded (one 5' end per read).")

    chrom, L, R, _ = _parse_region(region, "region")
    has_motif = motif is not None
    if has_motif:
        m_chrom, m_lo, m_hi, m_strand = _parse_region(motif, "motif")
        if m_chrom != chrom:
            raise ValueError(f"motif is on {m_chrom!r} but the region is on {chrom!r}.")
        if m_hi <= L or m_lo >= R:
            raise ValueError(f"motif {m_lo}-{m_hi} lies outside the region {L}-{R}.")
        m_strand = m_strand or "+"
    else:
        m_lo, m_hi, m_strand = L, R, "+"
        if feature in _NEEDS_MOTIF:
            raise ValueError(f"feature={feature!r} is defined relative to a motif; pass motif=...")
        if motif_overlap is not None:
            raise ValueError("motif_overlap needs a motif; pass motif=...")

    if coords is None:
        coords = "relative" if has_motif else "genomic"
    if coords not in {"genomic", "relative"}:
        raise ValueError(f"coords must be 'genomic' or 'relative', got {coords!r}.")

    c = 0.5 * (m_lo + m_hi) if has_motif else 0.5 * (L + R)
    is_minus = has_motif and m_strand == "-"
    flip_x = coords == "relative" and orient and is_minus
    up, dn = (R - c, c - L) if is_minus else (c - L, R - c)
    anchor = pd.DataFrame(
        {"chrom": [chrom], "center": [c], "start": [m_lo], "end": [m_hi], "strand": ["-" if is_minus else "+"]}
    )
    ev = extract_events(
        store, anchor, upstream=up, downstream=dn, len_range=len_range,
        overlap_mode="any", motif_overlap=motif_overlap,
    )

    rel_lo, rel_hi = (-up, dn) if flip_x else (float(L) - c, float(R) - c)
    if len(ev) == 0:
        pts = pd.DataFrame({k: [] for k in ("x", "pos", "frag_len", "end", "motif_overlap")})
    else:
        cols = {
            "midpoint": [("rel_mid", "midpoint")],
            "cuts": [("rel_proximal", "proximal"), ("rel_distal", "distal")] if has_motif
            else [("rel_left", "left"), ("rel_right", "right")],
            # With a motif these ends follow its 5'->3' orientation, so on a '-' motif the
            # upstream end is the fragment's genomic right end: "left"/"right" would mislead.
            "cuts_plus": [("rel_left", "upstream" if has_motif else "left")],
            "cuts_minus": [("rel_right", "downstream" if has_motif else "right")],
            "cuts_proximal": [("rel_proximal", "proximal")],
            "cuts_distal": [("rel_distal", "distal")],
        }[feature]
        parts = []
        for col, label in cols:
            rel = ev[col].to_numpy(dtype=np.float64)
            pos = (c - rel) if is_minus else (c + rel)
            keep = (pos >= L) & (pos < R)
            rel = rel[keep]
            pos = pos[keep]
            parts.append(
                pd.DataFrame(
                    {
                        "x": (-rel if (is_minus and not flip_x) else rel) if coords == "relative" else pos,
                        "pos": pos,
                        "frag_len": ev["frag_len"].to_numpy()[keep],
                        "end": label,
                        "motif_overlap": ev["motif_overlap"].to_numpy()[keep],
                    }
                )
            )
        pts = pd.concat(parts, ignore_index=True)
        pts["end"] = pd.Categorical(pts["end"], categories=[lab for _, lab in cols])
        if not has_motif:
            # Overlap with the whole window says nothing about a motif.
            pts["motif_overlap"] = np.nan
        pts["motif_overlap"] = pd.Categorical(pts["motif_overlap"], categories=["full", "partial", "none"])

    xlim = (float(L), float(R)) if coords == "genomic" else (float(rel_lo), float(rel_hi))
    return LocusVPlot(
        points=pts,
        chrom=chrom,
        start=L,
        end=R,
        motif=(m_lo, m_hi) if has_motif else None,
        motif_strand=m_strand,
        feature=feature,
        coords=coords,
        xlim=xlim,
        n_fragments=int(len(ev)),
        assay=store.assay.name,
        motif_overlap=motif_overlap,
        store_n_fragments=store.n_fragments,
        orient=orient,
        _genome_len_hist=np.bincount(store.lengths()) if store.n_fragments else np.zeros(0, dtype=np.int64),
    )
