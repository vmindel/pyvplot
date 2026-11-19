"""Numba-accelerated (with pure-NumPy fallback) strand-aware anchor extraction."""

from __future__ import annotations

import gzip
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.assays import resolve_len_range
from pyvplot.store import FragmentStore

try:
    from numba import get_num_threads, njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn

        return _decorator

    def prange(*args: Any) -> range:
        return range(*args)

    def get_num_threads() -> int:
        return 1


FEATURE_MIDPOINT = 0
FEATURE_CUTS = 1
FEATURE_CUTS_PLUS = 2
FEATURE_CUTS_MINUS = 3
FEATURE_COVERAGE = 4
FEATURE_CUTS_PROXIMAL = 5
FEATURE_CUTS_DISTAL = 6

_FEATURE_CODES: dict[str, int] = {
    "midpoint": FEATURE_MIDPOINT,
    "cuts": FEATURE_CUTS,
    "cuts_plus": FEATURE_CUTS_PLUS,
    "cuts_minus": FEATURE_CUTS_MINUS,
    "coverage": FEATURE_COVERAGE,
    # The end nearer to / farther from the anchor centre (unstranded stores only).
    "cuts_proximal": FEATURE_CUTS_PROXIMAL,
    "cuts_distal": FEATURE_CUTS_DISTAL,
    # ChIP-exo preset default. Stranded stores already hold one 5' end per read,
    # and the kernels read per-fragment strand, so this is the same as "cuts".
    "cuts_stranded": FEATURE_CUTS,
}


def feature_to_code(feature: str) -> int:
    """Convert feature name to integer kernel code."""
    if feature not in _FEATURE_CODES:
        raise ValueError(
            f"Unsupported feature {feature!r}. Expected one of {sorted(_FEATURE_CODES)}."
        )
    return _FEATURE_CODES[feature]


OVERLAP_OFF = 0
OVERLAP_ANY = 1
OVERLAP_FULL = 2
OVERLAP_NONE = 3

_OVERLAP_CODES: dict[str | None, int] = {
    None: OVERLAP_OFF,
    "any": OVERLAP_ANY,
    "full": OVERLAP_FULL,
    "none": OVERLAP_NONE,
}


def motif_overlap_to_code(motif_overlap: str | None) -> int:
    """Convert a ``motif_overlap`` setting to its integer kernel code."""
    if motif_overlap not in _OVERLAP_CODES:
        raise ValueError(
            f"Invalid motif_overlap {motif_overlap!r}; expected None, 'any', 'full' or 'none'."
        )
    return _OVERLAP_CODES[motif_overlap]


def _check_stranded_support(store: FragmentStore, feature_code: int, overlap_code: int) -> None:
    if store.strands is None:
        return
    if feature_code in (FEATURE_CUTS_PROXIMAL, FEATURE_CUTS_DISTAL):
        raise ValueError(
            "feature='cuts_proximal'/'cuts_distal' needs both ends of each fragment; this store is "
            "stranded (one 5' end per read), so there is no near or far end. Use 'cuts_plus'/"
            "'cuts_minus' to split by read strand."
        )
    if overlap_code != OVERLAP_OFF:
        raise ValueError(
            "motif_overlap needs fragment intervals; this store is stranded (one 5' end per read)."
        )


@njit(cache=True)
def _overlap_ok(s: float, e: float, m_lo: float, m_hi: float, code: int) -> bool:
    """Whether fragment ``[s, e)`` passes the motif-overlap filter for motif ``[m_lo, m_hi)``."""
    if code == OVERLAP_ANY:
        return s < m_hi and e > m_lo
    if code == OVERLAP_FULL:
        return s <= m_lo and e >= m_hi
    if code == OVERLAP_NONE:
        return not (s < m_hi and e > m_lo)
    return True


def _overlap_mask(
    s: np.ndarray, e: np.ndarray, m_lo: float, m_hi: float, code: int
) -> np.ndarray | None:
    """Vectorised [`_overlap_ok`][]; ``None`` when no filter is applied."""
    if code == OVERLAP_OFF:
        return None
    hit = (s < m_hi) & (e > m_lo)
    if code == OVERLAP_ANY:
        return hit
    if code == OVERLAP_FULL:
        return (s <= m_lo) & (e >= m_hi)
    return ~hit


@dataclass(frozen=True)
class NormalizedAnchors:
    """Validated, array-backed representation of genomic anchors.

    Attributes
    ----------
    chroms:
        1D array of chromosome names (length ``n_anchors``).
    centers:
        1D ``float64`` array of 0-based anchor centers.
    strands:
        1D ``int8`` array (``+1`` for forward/unstranded, ``-1`` for reverse).
    region_index:
        1D array preserving the original anchor index / identifiers.
    chrom_groups:
        Mapping ``chrom -> int64 array of original anchor indices`` for
        efficient per-chromosome processing.
    motif_starts, motif_ends:
        Optional 1D ``float64`` arrays giving each anchor's genomic interval
        ``[start, end)`` (from BED columns 2-3 or ``start``/``end`` columns).
        Used only by the ``motif_overlap`` filter. When absent, each anchor is
        the 1-bp interval ``[center - 0.5, center + 0.5)``.
    """

    chroms: np.ndarray
    centers: np.ndarray
    strands: np.ndarray
    region_index: np.ndarray
    chrom_groups: dict[str, np.ndarray]
    motif_starts: np.ndarray | None = None
    motif_ends: np.ndarray | None = None

    @property
    def n_anchors(self) -> int:
        return int(len(self.centers))

    def motif_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-anchor motif interval ``[start, end)`` in genomic coordinates."""
        if self.motif_starts is not None and self.motif_ends is not None:
            return self.motif_starts, self.motif_ends
        # Centred 1-bp interval: mirror-symmetric, and equal to [start, start + 1) for 1-bp BED records.
        return self.centers - 0.5, self.centers + 0.5


def _read_anchors_bed(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[tuple[str, int, int, float, str, str]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line or line[0] == "#" or line.startswith(("track ", "browser ")):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            center = 0.5 * (start + end)
            name = parts[3] if len(parts) >= 4 and parts[3] not in {"+", "-", "."} else str(idx)
            strand = "+"
            if len(parts) >= 6 and parts[5] in {"+", "-"}:
                strand = parts[5]
            elif len(parts) >= 4 and parts[3] in {"+", "-"}:
                strand = parts[3]
            records.append((chrom, start, end, center, strand, name))
    df = pd.DataFrame(records, columns=["chrom", "start", "end", "center", "strand", "name"])
    if len(df) > 0:
        df = df.set_index("name")
    return df


def normalize_anchors(anchors: Any) -> NormalizedAnchors:
    """Convert anchors from a DataFrame, dict, list, or BED path to [`NormalizedAnchors`][pyvplot.NormalizedAnchors].

    Accepted DataFrame columns:
    - Chromosome: ``chrom``, ``chr``, ``seqnames``, or ``chromosome``
    - Position: ``center``, ``pos``, ``midpoint``, or ``summit``; alternatively
      ``start`` and ``end`` (computes ``0.5 * (start + end)``)
    - Strand (optional): ``strand``, ``orientation``, or ``dir`` (``"-"`` or
      ``-1`` flips coordinates; defaults to ``"+"``)

    When ``start`` and ``end`` are both present they are kept as the motif
    interval for ``motif_overlap`` filtering, even if a ``center`` column sets
    the window centre.
    """
    if isinstance(anchors, NormalizedAnchors):
        return anchors

    if isinstance(anchors, (str, Path)):
        df = _read_anchors_bed(Path(anchors))
    elif isinstance(anchors, pd.DataFrame):
        df = anchors
    elif isinstance(anchors, dict):
        df = pd.DataFrame(anchors)
    else:
        df = pd.DataFrame(anchors)

    if len(df) == 0:
        return NormalizedAnchors(
            chroms=np.empty(0, dtype=object),
            centers=np.empty(0, dtype=np.float64),
            strands=np.empty(0, dtype=np.int8),
            region_index=df.index.to_numpy() if hasattr(df, "index") else np.empty(0, dtype=np.int64),
            chrom_groups={},
        )

    col_map = {str(c).lower(): c for c in df.columns}

    chrom_col = None
    for cand in ("chrom", "chr", "seqnames", "chromosome"):
        if cand in col_map:
            chrom_col = col_map[cand]
            break
    if chrom_col is None:
        raise ValueError(
            "Anchors must contain a chromosome column ('chrom', 'chr', 'seqnames', or 'chromosome')."
        )

    center_col = None
    for cand in ("center", "pos", "midpoint", "summit"):
        if cand in col_map:
            center_col = col_map[cand]
            break

    starts_a: np.ndarray | None = None
    ends_a: np.ndarray | None = None
    if "start" in col_map and "end" in col_map:
        starts_a = np.ascontiguousarray(df[col_map["start"]].to_numpy(dtype=np.float64))
        ends_a = np.ascontiguousarray(df[col_map["end"]].to_numpy(dtype=np.float64))
        if np.any(ends_a < starts_a):
            raise ValueError("Anchors have end < start; intervals must be [start, end).")

    if center_col is not None:
        centers = df[center_col].to_numpy(dtype=np.float64)
    elif starts_a is not None and ends_a is not None:
        centers = 0.5 * (starts_a + ends_a)
    else:
        raise ValueError(
            "Anchors must contain either a 'center'/'pos' column or both 'start' and 'end' columns."
        )

    strand_col = None
    for cand in ("strand", "orientation", "dir"):
        if cand in col_map:
            strand_col = col_map[cand]
            break

    if strand_col is None:
        strands = np.ones(len(df), dtype=np.int8)
    else:
        raw_st = df[strand_col].to_numpy()
        if np.issubdtype(raw_st.dtype, np.number):
            strands = np.where(raw_st < 0, -1, 1).astype(np.int8)
        else:
            st_str = pd.Series(raw_st).fillna("+").astype(str).str.strip().to_numpy()
            strands = np.where((st_str == "-") | (st_str == "-1"), -1, 1).astype(np.int8)

    chroms = df[chrom_col].astype(str).to_numpy(dtype=object)
    region_index = df.index.to_numpy()

    chrom_groups: dict[str, np.ndarray] = {}
    for c, idxs in pd.Series(np.arange(len(df), dtype=np.int64)).groupby(chroms, sort=False):
        chrom_groups[str(c)] = np.ascontiguousarray(idxs.to_numpy(dtype=np.int64))

    return NormalizedAnchors(
        chroms=chroms,
        centers=np.ascontiguousarray(centers, dtype=np.float64),
        strands=np.ascontiguousarray(strands, dtype=np.int8),
        region_index=region_index,
        chrom_groups=chrom_groups,
        motif_starts=starts_a,
        motif_ends=ends_a,
    )


def _warn_missing_chroms(store: FragmentStore, norm: NormalizedAnchors) -> None:
    """Warn when anchors sit on chromosomes the store has no fragments for.

    The usual cause is a naming mismatch (``chr1`` vs ``1``), which otherwise
    produces silent all-zero rows.
    """
    if norm.n_anchors == 0:
        return
    missing = [
        c for c in norm.chrom_groups if c not in store.starts or len(store.starts[c]) == 0
    ]
    if not missing:
        return
    n_missing = int(sum(len(norm.chrom_groups[c]) for c in missing))
    shown = ", ".join(repr(c) for c in missing[:5]) + (" ..." if len(missing) > 5 else "")
    store_chroms = ", ".join(repr(c) for c in store.chroms[:5]) + (
        " ..." if len(store.chroms) > 5 else ""
    )
    msg = (
        f"{n_missing} of {norm.n_anchors} anchors are on chromosomes with no fragments in the "
        f"store ({shown}); their rows will be zero. Store chromosomes: {store_chroms}."
    )
    if n_missing == norm.n_anchors:
        msg += " No anchor matched: check chromosome naming ('chr1' vs '1')."
    warnings.warn(msg, UserWarning, stacklevel=3)


# ----------------------------------------------------------------------
# Binary search helper (identical logic in Numba and pure Python/NumPy)
# ----------------------------------------------------------------------


@njit(cache=True)
def _lower_bound_int32(arr: np.ndarray, val: int) -> int:
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound_int32(arr: np.ndarray, val: int) -> int:
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------
# Numba kernels: V-plot 2D accumulation
# ----------------------------------------------------------------------


@njit(cache=True)
def _vplot_range_kernel(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_strands: np.ndarray,
    has_strands: bool,
    max_frag_len: int,
    centers: np.ndarray,
    anchor_strands: np.ndarray,
    start_idx: int,
    end_idx: int,
    upstream: float,
    downstream: float,
    pos_bin: float,
    n_pos: int,
    min_len: int,
    max_len: int,
    len_bin: int,
    n_len: int,
    feature_code: int,
    motif_lo: np.ndarray,
    motif_hi: np.ndarray,
    overlap_code: int,
    out_grid: np.ndarray,
) -> None:
    inv_pos_bin = 1.0 / pos_bin
    for i in range(start_idx, end_idx):
        c = centers[i]
        a_st = anchor_strands[i]
        if a_st >= 0:
            g_lo = c - upstream
            g_hi = c + downstream
        else:
            g_lo = c - downstream
            g_hi = c + upstream

        lo_val = int(math.floor(g_lo)) - max_frag_len - 1
        hi_val = int(math.ceil(g_hi)) + 1
        lo = _lower_bound_int32(frag_starts, lo_val)
        hi = _upper_bound_int32(frag_starts, hi_val)

        for j in range(lo, hi):
            s = frag_starts[j]
            e = frag_ends[j]
            fl = e - s
            if fl < min_len or fl >= max_len:
                continue
            li = (fl - min_len) // len_bin
            if li < 0 or li >= n_len:
                continue
            if overlap_code != OVERLAP_OFF and not _overlap_ok(s, e, motif_lo[i], motif_hi[i], overlap_code):
                continue

            if feature_code == FEATURE_MIDPOINT:
                mid = 0.5 * (s + e)
                rel = (mid - c) if a_st >= 0 else (c - mid)
                pi = int(math.floor((rel + upstream) * inv_pos_bin))
                if 0 <= pi < n_pos:
                    out_grid[li, pi] += 1.0

            elif feature_code == FEATURE_CUTS:
                if has_strands:
                    rel = (s - c) if a_st >= 0 else (c - s)
                    pi = int(math.floor((rel + upstream) * inv_pos_bin))
                    if 0 <= pi < n_pos:
                        out_grid[li, pi] += 1.0
                else:
                    if a_st >= 0:
                        rel1 = s - c
                        rel2 = e - c
                    else:
                        rel1 = c - e
                        rel2 = c - s
                    pi1 = int(math.floor((rel1 + upstream) * inv_pos_bin))
                    if 0 <= pi1 < n_pos:
                        out_grid[li, pi1] += 1.0
                    pi2 = int(math.floor((rel2 + upstream) * inv_pos_bin))
                    if 0 <= pi2 < n_pos:
                        out_grid[li, pi2] += 1.0

            elif feature_code == FEATURE_CUTS_PLUS:
                if has_strands:
                    if frag_strands[j] * a_st > 0:
                        rel = (s - c) if a_st >= 0 else (c - s)
                        pi = int(math.floor((rel + upstream) * inv_pos_bin))
                        if 0 <= pi < n_pos:
                            out_grid[li, pi] += 1.0
                else:
                    rel = (s - c) if a_st >= 0 else (c - e)
                    pi = int(math.floor((rel + upstream) * inv_pos_bin))
                    if 0 <= pi < n_pos:
                        out_grid[li, pi] += 1.0

            elif feature_code == FEATURE_CUTS_MINUS:
                if has_strands:
                    if frag_strands[j] * a_st < 0:
                        rel = (s - c) if a_st >= 0 else (c - s)
                        pi = int(math.floor((rel + upstream) * inv_pos_bin))
                        if 0 <= pi < n_pos:
                            out_grid[li, pi] += 1.0
                else:
                    rel = (e - c) if a_st >= 0 else (c - s)
                    pi = int(math.floor((rel + upstream) * inv_pos_bin))
                    if 0 <= pi < n_pos:
                        out_grid[li, pi] += 1.0

            elif feature_code in (FEATURE_CUTS_PROXIMAL, FEATURE_CUTS_DISTAL):
                r_l = (s - c) if a_st >= 0 else (c - e)
                r_r = (e - c) if a_st >= 0 else (c - s)
                # r_l + r_r >= 0 means the left end is at least as close; ties go to the left end.
                left_near = r_l + r_r >= 0
                if feature_code == FEATURE_CUTS_PROXIMAL:
                    rel = r_l if left_near else r_r
                else:
                    rel = r_r if left_near else r_l
                pi = int(math.floor((rel + upstream) * inv_pos_bin))
                if 0 <= pi < n_pos:
                    out_grid[li, pi] += 1.0


@njit(parallel=True, cache=True)
def _vplot_parallel_kernel(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_strands: np.ndarray,
    has_strands: bool,
    max_frag_len: int,
    centers: np.ndarray,
    anchor_strands: np.ndarray,
    n_chunks: int,
    upstream: float,
    downstream: float,
    pos_bin: float,
    n_pos: int,
    min_len: int,
    max_len: int,
    len_bin: int,
    n_len: int,
    feature_code: int,
    motif_lo: np.ndarray,
    motif_hi: np.ndarray,
    overlap_code: int,
) -> np.ndarray:
    n_anchors = centers.shape[0]
    chunk_grids = np.zeros((n_chunks, n_len, n_pos), dtype=np.float64)
    chunk_size = (n_anchors + n_chunks - 1) // n_chunks
    for ch in prange(n_chunks):
        s_idx = ch * chunk_size
        e_idx = s_idx + chunk_size
        if e_idx > n_anchors:
            e_idx = n_anchors
        if s_idx < e_idx:
            _vplot_range_kernel(
                frag_starts,
                frag_ends,
                frag_strands,
                has_strands,
                max_frag_len,
                centers,
                anchor_strands,
                s_idx,
                e_idx,
                upstream,
                downstream,
                pos_bin,
                n_pos,
                min_len,
                max_len,
                len_bin,
                n_len,
                feature_code,
                motif_lo,
                motif_hi,
                overlap_code,
                chunk_grids[ch],
            )
    out = np.zeros((n_len, n_pos), dtype=np.float64)
    for ch in range(n_chunks):
        for li in range(n_len):
            for pi in range(n_pos):
                out[li, pi] += chunk_grids[ch, li, pi]
    return out


# ----------------------------------------------------------------------
# Numba kernel: per-anchor matrix (n_anchors, n_bins)
# ----------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _matrix_parallel_kernel(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_strands: np.ndarray,
    has_strands: bool,
    max_frag_len: int,
    centers: np.ndarray,
    anchor_strands: np.ndarray,
    anchor_out_rows: np.ndarray,
    upstream: float,
    downstream: float,
    bin_size: float,
    n_bins: int,
    min_len: int,
    max_len: int,
    feature_code: int,
    motif_lo: np.ndarray,
    motif_hi: np.ndarray,
    overlap_code: int,
    out_matrix: np.ndarray,
) -> None:
    inv_bin = 1.0 / bin_size
    n_local = centers.shape[0]
    for i in prange(n_local):
        row = anchor_out_rows[i]
        c = centers[i]
        a_st = anchor_strands[i]
        if a_st >= 0:
            g_lo = c - upstream
            g_hi = c + downstream
        else:
            g_lo = c - downstream
            g_hi = c + upstream

        lo_val = int(math.floor(g_lo)) - max_frag_len - 1
        hi_val = int(math.ceil(g_hi)) + 1
        lo = _lower_bound_int32(frag_starts, lo_val)
        hi = _upper_bound_int32(frag_starts, hi_val)

        for j in range(lo, hi):
            s = frag_starts[j]
            e = frag_ends[j]
            fl = e - s
            if fl < min_len or fl > max_len:
                continue
            if overlap_code != OVERLAP_OFF and not _overlap_ok(s, e, motif_lo[i], motif_hi[i], overlap_code):
                continue

            if feature_code == FEATURE_MIDPOINT:
                mid = 0.5 * (s + e)
                rel = (mid - c) if a_st >= 0 else (c - mid)
                bi = int(math.floor((rel + upstream) * inv_bin))
                if 0 <= bi < n_bins:
                    out_matrix[row, bi] += 1.0

            elif feature_code == FEATURE_CUTS:
                if has_strands:
                    rel = (s - c) if a_st >= 0 else (c - s)
                    bi = int(math.floor((rel + upstream) * inv_bin))
                    if 0 <= bi < n_bins:
                        out_matrix[row, bi] += 1.0
                else:
                    if a_st >= 0:
                        rel1 = s - c
                        rel2 = e - c
                    else:
                        rel1 = c - e
                        rel2 = c - s
                    bi1 = int(math.floor((rel1 + upstream) * inv_bin))
                    if 0 <= bi1 < n_bins:
                        out_matrix[row, bi1] += 1.0
                    bi2 = int(math.floor((rel2 + upstream) * inv_bin))
                    if 0 <= bi2 < n_bins:
                        out_matrix[row, bi2] += 1.0

            elif feature_code == FEATURE_CUTS_PLUS:
                if has_strands:
                    if frag_strands[j] * a_st > 0:
                        rel = (s - c) if a_st >= 0 else (c - s)
                        bi = int(math.floor((rel + upstream) * inv_bin))
                        if 0 <= bi < n_bins:
                            out_matrix[row, bi] += 1.0
                else:
                    rel = (s - c) if a_st >= 0 else (c - e)
                    bi = int(math.floor((rel + upstream) * inv_bin))
                    if 0 <= bi < n_bins:
                        out_matrix[row, bi] += 1.0

            elif feature_code == FEATURE_CUTS_MINUS:
                if has_strands:
                    if frag_strands[j] * a_st < 0:
                        rel = (s - c) if a_st >= 0 else (c - s)
                        bi = int(math.floor((rel + upstream) * inv_bin))
                        if 0 <= bi < n_bins:
                            out_matrix[row, bi] += 1.0
                else:
                    rel = (e - c) if a_st >= 0 else (c - s)
                    bi = int(math.floor((rel + upstream) * inv_bin))
                    if 0 <= bi < n_bins:
                        out_matrix[row, bi] += 1.0

            elif feature_code in (FEATURE_CUTS_PROXIMAL, FEATURE_CUTS_DISTAL):
                r_l = (s - c) if a_st >= 0 else (c - e)
                r_r = (e - c) if a_st >= 0 else (c - s)
                left_near = r_l + r_r >= 0
                if feature_code == FEATURE_CUTS_PROXIMAL:
                    rel = r_l if left_near else r_r
                else:
                    rel = r_r if left_near else r_l
                bi = int(math.floor((rel + upstream) * inv_bin))
                if 0 <= bi < n_bins:
                    out_matrix[row, bi] += 1.0

            elif feature_code == FEATURE_COVERAGE:
                if a_st >= 0:
                    r_start = s - c
                    r_end = e - c
                else:
                    r_start = c - e
                    r_end = c - s
                c_start = r_start if r_start > -upstream else -upstream
                c_end = r_end if r_end < downstream else downstream
                if c_end > c_start:
                    b0 = int(math.floor((c_start + upstream) * inv_bin))
                    b1 = int(math.ceil((c_end + upstream) * inv_bin)) - 1
                    if b0 < 0:
                        b0 = 0
                    if b1 >= n_bins:
                        b1 = n_bins - 1
                    for b in range(b0, b1 + 1):
                        bin_lo = -upstream + b * bin_size
                        bin_hi = bin_lo + bin_size
                        ov_lo = c_start if c_start > bin_lo else bin_lo
                        ov_hi = c_end if c_end < bin_hi else bin_hi
                        if ov_hi > ov_lo:
                            out_matrix[row, b] += (ov_hi - ov_lo) * inv_bin


# ----------------------------------------------------------------------
# Pure-NumPy fallback implementations (exact parity with Numba kernels)
# ----------------------------------------------------------------------


def _proximal_distal_rel(
    s: np.ndarray, e: np.ndarray, c: float, a_st: int, feature_code: int
) -> np.ndarray:
    """Anchor-oriented position of the near (or far) end of each fragment."""
    r_l = (s - c) if a_st >= 0 else (c - e)
    r_r = (e - c) if a_st >= 0 else (c - s)
    left_near = (r_l + r_r) >= 0
    if feature_code == FEATURE_CUTS_PROXIMAL:
        return np.where(left_near, r_l, r_r)
    return np.where(left_near, r_r, r_l)


def _vplot_chrom_numpy(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_strands: np.ndarray | None,
    max_frag_len: int,
    centers: np.ndarray,
    anchor_strands: np.ndarray,
    upstream: float,
    downstream: float,
    pos_bin: float,
    n_pos: int,
    min_len: int,
    max_len: int,
    len_bin: int,
    n_len: int,
    feature_code: int,
    motif_lo: np.ndarray,
    motif_hi: np.ndarray,
    overlap_code: int,
    out_grid: np.ndarray,
) -> None:
    g_lo = np.where(anchor_strands >= 0, centers - upstream, centers - downstream)
    g_hi = np.where(anchor_strands >= 0, centers + downstream, centers + upstream)
    lo_idxs = np.searchsorted(frag_starts, np.floor(g_lo).astype(np.int64) - max_frag_len - 1, side="left")
    hi_idxs = np.searchsorted(frag_starts, np.ceil(g_hi).astype(np.int64) + 1, side="right")

    inv_pos_bin = 1.0 / pos_bin
    has_strands = frag_strands is not None

    for i in range(len(centers)):
        lo = int(lo_idxs[i])
        hi = int(hi_idxs[i])
        if lo >= hi:
            continue
        s = frag_starts[lo:hi].astype(np.float64)
        e = frag_ends[lo:hi].astype(np.float64)
        fl = frag_ends[lo:hi] - frag_starts[lo:hi]
        valid_len = (fl >= min_len) & (fl < max_len)
        ov = _overlap_mask(s, e, float(motif_lo[i]), float(motif_hi[i]), overlap_code)
        if ov is not None:
            valid_len &= ov
        if not np.any(valid_len):
            continue
        s = s[valid_len]
        e = e[valid_len]
        li = ((fl[valid_len] - min_len) // len_bin).astype(np.int64)
        fst = frag_strands[lo:hi][valid_len] if has_strands and frag_strands is not None else None

        c = float(centers[i])
        a_st = int(anchor_strands[i])

        if feature_code == FEATURE_MIDPOINT:
            mid = 0.5 * (s + e)
            rel = (mid - c) if a_st >= 0 else (c - mid)
            pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
            ok = (pi >= 0) & (pi < n_pos)
            if np.any(ok):
                np.add.at(out_grid, (li[ok], pi[ok]), 1.0)

        elif feature_code == FEATURE_CUTS:
            if has_strands:
                rel = (s - c) if a_st >= 0 else (c - s)
                pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                ok = (pi >= 0) & (pi < n_pos)
                if np.any(ok):
                    np.add.at(out_grid, (li[ok], pi[ok]), 1.0)
            else:
                rel1 = (s - c) if a_st >= 0 else (c - e)
                rel2 = (e - c) if a_st >= 0 else (c - s)
                for rel in (rel1, rel2):
                    pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                    ok = (pi >= 0) & (pi < n_pos)
                    if np.any(ok):
                        np.add.at(out_grid, (li[ok], pi[ok]), 1.0)

        elif feature_code == FEATURE_CUTS_PLUS:
            if has_strands and fst is not None:
                mask = (fst * a_st) > 0
                if np.any(mask):
                    rel = (s[mask] - c) if a_st >= 0 else (c - s[mask])
                    pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                    ok = (pi >= 0) & (pi < n_pos)
                    if np.any(ok):
                        np.add.at(out_grid, (li[mask][ok], pi[ok]), 1.0)
            else:
                rel = (s - c) if a_st >= 0 else (c - e)
                pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                ok = (pi >= 0) & (pi < n_pos)
                if np.any(ok):
                    np.add.at(out_grid, (li[ok], pi[ok]), 1.0)

        elif feature_code == FEATURE_CUTS_MINUS:
            if has_strands and fst is not None:
                mask = (fst * a_st) < 0
                if np.any(mask):
                    rel = (s[mask] - c) if a_st >= 0 else (c - s[mask])
                    pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                    ok = (pi >= 0) & (pi < n_pos)
                    if np.any(ok):
                        np.add.at(out_grid, (li[mask][ok], pi[ok]), 1.0)
            else:
                rel = (e - c) if a_st >= 0 else (c - s)
                pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
                ok = (pi >= 0) & (pi < n_pos)
                if np.any(ok):
                    np.add.at(out_grid, (li[ok], pi[ok]), 1.0)

        elif feature_code in (FEATURE_CUTS_PROXIMAL, FEATURE_CUTS_DISTAL):
            rel = _proximal_distal_rel(s, e, c, a_st, feature_code)
            pi = np.floor((rel + upstream) * inv_pos_bin).astype(np.int64)
            ok = (pi >= 0) & (pi < n_pos)
            if np.any(ok):
                np.add.at(out_grid, (li[ok], pi[ok]), 1.0)


def _matrix_chrom_numpy(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    frag_strands: np.ndarray | None,
    max_frag_len: int,
    centers: np.ndarray,
    anchor_strands: np.ndarray,
    anchor_out_rows: np.ndarray,
    upstream: float,
    downstream: float,
    bin_size: float,
    n_bins: int,
    min_len: int,
    max_len: int,
    feature_code: int,
    motif_lo: np.ndarray,
    motif_hi: np.ndarray,
    overlap_code: int,
    out_matrix: np.ndarray,
) -> None:
    g_lo = np.where(anchor_strands >= 0, centers - upstream, centers - downstream)
    g_hi = np.where(anchor_strands >= 0, centers + downstream, centers + upstream)
    lo_idxs = np.searchsorted(frag_starts, np.floor(g_lo).astype(np.int64) - max_frag_len - 1, side="left")
    hi_idxs = np.searchsorted(frag_starts, np.ceil(g_hi).astype(np.int64) + 1, side="right")

    inv_bin = 1.0 / bin_size
    has_strands = frag_strands is not None

    for i in range(len(centers)):
        lo = int(lo_idxs[i])
        hi = int(hi_idxs[i])
        if lo >= hi:
            continue
        fl = frag_ends[lo:hi] - frag_starts[lo:hi]
        valid_len = (fl >= min_len) & (fl <= max_len)
        ov = _overlap_mask(frag_starts[lo:hi], frag_ends[lo:hi], float(motif_lo[i]), float(motif_hi[i]), overlap_code)
        if ov is not None:
            valid_len &= ov
        if not np.any(valid_len):
            continue
        s = frag_starts[lo:hi][valid_len].astype(np.float64)
        e = frag_ends[lo:hi][valid_len].astype(np.float64)
        fst = frag_strands[lo:hi][valid_len] if has_strands and frag_strands is not None else None

        row = int(anchor_out_rows[i])
        c = float(centers[i])
        a_st = int(anchor_strands[i])

        if feature_code == FEATURE_MIDPOINT:
            mid = 0.5 * (s + e)
            rel = (mid - c) if a_st >= 0 else (c - mid)
            bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
            ok = (bi >= 0) & (bi < n_bins)
            if np.any(ok):
                np.add.at(out_matrix[row], bi[ok], 1.0)

        elif feature_code == FEATURE_CUTS:
            if has_strands:
                rel = (s - c) if a_st >= 0 else (c - s)
                bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                ok = (bi >= 0) & (bi < n_bins)
                if np.any(ok):
                    np.add.at(out_matrix[row], bi[ok], 1.0)
            else:
                rel1 = (s - c) if a_st >= 0 else (c - e)
                rel2 = (e - c) if a_st >= 0 else (c - s)
                for rel in (rel1, rel2):
                    bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                    ok = (bi >= 0) & (bi < n_bins)
                    if np.any(ok):
                        np.add.at(out_matrix[row], bi[ok], 1.0)

        elif feature_code == FEATURE_CUTS_PLUS:
            if has_strands and fst is not None:
                mask = (fst * a_st) > 0
                if np.any(mask):
                    rel = (s[mask] - c) if a_st >= 0 else (c - s[mask])
                    bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                    ok = (bi >= 0) & (bi < n_bins)
                    if np.any(ok):
                        np.add.at(out_matrix[row], bi[ok], 1.0)
            else:
                rel = (s - c) if a_st >= 0 else (c - e)
                bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                ok = (bi >= 0) & (bi < n_bins)
                if np.any(ok):
                    np.add.at(out_matrix[row], bi[ok], 1.0)

        elif feature_code == FEATURE_CUTS_MINUS:
            if has_strands and fst is not None:
                mask = (fst * a_st) < 0
                if np.any(mask):
                    rel = (s[mask] - c) if a_st >= 0 else (c - s[mask])
                    bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                    ok = (bi >= 0) & (bi < n_bins)
                    if np.any(ok):
                        np.add.at(out_matrix[row], bi[ok], 1.0)
            else:
                rel = (e - c) if a_st >= 0 else (c - s)
                bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
                ok = (bi >= 0) & (bi < n_bins)
                if np.any(ok):
                    np.add.at(out_matrix[row], bi[ok], 1.0)

        elif feature_code in (FEATURE_CUTS_PROXIMAL, FEATURE_CUTS_DISTAL):
            rel = _proximal_distal_rel(s, e, c, a_st, feature_code)
            bi = np.floor((rel + upstream) * inv_bin).astype(np.int64)
            ok = (bi >= 0) & (bi < n_bins)
            if np.any(ok):
                np.add.at(out_matrix[row], bi[ok], 1.0)

        elif feature_code == FEATURE_COVERAGE:
            r_start = (s - c) if a_st >= 0 else (c - e)
            r_end = (e - c) if a_st >= 0 else (c - s)
            c_start = np.maximum(r_start, -upstream)
            c_end = np.minimum(r_end, downstream)
            ov_mask = c_end > c_start
            if not np.any(ov_mask):
                continue
            cs = c_start[ov_mask]
            ce = c_end[ov_mask]
            b0s = np.maximum(0, np.floor((cs + upstream) * inv_bin).astype(np.int64))
            b1s = np.minimum(n_bins - 1, np.ceil((ce + upstream) * inv_bin).astype(np.int64) - 1)
            for k in range(len(cs)):
                b0 = int(b0s[k])
                b1 = int(b1s[k])
                lo_k = float(cs[k])
                hi_k = float(ce[k])
                for b in range(b0, b1 + 1):
                    bin_lo = -upstream + b * bin_size
                    bin_hi = bin_lo + bin_size
                    ov = min(hi_k, bin_hi) - max(lo_k, bin_lo)
                    if ov > 0.0:
                        out_matrix[row, b] += ov * inv_bin


# ----------------------------------------------------------------------
# Public extraction entry points
# ----------------------------------------------------------------------


def accumulate_vplot(
    store: FragmentStore,
    anchors: Any,
    upstream: int,
    downstream: int,
    min_len: int,
    max_len: int,
    pos_bin: int = 1,
    len_bin: int = 1,
    feature: str = "midpoint",
    use_numba: bool | None = None,
    motif_overlap: str | None = None,
) -> tuple[np.ndarray, NormalizedAnchors]:
    """Stream fragments around ``anchors`` directly into a 2D ``(n_len, n_pos)`` grid.

    Length bins are half-open ``[min_len, max_len)`` with uniform width ``len_bin``.
    Position bins are half-open ``[-upstream, downstream)`` with uniform width ``pos_bin``.
    ``motif_overlap`` keeps only fragments that overlap (``"any"``), contain
    (``"full"``) or miss (``"none"``) each anchor's motif interval.
    """
    if pos_bin <= 0 or len_bin <= 0:
        raise ValueError("`pos_bin` and `len_bin` must be positive integers.")
    total_span = upstream + downstream
    if total_span <= 0 or total_span % pos_bin != 0:
        raise ValueError(
            f"Total position span ({total_span}) must be a positive multiple of pos_bin ({pos_bin})."
        )
    len_span = max_len - min_len
    if len_span <= 0 or len_span % len_bin != 0:
        raise ValueError(
            f"Length span ({max_len} - {min_len} = {len_span}) must be a positive multiple of len_bin ({len_bin})."
        )

    fcode = feature_to_code(feature)
    if fcode == FEATURE_COVERAGE:
        raise ValueError("V-plot accumulation does not support feature='coverage'; use 'midpoint' or 'cuts'.")
    ocode = motif_overlap_to_code(motif_overlap)
    _check_stranded_support(store, fcode, ocode)

    norm = normalize_anchors(anchors)
    n_pos = total_span // pos_bin
    n_len = len_span // len_bin
    grid = np.zeros((n_len, n_pos), dtype=np.float64)

    if norm.n_anchors == 0 or store.n_fragments == 0:
        return grid, norm
    _warn_missing_chroms(store, norm)

    run_numba = HAS_NUMBA if use_numba is None else bool(use_numba and HAS_NUMBA)
    empty_strands = np.empty(0, dtype=np.int8)
    m_lo_all, m_hi_all = norm.motif_intervals()

    for chrom, idxs in norm.chrom_groups.items():
        if chrom not in store.starts or len(store.starts[chrom]) == 0:
            continue
        f_starts = store.starts[chrom]
        f_ends = store.ends[chrom]
        has_strands = store.strands is not None
        f_strands = store.strands[chrom] if store.strands is not None else empty_strands
        m_len = store.max_len[chrom]

        c_centers = np.ascontiguousarray(norm.centers[idxs], dtype=np.float64)
        c_strands = np.ascontiguousarray(norm.strands[idxs], dtype=np.int8)
        c_mlo = np.ascontiguousarray(m_lo_all[idxs], dtype=np.float64)
        c_mhi = np.ascontiguousarray(m_hi_all[idxs], dtype=np.float64)

        if run_numba:
            n_local = len(c_centers)
            if n_local >= 256:
                n_chunks = min(max(1, int(get_num_threads())), max(1, n_local // 128))
                grid += _vplot_parallel_kernel(
                    f_starts,
                    f_ends,
                    f_strands,
                    has_strands,
                    m_len,
                    c_centers,
                    c_strands,
                    n_chunks,
                    float(upstream),
                    float(downstream),
                    float(pos_bin),
                    n_pos,
                    int(min_len),
                    int(max_len),
                    int(len_bin),
                    n_len,
                    fcode,
                    c_mlo,
                    c_mhi,
                    ocode,
                )
            else:
                _vplot_range_kernel(
                    f_starts,
                    f_ends,
                    f_strands,
                    has_strands,
                    m_len,
                    c_centers,
                    c_strands,
                    0,
                    n_local,
                    float(upstream),
                    float(downstream),
                    float(pos_bin),
                    n_pos,
                    int(min_len),
                    int(max_len),
                    int(len_bin),
                    n_len,
                    fcode,
                    c_mlo,
                    c_mhi,
                    ocode,
                    grid,
                )
        else:
            _vplot_chrom_numpy(
                f_starts,
                f_ends,
                store.strands[chrom] if store.strands is not None else None,
                m_len,
                c_centers,
                c_strands,
                float(upstream),
                float(downstream),
                float(pos_bin),
                n_pos,
                int(min_len),
                int(max_len),
                int(len_bin),
                n_len,
                fcode,
                c_mlo,
                c_mhi,
                ocode,
                grid,
            )

    return grid, norm


def accumulate_matrix(
    store: FragmentStore,
    anchors: Any,
    upstream: int,
    downstream: int,
    bin_size: int = 10,
    feature: str = "midpoint",
    len_range: tuple[int, int] | None = None,
    use_numba: bool | None = None,
    motif_overlap: str | None = None,
) -> tuple[np.ndarray, NormalizedAnchors]:
    """Extract a ``(n_anchors, n_bins)`` signal matrix around ``anchors``.

    ``len_range = (min_len, max_len)`` uses inclusive bounds ``min_len <= length <= max_len``.
    ``motif_overlap`` filters fragments as in [`accumulate_vplot`][pyvplot.extract.accumulate_vplot].
    """
    if bin_size <= 0:
        raise ValueError("`bin_size` must be a positive integer.")
    total_span = upstream + downstream
    if total_span <= 0 or total_span % bin_size != 0:
        raise ValueError(
            f"Total window ({upstream} + {downstream} = {total_span}) must be a positive multiple of bin_size ({bin_size})."
        )

    fcode = feature_to_code(feature)
    ocode = motif_overlap_to_code(motif_overlap)
    _check_stranded_support(store, fcode, ocode)
    norm = normalize_anchors(anchors)
    n_bins = total_span // bin_size
    out = np.zeros((norm.n_anchors, n_bins), dtype=np.float64)

    if norm.n_anchors == 0 or store.n_fragments == 0:
        return out, norm
    _warn_missing_chroms(store, norm)

    if len_range is None:
        min_len = 0
        max_len = 2_147_483_647
    else:
        min_len, max_len = int(len_range[0]), int(len_range[1])
        if min_len > max_len:
            raise ValueError(f"Invalid len_range {len_range!r}: min_len > max_len.")

    run_numba = HAS_NUMBA if use_numba is None else bool(use_numba and HAS_NUMBA)
    empty_strands = np.empty(0, dtype=np.int8)
    m_lo_all, m_hi_all = norm.motif_intervals()

    for chrom, idxs in norm.chrom_groups.items():
        if chrom not in store.starts or len(store.starts[chrom]) == 0:
            continue
        f_starts = store.starts[chrom]
        f_ends = store.ends[chrom]
        has_strands = store.strands is not None
        f_strands = store.strands[chrom] if store.strands is not None else empty_strands
        m_len = store.max_len[chrom]

        c_centers = np.ascontiguousarray(norm.centers[idxs], dtype=np.float64)
        c_strands = np.ascontiguousarray(norm.strands[idxs], dtype=np.int8)
        c_mlo = np.ascontiguousarray(m_lo_all[idxs], dtype=np.float64)
        c_mhi = np.ascontiguousarray(m_hi_all[idxs], dtype=np.float64)

        if run_numba:
            _matrix_parallel_kernel(
                f_starts,
                f_ends,
                f_strands,
                has_strands,
                m_len,
                c_centers,
                c_strands,
                idxs,
                float(upstream),
                float(downstream),
                float(bin_size),
                n_bins,
                min_len,
                max_len,
                fcode,
                c_mlo,
                c_mhi,
                ocode,
                out,
            )
        else:
            _matrix_chrom_numpy(
                f_starts,
                f_ends,
                store.strands[chrom] if store.strands is not None else None,
                m_len,
                c_centers,
                c_strands,
                idxs,
                float(upstream),
                float(downstream),
                float(bin_size),
                n_bins,
                min_len,
                max_len,
                fcode,
                c_mlo,
                c_mhi,
                ocode,
                out,
            )

    return out, norm


def extract_events(
    store: FragmentStore,
    anchors: Any,
    flank: int = 300,
    upstream: int | None = None,
    downstream: int | None = None,
    len_range: tuple[int, int] | str | None = None,
    overlap_mode: str = "midpoint",
    motif_overlap: str | None = None,
) -> pd.DataFrame:
    """Extract a compact per-fragment event table around ``anchors``.

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to query.
    anchors:
        Anchor table (DataFrame, dict, or BED path).
    flank:
        Symmetric half-window (bp) used when ``upstream``/``downstream`` are omitted.
    upstream, downstream:
        Optional asymmetric window bounds (bp).
    len_range:
        Optional inclusive ``(min_len, max_len)`` fragment length filter, or the
        name of a size bin of the store's assay (``"tf"``, ``"nfr"``, ...).
    overlap_mode:
        Which fragments to include:
        - ``"midpoint"``: fragments whose midpoint falls in ``[-upstream, downstream)``
        - ``"any"``: fragments overlapping ``[-upstream, downstream)`` at any base
        - ``"cuts"``: fragments with at least one cut site in ``[-upstream, downstream)``
    motif_overlap:
        Optional filter on each fragment against the anchor's motif interval
        (BED start/end, or ``[center - 0.5, center + 0.5)`` for point anchors),
        applied on top of ``overlap_mode``:
        - ``None`` (default): no filter
        - ``"any"``: fragment overlaps the motif by at least 1 bp
        - ``"full"``: fragment contains the whole motif
        - ``"none"``: fragment does not overlap the motif

    Returns
    -------
    pandas.DataFrame
        Columns: ``anchor_idx``, ``rel_mid``, ``rel_left``, ``rel_right``,
        ``frag_len``. Unstranded stores add ``rel_proximal`` / ``rel_distal``
        (the end nearer to / farther from the anchor centre; ties go to
        ``rel_left``) and ``motif_overlap`` (``"full"``, ``"partial"`` or
        ``"none"``). Stranded stores add ``frag_strand`` instead.
    """
    up = float(flank if upstream is None else upstream)
    dn = float(flank if downstream is None else downstream)
    if overlap_mode not in {"midpoint", "any", "cuts"}:
        raise ValueError(
            f"Invalid overlap_mode {overlap_mode!r}; expected 'midpoint', 'any', or 'cuts'."
        )

    ocode = motif_overlap_to_code(motif_overlap)
    _check_stranded_support(store, FEATURE_CUTS, ocode)

    resolved_len = resolve_len_range(store.assay, len_range)
    norm = normalize_anchors(anchors)
    if norm.n_anchors == 0 or store.n_fragments == 0:
        return pd.DataFrame(columns=_event_columns(store.is_stranded))
    _warn_missing_chroms(store, norm)
    m_lo_all, m_hi_all = norm.motif_intervals()

    min_len = 0 if resolved_len is None else resolved_len[0]
    max_len = 2_147_483_647 if resolved_len is None else resolved_len[1]

    a_idxs_list: list[np.ndarray] = []
    rel_mid_list: list[np.ndarray] = []
    rel_left_list: list[np.ndarray] = []
    rel_right_list: list[np.ndarray] = []
    flen_list: list[np.ndarray] = []
    fstrand_list: list[np.ndarray] = []
    mov_list: list[np.ndarray] = []

    for chrom, idxs in norm.chrom_groups.items():
        if chrom not in store.starts or len(store.starts[chrom]) == 0:
            continue
        f_starts = store.starts[chrom]
        f_ends = store.ends[chrom]
        f_strands = store.strands[chrom] if store.strands is not None else None
        m_len = store.max_len[chrom]

        c_centers = norm.centers[idxs]
        c_strands = norm.strands[idxs]
        g_lo = np.where(c_strands >= 0, c_centers - up, c_centers - dn)
        g_hi = np.where(c_strands >= 0, c_centers + dn, c_centers + up)
        lo_idxs = np.searchsorted(f_starts, np.floor(g_lo).astype(np.int64) - m_len - 1, side="left")
        hi_idxs = np.searchsorted(f_starts, np.ceil(g_hi).astype(np.int64) + 1, side="right")

        for i in range(len(idxs)):
            lo = int(lo_idxs[i])
            hi = int(hi_idxs[i])
            if lo >= hi:
                continue
            s = f_starts[lo:hi]
            e = f_ends[lo:hi]
            fl = e - s
            valid = (fl >= min_len) & (fl <= max_len)
            if not np.any(valid):
                continue
            s_v = s[valid].astype(np.float64)
            e_v = e[valid].astype(np.float64)
            fl_v = fl[valid].astype(np.int32)
            c = float(c_centers[i])
            a_st = int(c_strands[i])

            if a_st >= 0:
                r_left = s_v - c
                r_right = e_v - c
            else:
                r_left = c - e_v
                r_right = c - s_v
            r_mid = 0.5 * (r_left + r_right)

            if overlap_mode == "midpoint":
                keep = (r_mid >= -up) & (r_mid < dn)
            elif overlap_mode == "any":
                keep = (r_right > -up) & (r_left < dn)
            else:  # cuts
                if f_strands is not None:
                    r_cut = (s_v - c) if a_st >= 0 else (c - s_v)
                    keep = (r_cut >= -up) & (r_cut < dn)
                else:
                    keep = ((r_left >= -up) & (r_left < dn)) | ((r_right >= -up) & (r_right < dn))

            ov = _overlap_mask(s_v, e_v, float(m_lo_all[idxs[i]]), float(m_hi_all[idxs[i]]), ocode)
            if ov is not None:
                keep &= ov
            if not np.any(keep):
                continue
            n_keep = int(np.sum(keep))
            a_idxs_list.append(np.full(n_keep, idxs[i], dtype=np.int64))
            rel_mid_list.append(r_mid[keep])
            rel_left_list.append(r_left[keep])
            rel_right_list.append(r_right[keep])
            flen_list.append(fl_v[keep])
            if f_strands is not None:
                eff_st = (f_strands[lo:hi][valid][keep] * a_st).astype(np.int8)
                fstrand_list.append(eff_st)
            else:
                m_lo = float(m_lo_all[idxs[i]])
                m_hi = float(m_hi_all[idxs[i]])
                sk, ek = s_v[keep], e_v[keep]
                mov = np.where(
                    (sk <= m_lo) & (ek >= m_hi),
                    "full",
                    np.where((sk < m_hi) & (ek > m_lo), "partial", "none"),
                )
                mov_list.append(mov)

    if not a_idxs_list:
        return pd.DataFrame(columns=_event_columns(store.is_stranded))

    data: dict[str, Any] = {
        "anchor_idx": np.concatenate(a_idxs_list),
        "rel_mid": np.concatenate(rel_mid_list),
        "rel_left": np.concatenate(rel_left_list),
        "rel_right": np.concatenate(rel_right_list),
        "frag_len": np.concatenate(flen_list),
    }
    if store.is_stranded:
        data["frag_strand"] = np.concatenate(fstrand_list)
    else:
        rl, rr = data["rel_left"], data["rel_right"]
        left_near = (rl + rr) >= 0
        data["rel_proximal"] = np.where(left_near, rl, rr)
        data["rel_distal"] = np.where(left_near, rr, rl)
        data["motif_overlap"] = pd.Categorical(
            np.concatenate(mov_list), categories=["full", "partial", "none"]
        )
    return pd.DataFrame(data)


def _event_columns(is_stranded: bool) -> list[str]:
    cols = ["anchor_idx", "rel_mid", "rel_left", "rel_right", "frag_len"]
    if is_stranded:
        return [*cols, "frag_strand"]
    return [*cols, "rel_proximal", "rel_distal", "motif_overlap"]
