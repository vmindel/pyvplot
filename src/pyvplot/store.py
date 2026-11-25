"""Compact, sorted per-chromosome fragment store for millisecond genomic queries."""

from __future__ import annotations

import gzip
import json
import warnings
from array import array
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd

from pyvplot.assays import AssayConfig, get_assay

if TYPE_CHECKING:
    from pyvplot.locus import LocusVPlot
    from pyvplot.matrices import CutSiteFootprint, SignalMatrix
    from pyvplot.pillars import LocusPillarScan
    from pyvplot.qc import PhasogramResult
    from pyvplot.vplot import VPlot

# SAM flags: unmapped (0x4) | mate unmapped (0x8) | secondary (0x100) | qcfail (0x200) | supplementary (0x800)
DEFAULT_PAIRED_EXCLUDE_FLAGS = 0x4 | 0x8 | 0x100 | 0x200 | 0x800
# Single-end SAM flags: unmapped (0x4) | secondary (0x100) | qcfail (0x200) | supplementary (0x800)
DEFAULT_SINGLE_EXCLUDE_FLAGS = 0x4 | 0x100 | 0x200 | 0x800
SAM_DUPLICATE_FLAG = 0x400

_STORE_FORMAT_VERSION = 1


_REGION_MODES = ("overlap", "midpoint")


def load_intervals(table: Any, what: str = "Regions") -> pd.DataFrame:
    """Read genomic intervals from a BED path (plain or ``.gz``) or a DataFrame.

    Returns a DataFrame with ``chrom``, ``start``, ``end`` and a default index. DataFrame
    columns are matched case-insensitively: ``chrom``/``chr``/``seqnames``/``chromosome``,
    ``start`` and ``end``. Intervals are 0-based, half-open like BED.
    """
    if isinstance(table, (str, Path)):
        path = Path(table)
        opener = gzip.open if path.suffix == ".gz" else open
        rows: list[tuple[str, int, int]] = []
        with opener(path, "rt", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line or line[0] == "#" or line.startswith(("track ", "browser ")):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    parts = line.split()
                if len(parts) < 3:
                    continue
                start, end = parse_int_columns(parts, (1, 2), path, lineno)
                rows.append((parts[0], start, end))
        out = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    else:
        df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        col_map = {str(c).lower(): c for c in df.columns}
        chrom_col = next(
            (col_map[k] for k in ("chrom", "chr", "seqnames", "chromosome") if k in col_map), None
        )
        if chrom_col is None or "start" not in col_map or "end" not in col_map:
            raise ValueError(
                f"{what} must have 'chrom' (or 'chr'), 'start' and 'end' columns; "
                f"got {list(df.columns)}."
            )
        out = pd.DataFrame(
            {
                "chrom": df[chrom_col].astype(str).to_numpy(),
                "start": df[col_map["start"]].to_numpy(dtype=np.int64),
                "end": df[col_map["end"]].to_numpy(dtype=np.int64),
            }
        )
    if len(out) and bool((out["end"] < out["start"]).any()):
        raise ValueError(f"{what} have end < start; intervals are [start, end).")
    return out


def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping or touching ``[start, end)`` intervals into sorted, disjoint ones."""
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    s = np.asarray(starts, dtype=np.int64)[order]
    e = np.asarray(ends, dtype=np.int64)[order]
    reach = np.maximum.accumulate(e)
    new_group = np.empty(len(s), dtype=np.bool_)
    new_group[0] = True
    new_group[1:] = s[1:] > reach[:-1]
    group_starts = np.flatnonzero(new_group)
    group_ends = np.append(group_starts[1:], len(s)) - 1
    return s[group_starts], reach[group_ends]


def _in_regions(
    frag_starts: np.ndarray,
    frag_ends: np.ndarray,
    reg_starts: np.ndarray,
    reg_ends: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Boolean mask of fragments that hit the (merged, sorted) regions."""
    fs = frag_starts.astype(np.int64, copy=False)
    fe = frag_ends.astype(np.int64, copy=False)
    if mode == "midpoint":
        mid = (fs + fe) // 2
        k = np.searchsorted(reg_starts, mid, side="right") - 1
        return (k >= 0) & (mid < reg_ends[np.maximum(k, 0)])
    # overlap: some region has start < frag_end and end > frag_start. Merged regions are
    # disjoint and sorted, so only the last one starting before the fragment end can qualify.
    k = np.searchsorted(reg_starts, fe, side="left") - 1
    return (k >= 0) & (reg_ends[np.maximum(k, 0)] > fs)


_PROGRESS_EVERY = 5_000_000


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


class _BamProgress:
    """Progress lines for ``from_bam(verbose=True)``: on each new chromosome and every N records."""

    def __init__(self, path: str, ref_names: tuple[str, ...], every: int = _PROGRESS_EVERY) -> None:
        import time

        self._time = time.perf_counter
        self.t0 = self._time()
        self.path = path
        self.ref_names = ref_names
        self.every = int(every)
        self.tid = -2
        self.seen: set[int] = set()
        self.next_at = self.every

    def elapsed(self) -> float:
        return self._time() - self.t0

    def log(self, msg: str) -> None:
        import sys

        print(f"[pyvplot.from_bam {self.elapsed():7.1f}s] {msg}", file=sys.stderr, flush=True)

    def update(self, n_seen: int, tid: int, n_kept: int) -> None:
        # Name-sorted / unsorted BAMs switch chromosome on almost every record, so a
        # chromosome is announced only the first time it is seen.
        due = n_seen >= self.next_at
        if due:
            self.next_at = (n_seen // self.every + 1) * self.every
        self.tid = tid
        new_chrom = tid not in self.seen
        if not (due or new_chrom):
            return
        self.seen.add(tid)
        chrom = self.ref_names[tid] if 0 <= tid < len(self.ref_names) else "*unmapped*"
        t = self.elapsed()
        rate = n_seen / t if t > 0 else 0.0
        prefix = "now on" if new_chrom else "on"
        self.log(
            f"{prefix} {chrom}: {_fmt_count(n_seen)} records read, "
            f"{_fmt_count(n_kept)} fragments kept ({_fmt_count(int(rate))} records/s)"
        )


class _ChromBuffers:
    """Compact per-chromosome accumulators (4 bytes per coordinate instead of ~36 for a list of ints)."""

    def __init__(self, stranded: bool) -> None:
        self.starts: dict[str, array] = defaultdict(lambda: array("i"))
        self.ends: dict[str, array] = defaultdict(lambda: array("i"))
        self.strands: dict[str, array] | None = defaultdict(lambda: array("b")) if stranded else None

    def add(self, chrom: str, start: int, end: int, strand: int = 1, count: int = 1) -> None:
        if count == 1:
            self.starts[chrom].append(start)
            self.ends[chrom].append(end)
            if self.strands is not None:
                self.strands[chrom].append(strand)
        else:
            self.starts[chrom].extend([start] * count)
            self.ends[chrom].extend([end] * count)
            if self.strands is not None:
                self.strands[chrom].extend([strand] * count)

    def to_numpy(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
        def _arr(a: array, dtype: Any) -> np.ndarray:
            return np.frombuffer(a, dtype=dtype).copy() if len(a) else np.empty(0, dtype=dtype)

        starts = {c: _arr(a, np.intc).astype(np.int32, copy=False) for c, a in self.starts.items()}
        ends = {c: _arr(a, np.intc).astype(np.int32, copy=False) for c, a in self.ends.items()}
        strands = (
            {c: _arr(a, np.byte).astype(np.int8, copy=False) for c, a in self.strands.items()}
            if self.strands is not None
            else None
        )
        return starts, ends, strands


def parse_int_columns(parts: list[str], cols: tuple[int, ...], path: Path | str, lineno: int) -> tuple[int, ...]:
    """Read integer columns of a BED-like line, naming the file and line when one is not a number."""
    try:
        return tuple(int(parts[c]) for c in cols)
    except (ValueError, IndexError) as exc:
        shown = "\t".join(parts)[:80]
        raise ValueError(
            f"{path}:{lineno}: cannot read integer column(s) {[c + 1 for c in cols]} from "
            f"{shown!r}. Header lines must start with '#', 'track' or 'browser'."
        ) from exc


def _dedup_sorted_chrom(
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Deduplicate identical coordinate entries in pre-sorted per-chromosome arrays."""
    n = len(starts)
    if n <= 1:
        return starts, ends, strands
    if strands is None:
        keep = np.empty(n, dtype=np.bool_)
        keep[0] = True
        keep[1:] = (starts[1:] != starts[:-1]) | (ends[1:] != ends[:-1])
        return starts[keep], ends[keep], None
    keep = np.empty(n, dtype=np.bool_)
    keep[0] = True
    keep[1:] = (
        (starts[1:] != starts[:-1])
        | (ends[1:] != ends[:-1])
        | (strands[1:] != strands[:-1])
    )
    return starts[keep], ends[keep], strands[keep]


def _sort_chrom_arrays(
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Sort per-chromosome arrays ascending by (start, end[, strand])."""
    if len(starts) <= 1:
        return starts, ends, strands
    if strands is None:
        order = np.lexsort((ends, starts))
        return starts[order], ends[order], None
    order = np.lexsort((strands, ends, starts))
    return starts[order], ends[order], strands[order]


class FragmentStore:
    """Per-chromosome sorted paired-end (or stranded single-end) fragment store.

    All genomic coordinates use 0-based, half-open intervals ``[start, end)``
    stored as ``int32`` NumPy arrays sorted ascending by ``start`` (and
    tie-broken by ``end``).

    Attributes
    ----------
    starts:
        Mapping ``chrom -> int32 array`` of sorted fragment start coordinates.
    ends:
        Mapping ``chrom -> int32 array`` of paired fragment end coordinates.
    strands:
        Optional mapping ``chrom -> int8 array`` (``+1`` / ``-1``) for stranded
        single-end assays such as ChIP-exo. ``None`` for standard paired-end
        stores.
    max_len:
        Mapping ``chrom -> int`` giving the maximum fragment length on each
        chromosome, used to tightly bound left binary-search windows.
    n_fragments:
        Total number of stored fragments across all chromosomes.
    assay:
        [`AssayConfig`][pyvplot.AssayConfig] describing the assay geometry.
    metadata:
        Arbitrary JSON-serializable metadata dictionary.
    """

    def __init__(
        self,
        starts: dict[str, np.ndarray],
        ends: dict[str, np.ndarray],
        assay: str | AssayConfig = "generic",
        strands: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
        sort: bool = True,
        dedup: bool = False,
    ) -> None:
        self.assay: AssayConfig = get_assay(assay)
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}

        if set(starts.keys()) != set(ends.keys()):
            raise ValueError("`starts` and `ends` must have the exact same chromosome keys.")
        if strands is not None and set(strands.keys()) != set(starts.keys()):
            raise ValueError("`strands` must have the exact same chromosome keys as `starts`.")

        self.starts: dict[str, np.ndarray] = {}
        self.ends: dict[str, np.ndarray] = {}
        self.strands: dict[str, np.ndarray] | None = {} if strands is not None else None
        self.max_len: dict[str, int] = {}

        total = 0
        for chrom in starts:
            s = np.ascontiguousarray(starts[chrom], dtype=np.int32)
            e = np.ascontiguousarray(ends[chrom], dtype=np.int32)
            if s.ndim != 1 or e.ndim != 1 or len(s) != len(e):
                raise ValueError(
                    f"Chromosome {chrom!r}: starts and ends must be 1D arrays of equal length."
                )
            st: np.ndarray | None = None
            if strands is not None:
                st = np.ascontiguousarray(strands[chrom], dtype=np.int8)
                if st.ndim != 1 or len(st) != len(s):
                    raise ValueError(
                        f"Chromosome {chrom!r}: strands must be a 1D array matching starts."
                    )

            if len(s) > 0:
                if np.any(e <= s):
                    raise ValueError(
                        f"Chromosome {chrom!r}: all fragments must satisfy start < end."
                    )
                if st is not None and np.any(e - s != 1):
                    raise ValueError(
                        f"Chromosome {chrom!r}: a stranded store holds one 5' base per read "
                        "([pos, pos + 1)), but some intervals are longer. Build it with "
                        "FragmentStore.from_arrays() or from_dataframe(), which reduce read "
                        "intervals to their 5' end."
                    )
                if sort:
                    s, e, st = _sort_chrom_arrays(s, e, st)
                if dedup:
                    s, e, st = _dedup_sorted_chrom(s, e, st)
                s = np.ascontiguousarray(s, dtype=np.int32)
                e = np.ascontiguousarray(e, dtype=np.int32)
                if st is not None:
                    st = np.ascontiguousarray(st, dtype=np.int8)
                self.max_len[chrom] = int(np.max(e - s))
            else:
                self.max_len[chrom] = 0

            self.starts[chrom] = s
            self.ends[chrom] = e
            if self.strands is not None and st is not None:
                self.strands[chrom] = st
            total += len(s)

        self.n_fragments: int = total

    @property
    def chroms(self) -> list[str]:
        """List of chromosomes present in the store."""
        return list(self.starts.keys())

    @property
    def is_stranded(self) -> bool:
        """Whether per-fragment strand arrays are stored (e.g., ChIP-exo)."""
        return self.strands is not None

    def __len__(self) -> int:
        return self.n_fragments

    def __repr__(self) -> str:
        return (
            f"FragmentStore(n_fragments={self.n_fragments:,}, "
            f"n_chroms={len(self.starts)}, assay={self.assay.name!r})"
        )

    def lengths(self, chrom: str | None = None) -> np.ndarray:
        """Return 1D ``int32`` array of fragment lengths for one or all chromosomes."""
        if chrom is not None:
            if chrom not in self.starts:
                return np.empty(0, dtype=np.int32)
            return (self.ends[chrom] - self.starts[chrom]).astype(np.int32, copy=False)
        if self.n_fragments == 0:
            return np.empty(0, dtype=np.int32)
        parts = [
            (self.ends[c] - self.starts[c]).astype(np.int32, copy=False)
            for c in self.starts
            if len(self.starts[c]) > 0
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)

    def chrom_counts(self) -> dict[str, int]:
        """Return fragment count per chromosome."""
        return {c: int(len(arr)) for c, arr in self.starts.items()}

    def filter(
        self,
        min_len: int | None = None,
        max_len: int | None = None,
        chroms: Collection[str] | None = None,
        exclude_chroms: Collection[str] | None = None,
        dedup: bool = False,
    ) -> FragmentStore:
        """Return a filtered copy of this [`FragmentStore`][pyvplot.FragmentStore].

        Length bounds ``min_len`` and ``max_len`` are inclusive.
        """
        keep_chroms = set(chroms) if chroms is not None else None
        drop_chroms = set(exclude_chroms) if exclude_chroms is not None else set()

        new_starts: dict[str, np.ndarray] = {}
        new_ends: dict[str, np.ndarray] = {}
        new_strands: dict[str, np.ndarray] | None = {} if self.strands is not None else None

        for chrom in self.starts:
            if keep_chroms is not None and chrom not in keep_chroms:
                continue
            if chrom in drop_chroms:
                continue

            s = self.starts[chrom]
            e = self.ends[chrom]
            st = self.strands[chrom] if self.strands is not None else None

            if len(s) > 0 and (min_len is not None or max_len is not None):
                fl = e - s
                mask = np.ones(len(s), dtype=np.bool_)
                if min_len is not None:
                    mask &= fl >= min_len
                if max_len is not None:
                    mask &= fl <= max_len
                s = s[mask]
                e = e[mask]
                if st is not None:
                    st = st[mask]

            new_starts[chrom] = s.copy()
            new_ends[chrom] = e.copy()
            if new_strands is not None and st is not None:
                new_strands[chrom] = st.copy()

        return FragmentStore(
            starts=new_starts,
            ends=new_ends,
            assay=self.assay,
            strands=new_strands,
            metadata=dict(self.metadata),
            sort=False,
            dedup=dedup,
        )

    def exclude_regions(
        self,
        regions: Any,
        mode: str = "overlap",
    ) -> FragmentStore:
        """Return a copy without the fragments that hit ``regions`` (e.g. an ENCODE-style blacklist).

        Parameters
        ----------
        regions:
            A BED path (plain or ``.gz``; ``#``, ``track`` and ``browser`` lines are
            skipped) or a DataFrame with ``chrom`` (or ``chr`` / ``seqnames`` /
            ``chromosome``), ``start`` and ``end`` columns. Intervals are 0-based,
            half-open like BED, and may overlap each other.
        mode:
            ``"overlap"`` (default) drops every fragment sharing at least one base with a
            region, so a fragment poking into the region by a single base goes too.
            ``"midpoint"`` drops fragments whose midpoint lies inside a region. For a
            stranded store each read is one base, so both modes agree.

        Notes
        -----
        ``n_fragments`` of the result counts only the kept fragments, so CPM scaling no
        longer includes the blacklisted regions. Chromosomes absent from ``regions`` are
        left untouched, and a warning is raised when no region lies on a chromosome of the
        store (usually ``chr1`` vs ``1``). What was removed is recorded in
        ``metadata["exclude_regions"]``.
        """
        if mode not in _REGION_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {list(_REGION_MODES)}.")
        table = load_intervals(regions, what="Regions")
        source = str(regions) if isinstance(regions, (str, Path)) else "DataFrame"

        if len(table) and not set(table["chrom"]) & set(self.starts):
            shown = ", ".join(repr(c) for c in list(dict.fromkeys(table["chrom"]))[:5])
            have = ", ".join(repr(c) for c in self.chroms[:5])
            warnings.warn(
                f"None of the {len(table)} regions lie on a chromosome of this store "
                f"(regions: {shown}; store: {have}). Nothing was removed: check chromosome "
                "naming ('chr1' vs '1').",
                UserWarning,
                stacklevel=2,
            )

        by_chrom = {
            str(c): merge_intervals(g["start"].to_numpy(), g["end"].to_numpy())
            for c, g in table.groupby("chrom", sort=False)
        }

        new_starts: dict[str, np.ndarray] = {}
        new_ends: dict[str, np.ndarray] = {}
        new_strands: dict[str, np.ndarray] | None = {} if self.strands is not None else None
        n_removed = 0
        for chrom in self.starts:
            s, e = self.starts[chrom], self.ends[chrom]
            st = self.strands[chrom] if self.strands is not None else None
            if chrom in by_chrom and len(s) > 0:
                reg_s, reg_e = by_chrom[chrom]
                drop = _in_regions(s, e, reg_s, reg_e, mode)
                n_removed += int(drop.sum())
                keep = ~drop
                s, e = s[keep], e[keep]
                st = st[keep] if st is not None else None
            new_starts[chrom] = s.copy()
            new_ends[chrom] = e.copy()
            if new_strands is not None and st is not None:
                new_strands[chrom] = st.copy()

        n_regions = int(sum(len(v[0]) for v in by_chrom.values()))
        meta = dict(self.metadata)
        meta["exclude_regions"] = [
            *meta.get("exclude_regions", []),
            {
                "source": source,
                "mode": mode,
                "n_regions": n_regions,
                "n_removed": int(n_removed),
                "n_before": int(self.n_fragments),
            },
        ]
        return FragmentStore(
            starts=new_starts,
            ends=new_ends,
            assay=self.assay,
            strands=new_strands,
            metadata=meta,
            sort=False,
        )

    def to_dataframe(self, chrom: str | None = None) -> pd.DataFrame:
        """Export stored fragments to a pandas DataFrame."""
        target_chroms = [chrom] if chrom is not None else list(self.starts.keys())
        rows: list[pd.DataFrame] = []
        for c in target_chroms:
            if c not in self.starts or len(self.starts[c]) == 0:
                continue
            data: dict[str, Any] = {
                "chrom": np.repeat(c, len(self.starts[c])),
                "start": self.starts[c],
                "end": self.ends[c],
            }
            if self.strands is not None:
                data["strand"] = np.where(self.strands[c] >= 0, "+", "-")
            rows.append(pd.DataFrame(data))
        if not rows:
            cols = ["chrom", "start", "end"] + (["strand"] if self.strands is not None else [])
            return pd.DataFrame(columns=cols)
        return pd.concat(rows, ignore_index=True)

    @classmethod
    def from_arrays(
        cls,
        starts: dict[str, np.ndarray],
        ends: dict[str, np.ndarray],
        assay: str | AssayConfig = "generic",
        strands: dict[str, np.ndarray] | None = None,
        apply_shift: bool = False,
        min_len: int | None = None,
        max_len: int | None = None,
        dedup: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> FragmentStore:
        """Construct a [`FragmentStore`][pyvplot.FragmentStore] from per-chromosome coordinate arrays.

        With ``strands`` (``+1`` / ``-1``) the store holds one 5' base per read, so read
        intervals are reduced to their 5' end: ``start`` for ``+`` reads, ``end - 1`` for ``-``.
        """
        cfg = get_assay(assay)
        out_starts: dict[str, np.ndarray] = {}
        out_ends: dict[str, np.ndarray] = {}
        out_strands: dict[str, np.ndarray] | None = {} if strands is not None else None

        for chrom in starts:
            s = np.asarray(starts[chrom], dtype=np.int64)
            e = np.asarray(ends[chrom], dtype=np.int64)
            st = np.asarray(strands[chrom], dtype=np.int8) if strands is not None else None

            if st is not None and len(s) > 0:
                if len(st) != len(s) or len(e) != len(s):
                    raise ValueError(
                        f"Chromosome {chrom!r}: starts, ends and strands must have equal lengths."
                    )
                # Stranded stores hold the 5' base of each read: the first base of a + read,
                # the last base of a - read. A no-op for inputs that already are 1 bp long.
                pos = np.where(st >= 0, s, e - 1)
                s, e = pos, pos + 1

            if apply_shift and len(s) > 0:
                if cfg.paired_end:
                    s = s + cfg.left_shift
                    e = e + cfg.right_shift
                elif st is not None:
                    shift = np.where(st >= 0, cfg.left_shift, cfg.right_shift)
                    s = s + shift
                    e = s + 1
                else:
                    s = s + cfg.left_shift
                    e = e + cfg.right_shift

            if len(s) > 0:
                valid = (s >= 0) & (e > s)
                fl = e - s
                if min_len is not None:
                    valid &= fl >= min_len
                if max_len is not None:
                    valid &= fl <= max_len
                s = s[valid]
                e = e[valid]
                if st is not None:
                    st = st[valid]

            out_starts[chrom] = s.astype(np.int32, copy=False)
            out_ends[chrom] = e.astype(np.int32, copy=False)
            if out_strands is not None and st is not None:
                out_strands[chrom] = st.astype(np.int8, copy=False)

        return cls(
            starts=out_starts,
            ends=out_ends,
            assay=cfg,
            strands=out_strands,
            metadata=metadata,
            sort=True,
            dedup=dedup,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        assay: str | AssayConfig = "generic",
        chrom_col: str = "chrom",
        start_col: str = "start",
        end_col: str = "end",
        strand_col: str | None = None,
        apply_shift: bool = False,
        min_len: int | None = None,
        max_len: int | None = None,
        dedup: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> FragmentStore:
        """Construct a [`FragmentStore`][pyvplot.FragmentStore] from a pandas DataFrame."""
        starts: dict[str, np.ndarray] = {}
        ends: dict[str, np.ndarray] = {}
        strands: dict[str, np.ndarray] | None = {} if strand_col is not None else None

        if len(df) > 0:
            for chrom, grp in df.groupby(chrom_col, sort=False):
                c = str(chrom)
                starts[c] = grp[start_col].to_numpy(dtype=np.int64)
                ends[c] = grp[end_col].to_numpy(dtype=np.int64)
                if strands is not None and strand_col is not None:
                    raw_st = grp[strand_col].to_numpy()
                    if np.issubdtype(raw_st.dtype, np.number):
                        strands[c] = np.where(raw_st < 0, -1, 1).astype(np.int8)
                    else:
                        strands[c] = np.where(
                            np.isin(raw_st.astype(str), ["-", "-1"]), -1, 1
                        ).astype(np.int8)

        return cls.from_arrays(
            starts=starts,
            ends=ends,
            assay=assay,
            strands=strands,
            apply_shift=apply_shift,
            min_len=min_len,
            max_len=max_len,
            dedup=dedup,
            metadata=metadata,
        )

    @classmethod
    def from_bam(
        cls,
        bam_path: str | Path,
        assay: str | AssayConfig = "chec",
        min_mapq: int = 20,
        exclude_flags: int | None = None,
        dedup: bool | str = False,
        min_len: int = 10,
        max_len: int = 1000,
        chroms: Collection[str] | None = None,
        exclude_chroms: Collection[str] | None = None,
        pairing: str = "read1",
        apply_shift: bool = True,
        threads: int = 1,
        exclude_regions: Any = None,
        exclude_mode: str = "overlap",
        verbose: bool = False,
    ) -> FragmentStore:
        """Ingest fragments from a BAM/CRAM file via ``pysam``.

        Parameters
        ----------
        bam_path:
            Path to BAM (or CRAM) file.
        assay:
            Assay preset name or [`AssayConfig`][pyvplot.AssayConfig].
        min_mapq:
            Minimum mapping quality (MAPQ) threshold.
        exclude_flags:
            SAM bitmask of flags to skip. Defaults to unmapped, mate-unmapped
            (paired-end), secondary, qcfail, and supplementary reads.
        dedup:
            Duplicate filtering mode: ``False`` (none), ``True`` (exclude SAM
            duplicate flag ``0x400`` and deduplicate identical coordinates),
            ``"flag"`` (only SAM ``0x400``), or ``"coords"`` (only coordinate
            deduplication).
        min_len, max_len:
            Inclusive fragment length bounds (bp) applied after endpoint shift
            for paired-end assays.
        chroms:
            Optional whitelist of chromosomes to keep.
        exclude_chroms:
            Optional chromosomes to exclude (e.g., ``{"chrM", "MT"}``).
        pairing:
            Paired-end extraction strategy: ``"read1"`` (fast O(1)-memory pass
            using proper-pair R1 alignments and ``template_length``) or
            ``"mate"`` (buffers mates by query name to verify both mates pass
            ``min_mapq`` and compute exact outer bounds).
        apply_shift:
            Whether to apply ``assay.left_shift`` / ``assay.right_shift``
            during ingestion (default ``True``).
        threads:
            Extra threads pysam uses to decompress the BAM (default ``1``). The read
            loop itself stays single-threaded, but decompression is a large share of
            the time on deep libraries.
        exclude_regions:
            Optional blacklist: a BED path or a DataFrame with ``chrom``, ``start``,
            ``end``. Fragments hitting it are dropped right after ingestion, so
            ``n_fragments`` (and CPM) excludes them. See
            [`exclude_regions`][pyvplot.FragmentStore.exclude_regions].
        exclude_mode:
            ``"overlap"`` (default) or ``"midpoint"``; how a fragment is matched to a region.
        verbose:
            Print progress to stderr (on each new chromosome and every 5M BAM records)
            and a final summary of records read and why they were dropped. The same
            counts are always stored in ``metadata["ingest_stats"]``, verbose or not.

        Notes
        -----
        ``metadata["ingest_stats"]`` counts BAM *records*, not fragments. With
        ``pairing="read1"`` every mate-2 record is skipped by design (one fragment per
        R1), so ``skipped_pairing`` is about half of ``records_read`` on a clean
        paired-end BAM. Keys: ``records_read``, ``skipped_flag``, ``skipped_pairing``
        (not a proper pair / mate 2 / mate on another chromosome / zero TLEN),
        ``skipped_mapq``, ``skipped_chrom``, ``skipped_length`` (outside
        ``[min_len, max_len]`` or a negative start after shifting),
        ``unpaired_mates`` (``pairing="mate"`` only), ``fragments_parsed``,
        ``removed_coord_dedup``, ``removed_blacklist``, ``fragments_kept``,
        ``seconds``. With ``pairing="mate"``, ``skipped_length`` and ``fragments_parsed``
        count pairs and the first-seen mate of each pair is not counted under any key, so
        the ``skipped_*`` keys plus ``fragments_parsed`` fall short of ``records_read``.
        Flag order matters: a read failing several filters is counted
        under the first one checked (flag, pairing, MAPQ, chromosome, length).
        """
        try:
            import pysam
        except ImportError as exc:
            raise ImportError(
                "Reading BAM files requires `pysam`. Install it with `pip install pyvplot[bam]`."
            ) from exc

        cfg = get_assay(assay)
        if pairing not in {"read1", "mate"}:
            raise ValueError(f"Invalid pairing mode {pairing!r}; expected 'read1' or 'mate'.")

        if exclude_flags is None:
            excl = (
                DEFAULT_PAIRED_EXCLUDE_FLAGS if cfg.paired_end else DEFAULT_SINGLE_EXCLUDE_FLAGS
            )
        else:
            excl = int(exclude_flags)

        dedup_coords = False
        if dedup is True:
            excl |= SAM_DUPLICATE_FLAG
            dedup_coords = True
        elif dedup == "flag":
            excl |= SAM_DUPLICATE_FLAG
        elif dedup == "coords":
            dedup_coords = True
        elif dedup is not False:
            raise ValueError(
                f"Invalid dedup value {dedup!r}; expected False, True, 'flag', or 'coords'."
            )

        keep_chroms = set(chroms) if chroms is not None else None
        drop_chroms = set(exclude_chroms) if exclude_chroms is not None else set()

        left_shift = cfg.left_shift if apply_shift else 0
        right_shift = cfg.right_shift if apply_shift else 0

        buf = _ChromBuffers(stranded=not cfg.paired_end)

        n_seen = 0
        n_flag = n_pair = n_mapq = n_chrom = n_len = n_kept = 0
        n_unpaired = 0
        progress: _BamProgress | None = None

        path_str = str(bam_path)
        with pysam.AlignmentFile(path_str, "rb", threads=max(1, int(threads))) as bam:
            ref_names = bam.references
            allowed_tids: set[int] = set()
            for tid, rname in enumerate(ref_names):
                if keep_chroms is not None and rname not in keep_chroms:
                    continue
                if rname in drop_chroms:
                    continue
                allowed_tids.add(tid)

            if verbose:
                progress = _BamProgress(path_str, tuple(ref_names))
                progress.log(
                    f"reading {path_str} (assay={cfg.name}, pairing="
                    f"{pairing if cfg.paired_end else 'single-end'}, min_mapq={min_mapq}, "
                    f"length {min_len}-{max_len}, {len(allowed_tids)}/{len(ref_names)} "
                    "chromosomes allowed)"
                )

            if not cfg.paired_end:
                # Single-end / ChIP-exo: record stranded 5' cleavage sites as [pos, pos + 1)
                for read in bam.fetch(until_eof=True):
                    n_seen += 1
                    tid = read.reference_id
                    if progress is not None and (tid != progress.tid or n_seen >= progress.next_at):
                        progress.update(n_seen, tid, n_kept)
                    if read.flag & excl:
                        n_flag += 1
                        continue
                    if read.mapping_quality < min_mapq:
                        n_mapq += 1
                        continue
                    if tid not in allowed_tids:
                        n_chrom += 1
                        continue
                    if read.is_reverse:
                        ref_end = read.reference_end
                        if ref_end is None:
                            n_len += 1
                            continue
                        pos = ref_end - 1 + right_shift
                        st = -1
                    else:
                        pos = read.reference_start + left_shift
                        st = 1
                    if pos < 0:
                        n_len += 1
                        continue
                    buf.add(ref_names[tid], pos, pos + 1, st)
                    n_kept += 1

            elif pairing == "read1":
                for read in bam.fetch(until_eof=True):
                    n_seen += 1
                    tid = read.reference_id
                    if progress is not None and (tid != progress.tid or n_seen >= progress.next_at):
                        progress.update(n_seen, tid, n_kept)
                    if read.flag & excl:
                        n_flag += 1
                        continue
                    if not read.is_proper_pair or not read.is_read1:
                        n_pair += 1
                        continue
                    if read.mapping_quality < min_mapq:
                        n_mapq += 1
                        continue
                    if tid not in allowed_tids:
                        n_chrom += 1
                        continue
                    if read.next_reference_id != tid:
                        n_pair += 1
                        continue
                    tlen = read.template_length
                    if tlen == 0:
                        n_pair += 1
                        continue
                    if tlen > 0:
                        start = read.reference_start
                        end = start + tlen
                    else:
                        ref_end = read.reference_end
                        if ref_end is None:
                            n_pair += 1
                            continue
                        end = ref_end
                        start = end + tlen

                    start += left_shift
                    end += right_shift
                    frag_len = end - start
                    if start < 0 or frag_len < min_len or frag_len > max_len:
                        n_len += 1
                        continue
                    buf.add(ref_names[tid], start, end)
                    n_kept += 1

            else:  # pairing == "mate"
                pending: dict[tuple[int, str], tuple[int, int]] = {}
                for read in bam.fetch(until_eof=True):
                    n_seen += 1
                    tid = read.reference_id
                    if progress is not None and (tid != progress.tid or n_seen >= progress.next_at):
                        progress.update(n_seen, tid, n_kept)
                    if read.flag & excl:
                        n_flag += 1
                        continue
                    if not read.is_proper_pair:
                        n_pair += 1
                        continue
                    if read.mapping_quality < min_mapq:
                        n_mapq += 1
                        continue
                    if tid not in allowed_tids:
                        n_chrom += 1
                        continue
                    if read.next_reference_id != tid:
                        n_pair += 1
                        continue
                    ref_end = read.reference_end
                    if ref_end is None:
                        n_pair += 1
                        continue
                    key = (tid, read.query_name)
                    mate = pending.pop(key, None)
                    if mate is None:
                        pending[key] = (read.reference_start, ref_end)
                    else:
                        start = min(mate[0], read.reference_start) + left_shift
                        end = max(mate[1], ref_end) + right_shift
                        frag_len = end - start
                        if start < 0 or frag_len < min_len or frag_len > max_len:
                            n_len += 1
                            continue
                        buf.add(ref_names[tid], start, end)
                        n_kept += 1
                n_unpaired = len(pending)

        if progress is not None:
            progress.log(
                f"finished reading: {_fmt_count(n_seen)} records, {_fmt_count(n_kept)} fragments; "
                "sorting" + (" and deduplicating" if dedup_coords else "")
            )

        starts_np, ends_np, strands_np = buf.to_numpy()
        del buf

        store = cls(
            starts=starts_np,
            ends=ends_np,
            assay=cfg,
            strands=strands_np,
            metadata={"source": path_str, "source_type": "bam", "min_mapq": min_mapq},
            sort=True,
            dedup=dedup_coords,
        )
        n_after_dedup = store.n_fragments
        if exclude_regions is not None:
            if progress is not None:
                progress.log("removing blacklisted fragments")
            store = store.exclude_regions(exclude_regions, mode=exclude_mode)

        stats: dict[str, Any] = {
            "records_read": n_seen,
            "skipped_flag": n_flag,
            "skipped_pairing": n_pair,
            "skipped_mapq": n_mapq,
            "skipped_chrom": n_chrom,
            "skipped_length": n_len,
            "fragments_parsed": n_kept,
            "removed_coord_dedup": n_kept - n_after_dedup,
            "removed_blacklist": n_after_dedup - store.n_fragments,
            "fragments_kept": store.n_fragments,
        }
        if cfg.paired_end and pairing == "mate":
            stats["unpaired_mates"] = n_unpaired
        if progress is not None:
            stats["seconds"] = round(progress.elapsed(), 2)
        store.metadata["ingest_stats"] = stats

        if progress is not None:
            pct = (lambda k: f"{100.0 * k / n_seen:5.1f}%") if n_seen else (lambda k: "  n/a")
            lines = [
                f"records read          {n_seen:>14,}",
                f"  skipped: SAM flags  {n_flag:>14,}  {pct(n_flag)}  (exclude_flags=0x{excl:x})",
                f"  skipped: pairing    {n_pair:>14,}  {pct(n_pair)}"
                + ("  (includes every mate-2 record: one fragment per R1)"
                   if cfg.paired_end and pairing == "read1" else ""),
                f"  skipped: MAPQ<{min_mapq:<3}   {n_mapq:>14,}  {pct(n_mapq)}",
                f"  skipped: chromosome {n_chrom:>14,}  {pct(n_chrom)}",
                f"  skipped: length     {n_len:>14,}  {pct(n_len)}  (outside {min_len}-{max_len} bp)",
            ]
            if "unpaired_mates" in stats:
                lines.append(f"  mates never paired  {n_unpaired:>14,}")
            lines += [
                f"fragments parsed      {n_kept:>14,}",
                f"  removed: coord dedup{stats['removed_coord_dedup']:>14,}",
                f"  removed: blacklist  {stats['removed_blacklist']:>14,}"
                + (f"  ({exclude_mode})" if exclude_regions is not None else "  (none given)"),
                f"fragments kept        {store.n_fragments:>14,}  on {len(store.chroms)} chromosomes",
            ]
            progress.log("done\n    " + "\n    ".join(lines))
        return store

    @classmethod
    def from_fragments_bed(
        cls,
        path: str | Path,
        assay: str | AssayConfig = "generic",
        apply_shift: bool = False,
        min_len: int = 1,
        max_len: int | None = None,
        chroms: Collection[str] | None = None,
        exclude_chroms: Collection[str] | None = None,
        dedup: bool = False,
        bedpe: bool | None = None,
        use_count: bool = False,
        exclude_regions: Any = None,
        exclude_mode: str = "overlap",
    ) -> FragmentStore:
        """Ingest fragments from a BED, 10x/ArchR ``.frag.tsv[.gz]``, or BEDPE file.

        Parameters
        ----------
        path:
            Path to plain-text or gzipped BED / fragments TSV / BEDPE file.
        assay:
            Assay preset name or [`AssayConfig`][pyvplot.AssayConfig].
        apply_shift:
            Whether to apply ``assay.left_shift`` / ``assay.right_shift``
            (default ``False`` since 10x/ArchR fragment files are typically
            already Tn5-shifted).
        min_len, max_len:
            Inclusive fragment length bounds (bp).
        chroms, exclude_chroms:
            Optional chromosome include/exclude filters.
        dedup:
            Whether to deduplicate identical fragment coordinates.
        bedpe:
            If ``True``, parse as 6+-column BEDPE (``chrom1 start1 end1 chrom2
            start2 end2``). If ``None`` (default), auto-detect from file extension
            or column structure of the first data line.
        use_count:
            For 5-column 10x/ArchR fragment files (``chrom start end barcode
            count``), if ``True``, emit each fragment ``count`` times.
        """
        cfg = get_assay(assay)
        path_obj = Path(path)
        keep_chroms = set(chroms) if chroms is not None else None
        drop_chroms = set(exclude_chroms) if exclude_chroms is not None else set()

        left_shift = cfg.left_shift if apply_shift else 0
        right_shift = cfg.right_shift if apply_shift else 0

        is_bedpe = bedpe
        if is_bedpe is None and ".bedpe" in path_obj.name.lower():
            is_bedpe = True

        opener = gzip.open if path_obj.suffix == ".gz" else open
        buf = _ChromBuffers(stranded=not cfg.paired_end)

        with opener(path_obj, "rt", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line or line[0] == "#" or line.startswith(("track ", "browser ")):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    parts = line.split()
                if len(parts) < 3:
                    continue

                if is_bedpe is None:
                    # Auto-detect BEDPE: col 0 & 3 are chrom strings and cols 1,2,4,5 are ints
                    if len(parts) >= 6:
                        try:
                            int(parts[1])
                            int(parts[2])
                            int(parts[4])
                            int(parts[5])
                            # Check col 3 is not a strand or integer score
                            is_bedpe = parts[3] not in {"+", "-", "."} and not parts[3].lstrip("-").isdigit()
                        except ValueError:
                            is_bedpe = False
                    else:
                        is_bedpe = False

                if is_bedpe:
                    if len(parts) < 6:
                        continue
                    c1, c2 = parts[0], parts[3]
                    if c1 != c2:
                        continue
                    chrom = c1
                    if keep_chroms is not None and chrom not in keep_chroms:
                        continue
                    if chrom in drop_chroms:
                        continue
                    s1, e1, s2, e2 = parse_int_columns(parts, (1, 2, 4, 5), path_obj, lineno)
                    start = min(s1, s2) + left_shift
                    end = max(e1, e2) + right_shift
                    count = 1
                    st = 1
                else:
                    chrom = parts[0]
                    if keep_chroms is not None and chrom not in keep_chroms:
                        continue
                    if chrom in drop_chroms:
                        continue
                    start, end = parse_int_columns(parts, (1, 2), path_obj, lineno)
                    st = 1
                    if not cfg.paired_end:
                        if (len(parts) >= 6 and parts[5] == "-") or (len(parts) >= 4 and parts[3] == "-"):
                            st = -1
                        pos = (end - 1 + right_shift) if st < 0 else (start + left_shift)
                        start = pos
                        end = pos + 1
                    else:
                        start += left_shift
                        end += right_shift

                    count = 1
                    if use_count and len(parts) >= 5:
                        try:
                            count = max(1, int(parts[4]))
                        except ValueError:
                            count = 1

                frag_len = end - start
                if start < 0 or frag_len <= 0:
                    continue
                if cfg.paired_end:
                    if frag_len < min_len:
                        continue
                    if max_len is not None and frag_len > max_len:
                        continue

                buf.add(chrom, start, end, st, count)

        starts_np, ends_np, strands_np = buf.to_numpy()

        store = cls(
            starts=starts_np,
            ends=ends_np,
            assay=cfg,
            strands=strands_np,
            metadata={"source": str(path_obj), "source_type": "bedpe" if is_bedpe else "bed"},
            sort=True,
            dedup=dedup,
        )
        if exclude_regions is not None:
            store = store.exclude_regions(exclude_regions, mode=exclude_mode)
        return store

    def save(
        self,
        path: str | Path,
        format: str | None = None,
        compressed: bool = False,
    ) -> Path:
        """Persist the fragment store to disk as ``.npz`` or ``.parquet``.

        Parameters
        ----------
        path:
            Destination path. Format is inferred from suffix (``.npz`` or
            ``.parquet``) unless explicitly passed via ``format``.
        format:
            ``"npz"`` or ``"parquet"`` (optional). A ``.npz`` file always gets the
            ``.npz`` suffix; use the returned path to load it back.
        compressed:
            For ``.npz``, use ``np.savez_compressed`` instead of ``np.savez``
            (smaller files, slightly slower load).
        """
        out_path = Path(path)
        fmt = format.lower() if format is not None else None
        if fmt is None:
            suffix = out_path.suffix.lower()
            fmt = "parquet" if suffix in {".parquet", ".pq"} else "npz"
        if fmt == "npz" and out_path.suffix.lower() != ".npz":
            # np.savez appends ".npz" itself; do it here so the returned path exists.
            if format is None:
                out_path = out_path.with_suffix(".npz")
            else:
                out_path = out_path.with_name(out_path.name + ".npz")

        meta_payload = {
            "version": _STORE_FORMAT_VERSION,
            "assay": self.assay.to_dict(),
            "chroms": self.chroms,
            "is_stranded": self.is_stranded,
            "metadata": self.metadata,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "npz":
            arrays: dict[str, np.ndarray] = {
                "__meta__": np.array(json.dumps(meta_payload), dtype=np.str_),
            }
            for idx, chrom in enumerate(self.chroms):
                arrays[f"s_{idx}"] = self.starts[chrom]
                arrays[f"e_{idx}"] = self.ends[chrom]
                if self.strands is not None:
                    arrays[f"t_{idx}"] = self.strands[chrom]
            if compressed:
                np.savez_compressed(out_path, **arrays)
            else:
                np.savez(out_path, **arrays)
            return out_path

        if fmt == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "Parquet persistence requires `pyarrow`. Install it with `pip install pyvplot[io]`."
                ) from exc

            chrom_parts: list[np.ndarray] = []
            start_parts: list[np.ndarray] = []
            end_parts: list[np.ndarray] = []
            strand_parts: list[np.ndarray] = []

            for chrom in self.chroms:
                n = len(self.starts[chrom])
                if n == 0:
                    continue
                chrom_parts.append(np.repeat(chrom, n))
                start_parts.append(self.starts[chrom])
                end_parts.append(self.ends[chrom])
                if self.strands is not None:
                    strand_parts.append(self.strands[chrom])

            fields = [
                pa.field("chrom", pa.string()),
                pa.field("start", pa.int32()),
                pa.field("end", pa.int32()),
            ]
            cols: list[Any] = [
                pa.array(np.concatenate(chrom_parts) if chrom_parts else [], type=pa.string()),
                pa.array(
                    np.concatenate(start_parts) if start_parts else np.empty(0, dtype=np.int32),
                    type=pa.int32(),
                ),
                pa.array(
                    np.concatenate(end_parts) if end_parts else np.empty(0, dtype=np.int32),
                    type=pa.int32(),
                ),
            ]
            if self.strands is not None:
                fields.append(pa.field("strand", pa.int8()))
                cols.append(
                    pa.array(
                        np.concatenate(strand_parts)
                        if strand_parts
                        else np.empty(0, dtype=np.int8),
                        type=pa.int8(),
                    )
                )

            schema = pa.schema(
                fields,
                metadata={b"pyvplot_meta": json.dumps(meta_payload).encode("utf-8")},
            )
            table = pa.Table.from_arrays(cols, schema=schema)
            pq.write_table(table, out_path)
            return out_path

        raise ValueError(f"Unsupported format {fmt!r}; expected 'npz' or 'parquet'.")

    @classmethod
    def load(cls, path: str | Path, format: str | None = None) -> FragmentStore:
        """Load a persisted [`FragmentStore`][pyvplot.FragmentStore] from ``.npz`` or ``.parquet``."""
        in_path = Path(path)
        fmt = format.lower() if format is not None else None
        if fmt is None:
            fmt = "parquet" if in_path.suffix.lower() in {".parquet", ".pq"} else "npz"

        if fmt == "npz":
            with np.load(in_path, allow_pickle=False) as data:
                meta = json.loads(str(data["__meta__"].item()))
                assay = AssayConfig.from_dict(meta["assay"])
                chroms: list[str] = meta["chroms"]
                is_stranded: bool = bool(meta.get("is_stranded", False))
                starts = {c: np.ascontiguousarray(data[f"s_{i}"], dtype=np.int32) for i, c in enumerate(chroms)}
                ends = {c: np.ascontiguousarray(data[f"e_{i}"], dtype=np.int32) for i, c in enumerate(chroms)}
                strands = (
                    {c: np.ascontiguousarray(data[f"t_{i}"], dtype=np.int8) for i, c in enumerate(chroms)}
                    if is_stranded
                    else None
                )
            return cls(
                starts=starts,
                ends=ends,
                assay=assay,
                strands=strands,
                metadata=meta.get("metadata", {}),
                sort=False,
            )

        if fmt == "parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError(
                    "Parquet loading requires `pyarrow`. Install it with `pip install pyvplot[io]`."
                ) from exc

            table = pq.read_table(in_path)
            raw_meta = (table.schema.metadata or {}).get(b"pyvplot_meta")
            if raw_meta is not None:
                meta = json.loads(raw_meta.decode("utf-8"))
                assay = AssayConfig.from_dict(meta["assay"])
                expected_chroms: list[str] = meta.get("chroms", [])
                is_stranded = bool(meta.get("is_stranded", "strand" in table.column_names))
                extra_meta = meta.get("metadata", {})
            else:
                assay = get_assay("generic")
                expected_chroms = []
                is_stranded = "strand" in table.column_names
                extra_meta = {}

            chrom_arr = table["chrom"].to_numpy(zero_copy_only=False)
            start_arr = table["start"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            end_arr = table["end"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            strand_arr = (
                table["strand"].to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
                if is_stranded and "strand" in table.column_names
                else None
            )

            starts: dict[str, np.ndarray] = {c: np.empty(0, dtype=np.int32) for c in expected_chroms}
            ends: dict[str, np.ndarray] = {c: np.empty(0, dtype=np.int32) for c in expected_chroms}
            strands: dict[str, np.ndarray] | None = (
                {c: np.empty(0, dtype=np.int8) for c in expected_chroms} if is_stranded else None
            )

            if len(chrom_arr) > 0:
                # Partition by chromosome preserving order
                unique_chroms, idx = np.unique(chrom_arr, return_index=True)
                ordered_chroms = unique_chroms[np.argsort(idx)]
                for c in ordered_chroms:
                    c_str = str(c)
                    mask = chrom_arr == c
                    starts[c_str] = np.ascontiguousarray(start_arr[mask], dtype=np.int32)
                    ends[c_str] = np.ascontiguousarray(end_arr[mask], dtype=np.int32)
                    if strands is not None and strand_arr is not None:
                        strands[c_str] = np.ascontiguousarray(strand_arr[mask], dtype=np.int8)

            return cls(
                starts=starts,
                ends=ends,
                assay=assay,
                strands=strands,
                metadata=extra_meta,
                sort=raw_meta is None,
            )

        raise ValueError(f"Unsupported format {fmt!r}; expected 'npz' or 'parquet'.")

    # ------------------------------------------------------------------
    # Method forms of the analysis functions. Signatures mirror the module-level
    # functions exactly (tests/test_store_methods.py enforces this).
    # ------------------------------------------------------------------

    def vplot(
        self,
        anchors: Any,
        flank: int = 300,
        upstream: int | None = None,
        downstream: int | None = None,
        min_len: int = 20,
        max_len: int = 250,
        pos_bin: int = 1,
        len_bin: int = 1,
        feature: str = "midpoint",
        motif_overlap: str | None = None,
        use_numba: bool | None = None,
    ) -> VPlot:
        """V-plot around ``anchors``. Same as [`pyvplot.vplot`][pyvplot.vplot] with this store."""
        from pyvplot.vplot import vplot

        return vplot(
            self, anchors, flank=flank, upstream=upstream, downstream=downstream,
            min_len=min_len, max_len=max_len, pos_bin=pos_bin, len_bin=len_bin,
            feature=feature, motif_overlap=motif_overlap, use_numba=use_numba,
        )

    def locus_vplot(
        self,
        region: Any,
        motif: Any = None,
        feature: str = "midpoint",
        len_range: tuple[int, int] | str | None = None,
        motif_overlap: str | None = None,
        coords: str | None = None,
        orient: bool = True,
    ) -> LocusVPlot:
        """Fragments in one window. Same as [`pyvplot.locus_vplot`][pyvplot.locus_vplot] with this store."""
        from pyvplot.locus import locus_vplot

        return locus_vplot(
            self, region, motif=motif, feature=feature, len_range=len_range,
            motif_overlap=motif_overlap, coords=coords, orient=orient,
        )

    def to_matrix(
        self,
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
        """Per-anchor binned signal. Same as [`pyvplot.to_matrix`][pyvplot.to_matrix] with this store."""
        from pyvplot.matrices import to_matrix

        return to_matrix(
            self, anchors, flank=flank, upstream=upstream, downstream=downstream,
            bin_size=bin_size, feature=feature, len_range=len_range, normalize=normalize,
            motif_overlap=motif_overlap, use_numba=use_numba,
        )

    def cut_site_footprint(
        self,
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
        """1-bp cut-site profile. Same as [`pyvplot.cut_site_footprint`][pyvplot.cut_site_footprint] with this store."""
        from pyvplot.matrices import cut_site_footprint

        return cut_site_footprint(
            self, anchors, flank=flank, upstream=upstream, downstream=downstream,
            bin_size=bin_size, len_range=len_range, split_strands=split_strands,
            normalize=normalize, agg=agg, keep_matrices=keep_matrices, split_by=split_by,
            motif_overlap=motif_overlap, use_numba=use_numba,
        )

    def footprint_score(
        self,
        anchors: Any,
        motif_half_width: int = 10,
        flank_width: int = 30,
        len_range: tuple[int, int] | str | None = None,
        feature: str = "cuts",
        pseudocount: float = 0.05,
        normalize: str | None = None,
        use_numba: bool | None = None,
    ) -> pd.DataFrame:
        """Per-anchor protection scores. Same as [`pyvplot.footprint_score`][pyvplot.footprint_score] with this store."""
        from pyvplot.matrices import footprint_score

        return footprint_score(
            self, anchors, motif_half_width=motif_half_width, flank_width=flank_width,
            len_range=len_range, feature=feature, pseudocount=pseudocount,
            normalize=normalize, use_numba=use_numba,
        )

    def extract_events(
        self,
        anchors: Any,
        flank: float = 300,
        upstream: float | None = None,
        downstream: float | None = None,
        len_range: tuple[int, int] | str | None = None,
        overlap_mode: str = "midpoint",
        motif_overlap: str | None = None,
    ) -> pd.DataFrame:
        """Per-fragment table around anchors. Same as [`pyvplot.extract_events`][pyvplot.extract_events]."""
        from pyvplot.extract import extract_events

        return extract_events(
            self, anchors, flank=flank, upstream=upstream, downstream=downstream,
            len_range=len_range, overlap_mode=overlap_mode, motif_overlap=motif_overlap,
        )

    def insert_size_distribution(
        self,
        min_len: int | None = None,
        max_len: int | None = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Fragment-length histogram. Same as [`pyvplot.insert_size_distribution`][pyvplot.insert_size_distribution]."""
        from pyvplot.qc import insert_size_distribution

        return insert_size_distribution(self, min_len=min_len, max_len=max_len, normalize=normalize)

    def phasogram(
        self,
        max_lag: int = 1000,
        chroms: Collection[str] | None = None,
        len_range: tuple[int, int] | str | None = None,
        mode: str = "midpoint",
        max_positions_per_chrom: int | None = None,
        smooth_sigma: float = 5.0,
        min_peak_lag: int = 120,
        max_peak_lag: int = 850,
        expected_nrl_range: tuple[float, float] = (140.0, 240.0),
        use_numba: bool | None = None,
    ) -> PhasogramResult:
        """Pair-distance histogram and NRL. Same as [`pyvplot.phasogram`][pyvplot.phasogram]."""
        from pyvplot.qc import phasogram

        return phasogram(
            self, max_lag=max_lag, chroms=chroms, len_range=len_range, mode=mode,
            max_positions_per_chrom=max_positions_per_chrom, smooth_sigma=smooth_sigma,
            min_peak_lag=min_peak_lag, max_peak_lag=max_peak_lag,
            expected_nrl_range=expected_nrl_range, use_numba=use_numba,
        )

    @overload
    def frip(self, peaks: Any, mode: str = ..., return_per_peak: Literal[False] = ...) -> float: ...

    @overload
    def frip(
        self, peaks: Any, mode: str = ..., *, return_per_peak: Literal[True]
    ) -> tuple[float, np.ndarray]: ...

    @overload
    def frip(
        self, peaks: Any, mode: str = ..., return_per_peak: bool = ...
    ) -> float | tuple[float, np.ndarray]: ...

    def frip(
        self,
        peaks: Any,
        mode: str = "midpoint",
        return_per_peak: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Fraction of fragments in peaks. Same as [`pyvplot.frip`][pyvplot.frip]."""
        from pyvplot.qc import frip

        return frip(self, peaks, mode=mode, return_per_peak=return_per_peak)

    def qc_summary(self, peaks: Any = None) -> dict[str, Any]:
        """Length statistics, size-bin fractions and optional FRiP. Same as [`pyvplot.qc_summary`][pyvplot.qc_summary]."""
        from pyvplot.qc import qc_summary

        return qc_summary(self, peaks=peaks)

    def scan_locus_pillars(
        self,
        region: Any,
        *,
        len_range: tuple[int, int] = (20, 250),
        len_bin: int = 10,
        tf_len_range: tuple[int, int] = (20, 120),
        pillar_window: int = 3,
        footprint_widths: Sequence[int] | np.ndarray | None = None,
        motif_width_range: tuple[int, int] = (10, 24),
        flank_pad: int = 2,
        flank_mode: str = "both",
        pseudocount: float = 2.0,
        min_pillar_score: float = 8.0,
        min_len_bins: int = 4,
        min_pillar_prominence: float = 2.0,
        min_protection_score: float = 0.75,
        min_raw_protection: float = 0.20,
        min_motif_distance: int = 12,
        expected_cuts: np.ndarray | None = None,
    ) -> LocusPillarScan:
        """Scan one window for multi-length cut pillars and protected motifs. Same as [`pyvplot.scan_locus_pillars`][pyvplot.scan_locus_pillars]."""
        from pyvplot.pillars import scan_locus_pillars

        return scan_locus_pillars(
            self,
            region,
            len_range=len_range,
            len_bin=len_bin,
            tf_len_range=tf_len_range,
            pillar_window=pillar_window,
            footprint_widths=footprint_widths,
            motif_width_range=motif_width_range,
            flank_pad=flank_pad,
            flank_mode=flank_mode,
            pseudocount=pseudocount,
            min_pillar_score=min_pillar_score,
            min_len_bins=min_len_bins,
            min_pillar_prominence=min_pillar_prominence,
            min_protection_score=min_protection_score,
            min_raw_protection=min_raw_protection,
            min_motif_distance=min_motif_distance,
            expected_cuts=expected_cuts,
        )

    def scan_genome_pillars(
        self,
        *,
        chroms: Collection[str] | None = None,
        len_range: tuple[int, int] = (20, 250),
        len_bin: int = 10,
        tf_len_range: tuple[int, int] = (20, 120),
        pillar_window: int = 3,
        motif_width_range: tuple[int, int] = (10, 24),
        min_bin_frags: int = 12,
        flank_mode: str = "both",
        pseudocount: float = 2.0,
        min_pillar_score: float = 8.0,
        min_len_bins: int = 4,
        min_pillar_prominence: float = 2.0,
        min_protection_score: float = 0.75,
        min_raw_protection: float = 0.20,
        min_motif_distance: int = 12,
        expected_cuts: Callable[[str, int, int], np.ndarray] | None = None,
        return_pillars: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Scan active windows genome-wide for protected motifs and cut pillars. Same as [`pyvplot.scan_genome_pillars`][pyvplot.scan_genome_pillars]."""
        from pyvplot.pillars import scan_genome_pillars

        return scan_genome_pillars(
            self,
            chroms=chroms,
            len_range=len_range,
            len_bin=len_bin,
            tf_len_range=tf_len_range,
            pillar_window=pillar_window,
            motif_width_range=motif_width_range,
            min_bin_frags=min_bin_frags,
            flank_mode=flank_mode,
            pseudocount=pseudocount,
            min_pillar_score=min_pillar_score,
            min_len_bins=min_len_bins,
            min_pillar_prominence=min_pillar_prominence,
            min_protection_score=min_protection_score,
            min_raw_protection=min_raw_protection,
            min_motif_distance=min_motif_distance,
            expected_cuts=expected_cuts,
            return_pillars=return_pillars,
        )

    @classmethod
    def merge(
        cls,
        stores: Iterable[FragmentStore],
        dedup: bool = False,
        assay: str | AssayConfig | None = None,
    ) -> FragmentStore:
        """Merge multiple [`FragmentStore`][pyvplot.FragmentStore] instances into one.

        All stores must share one assay configuration (and all be stranded or none);
        otherwise pass ``assay=`` to say which one the merged store should carry. The
        merged store records the ``source`` of each input in ``metadata["merged_from"]``
        and each input's full metadata (``ingest_stats``, ``exclude_regions``, ...) in
        ``metadata["merged_inputs"]``, in input order. With ``dedup=True`` the number of
        coordinate duplicates removed across inputs is ``metadata["removed_coord_dedup"]``.
        """
        store_list = list(stores)
        if not store_list:
            raise ValueError("Cannot merge an empty collection of FragmentStores.")
        is_stranded = store_list[0].is_stranded
        for st in store_list[1:]:
            if st.is_stranded != is_stranded:
                raise ValueError("Cannot merge stranded and unstranded FragmentStores.")
        if assay is None:
            assay_cfg = store_list[0].assay
            for st in store_list[1:]:
                if st.assay != assay_cfg:
                    raise ValueError(
                        f"Cannot merge stores with different assays ({assay_cfg.name!r} vs "
                        f"{st.assay.name!r}); pass assay=... to choose the merged store's assay."
                    )
        else:
            assay_cfg = get_assay(assay)

        all_chroms: list[str] = []
        seen: set[str] = set()
        for st in store_list:
            for c in st.chroms:
                if c not in seen:
                    seen.add(c)
                    all_chroms.append(c)

        starts: dict[str, np.ndarray] = {}
        ends: dict[str, np.ndarray] = {}
        strands: dict[str, np.ndarray] | None = {} if is_stranded else None

        for c in all_chroms:
            s_parts = [st.starts[c] for st in store_list if c in st.starts and len(st.starts[c]) > 0]
            e_parts = [st.ends[c] for st in store_list if c in st.ends and len(st.ends[c]) > 0]
            starts[c] = np.concatenate(s_parts) if s_parts else np.empty(0, dtype=np.int32)
            ends[c] = np.concatenate(e_parts) if e_parts else np.empty(0, dtype=np.int32)
            if strands is not None:
                t_parts = [
                    st.strands[c]
                    for st in store_list
                    if st.strands is not None and c in st.strands and len(st.strands[c]) > 0
                ]
                strands[c] = np.concatenate(t_parts) if t_parts else np.empty(0, dtype=np.int8)

        n_in = int(sum(st.n_fragments for st in store_list))
        merged = cls(
            starts=starts,
            ends=ends,
            assay=assay_cfg,
            strands=strands,
            metadata={
                "source_type": "merge",
                "merged_from": [st.metadata.get("source") for st in store_list],
                "merged_inputs": [dict(st.metadata) for st in store_list],
            },
            sort=True,
            dedup=dedup,
        )
        if dedup:
            merged.metadata["removed_coord_dedup"] = n_in - merged.n_fragments
        return merged
