"""Compact, sorted per-chromosome fragment store for millisecond genomic queries."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.assays import AssayConfig, get_assay

# SAM flags: unmapped (0x4) | mate unmapped (0x8) | secondary (0x100) | qcfail (0x200) | supplementary (0x800)
DEFAULT_PAIRED_EXCLUDE_FLAGS = 0x4 | 0x8 | 0x100 | 0x200 | 0x800
# Single-end SAM flags: unmapped (0x4) | secondary (0x100) | qcfail (0x200) | supplementary (0x800)
DEFAULT_SINGLE_EXCLUDE_FLAGS = 0x4 | 0x100 | 0x200 | 0x800
SAM_DUPLICATE_FLAG = 0x400

_STORE_FORMAT_VERSION = 1


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
        :class:`~pyvplot.assays.AssayConfig` describing the assay geometry.
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
        """Return a filtered copy of this :class:`FragmentStore`.

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
        """Construct a :class:`FragmentStore` from per-chromosome coordinate arrays."""
        cfg = get_assay(assay)
        out_starts: dict[str, np.ndarray] = {}
        out_ends: dict[str, np.ndarray] = {}
        out_strands: dict[str, np.ndarray] | None = {} if strands is not None else None

        for chrom in starts:
            s = np.asarray(starts[chrom], dtype=np.int64)
            e = np.asarray(ends[chrom], dtype=np.int64)
            st = np.asarray(strands[chrom], dtype=np.int8) if strands is not None else None

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
        """Construct a :class:`FragmentStore` from a pandas DataFrame."""
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
    ) -> FragmentStore:
        """Ingest fragments from a BAM/CRAM file via ``pysam``.

        Parameters
        ----------
        bam_path:
            Path to BAM (or CRAM) file.
        assay:
            Assay preset name or :class:`~pyvplot.assays.AssayConfig`.
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

        raw_starts: dict[str, list[int]] = defaultdict(list)
        raw_ends: dict[str, list[int]] = defaultdict(list)
        raw_strands: dict[str, list[int]] | None = defaultdict(list) if not cfg.paired_end else None

        path_str = str(bam_path)
        with pysam.AlignmentFile(path_str, "rb") as bam:
            ref_names = bam.references
            allowed_tids: set[int] = set()
            for tid, rname in enumerate(ref_names):
                if keep_chroms is not None and rname not in keep_chroms:
                    continue
                if rname in drop_chroms:
                    continue
                allowed_tids.add(tid)

            if not cfg.paired_end:
                # Single-end / ChIP-exo: record stranded 5' cleavage sites as [pos, pos + 1)
                for read in bam.fetch(until_eof=True):
                    if read.flag & excl:
                        continue
                    if read.mapping_quality < min_mapq:
                        continue
                    tid = read.reference_id
                    if tid not in allowed_tids:
                        continue
                    if read.is_reverse:
                        ref_end = read.reference_end
                        if ref_end is None:
                            continue
                        pos = ref_end - 1 + right_shift
                        st = -1
                    else:
                        pos = read.reference_start + left_shift
                        st = 1
                    if pos < 0:
                        continue
                    chrom = ref_names[tid]
                    raw_starts[chrom].append(pos)
                    raw_ends[chrom].append(pos + 1)
                    assert raw_strands is not None
                    raw_strands[chrom].append(st)

            elif pairing == "read1":
                for read in bam.fetch(until_eof=True):
                    if read.flag & excl:
                        continue
                    if not read.is_proper_pair or not read.is_read1:
                        continue
                    if read.mapping_quality < min_mapq:
                        continue
                    tid = read.reference_id
                    if tid not in allowed_tids or read.next_reference_id != tid:
                        continue
                    tlen = read.template_length
                    if tlen == 0:
                        continue
                    if tlen > 0:
                        start = read.reference_start
                        end = start + tlen
                    else:
                        ref_end = read.reference_end
                        if ref_end is None:
                            continue
                        end = ref_end
                        start = end + tlen

                    start += left_shift
                    end += right_shift
                    frag_len = end - start
                    if start < 0 or frag_len < min_len or frag_len > max_len:
                        continue
                    chrom = ref_names[tid]
                    raw_starts[chrom].append(start)
                    raw_ends[chrom].append(end)

            else:  # pairing == "mate"
                pending: dict[tuple[int, str], tuple[int, int]] = {}
                for read in bam.fetch(until_eof=True):
                    if read.flag & excl:
                        continue
                    if not read.is_proper_pair:
                        continue
                    if read.mapping_quality < min_mapq:
                        continue
                    tid = read.reference_id
                    if tid not in allowed_tids or read.next_reference_id != tid:
                        continue
                    ref_end = read.reference_end
                    if ref_end is None:
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
                            continue
                        chrom = ref_names[tid]
                        raw_starts[chrom].append(start)
                        raw_ends[chrom].append(end)

        starts_np = {c: np.asarray(v, dtype=np.int32) for c, v in raw_starts.items()}
        ends_np = {c: np.asarray(v, dtype=np.int32) for c, v in raw_ends.items()}
        strands_np = (
            {c: np.asarray(v, dtype=np.int8) for c, v in raw_strands.items()}
            if raw_strands is not None
            else None
        )

        return cls(
            starts=starts_np,
            ends=ends_np,
            assay=cfg,
            strands=strands_np,
            metadata={"source": path_str, "source_type": "bam", "min_mapq": min_mapq},
            sort=True,
            dedup=dedup_coords,
        )

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
    ) -> FragmentStore:
        """Ingest fragments from a BED, 10x/ArchR ``.frag.tsv[.gz]``, or BEDPE file.

        Parameters
        ----------
        path:
            Path to plain-text or gzipped BED / fragments TSV / BEDPE file.
        assay:
            Assay preset name or :class:`~pyvplot.assays.AssayConfig`.
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
        raw_starts: dict[str, list[int]] = defaultdict(list)
        raw_ends: dict[str, list[int]] = defaultdict(list)
        raw_strands: dict[str, list[int]] | None = defaultdict(list) if not cfg.paired_end else None

        with opener(path_obj, "rt", encoding="utf-8") as fh:
            for line in fh:
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
                    s1, e1 = int(parts[1]), int(parts[2])
                    s2, e2 = int(parts[4]), int(parts[5])
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
                    start = int(parts[1])
                    end = int(parts[2])
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

                if count == 1:
                    raw_starts[chrom].append(start)
                    raw_ends[chrom].append(end)
                    if raw_strands is not None:
                        raw_strands[chrom].append(st)
                else:
                    raw_starts[chrom].extend([start] * count)
                    raw_ends[chrom].extend([end] * count)
                    if raw_strands is not None:
                        raw_strands[chrom].extend([st] * count)

        starts_np = {c: np.asarray(v, dtype=np.int32) for c, v in raw_starts.items()}
        ends_np = {c: np.asarray(v, dtype=np.int32) for c, v in raw_ends.items()}
        strands_np = (
            {c: np.asarray(v, dtype=np.int8) for c, v in raw_strands.items()}
            if raw_strands is not None
            else None
        )

        return cls(
            starts=starts_np,
            ends=ends_np,
            assay=cfg,
            strands=strands_np,
            metadata={"source": str(path_obj), "source_type": "bedpe" if is_bedpe else "bed"},
            sort=True,
            dedup=dedup,
        )

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
            ``"npz"`` or ``"parquet"`` (optional).
        compressed:
            For ``.npz``, use ``np.savez_compressed`` instead of ``np.savez``
            (smaller files, slightly slower load).
        """
        out_path = Path(path)
        fmt = format.lower() if format is not None else None
        if fmt is None:
            suffix = out_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                fmt = "parquet"
            else:
                fmt = "npz"
                if suffix != ".npz":
                    out_path = out_path.with_suffix(".npz")

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
        """Load a persisted :class:`FragmentStore` from ``.npz`` or ``.parquet``."""
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
    # Ergonomic convenience methods delegating to analytical modules
    # ------------------------------------------------------------------

    def vplot(self, anchors: Any, **kwargs: Any) -> Any:
        """Compute a :class:`~pyvplot.vplot.VPlot` around ``anchors``."""
        from pyvplot.vplot import vplot

        return vplot(self, anchors, **kwargs)

    def to_matrix(self, anchors: Any, **kwargs: Any) -> Any:
        """Compute a size-stratified :class:`~pyvplot.matrices.SignalMatrix` around ``anchors``."""
        from pyvplot.matrices import to_matrix

        return to_matrix(self, anchors, **kwargs)

    def cut_site_footprint(self, anchors: Any, **kwargs: Any) -> Any:
        """Compute 1-bp cleavage cut-site footprints around ``anchors``."""
        from pyvplot.matrices import cut_site_footprint

        return cut_site_footprint(self, anchors, **kwargs)

    def footprint_score(self, anchors: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute per-anchor footprint depth and protection scores."""
        from pyvplot.matrices import footprint_score

        return footprint_score(self, anchors, **kwargs)

    def extract_events(self, anchors: Any, **kwargs: Any) -> pd.DataFrame:
        """Extract per-fragment relative coordinates around ``anchors``."""
        from pyvplot.extract import extract_events

        return extract_events(self, anchors, **kwargs)

    def insert_size_distribution(self, **kwargs: Any) -> pd.DataFrame:
        """Compute insert-size histogram across stored fragments."""
        from pyvplot.qc import insert_size_distribution

        return insert_size_distribution(self, **kwargs)

    def phasogram(self, **kwargs: Any) -> Any:
        """Compute nucleosome phasing pair-distance histogram."""
        from pyvplot.qc import phasogram

        return phasogram(self, **kwargs)

    def frip(self, peaks: Any, **kwargs: Any) -> Any:
        """Compute Fraction of Fragments in Peaks (FRiP)."""
        from pyvplot.qc import frip

        return frip(self, peaks, **kwargs)

    def qc_summary(self, peaks: Any = None) -> dict[str, Any]:
        """Compute summary QC statistics for this store."""
        from pyvplot.qc import qc_summary

        return qc_summary(self, peaks=peaks)

    @classmethod
    def merge(cls, stores: Iterable[FragmentStore], dedup: bool = False) -> FragmentStore:
        """Merge multiple :class:`FragmentStore` instances into one."""
        store_list = list(stores)
        if not store_list:
            raise ValueError("Cannot merge an empty collection of FragmentStores.")
        assay = store_list[0].assay
        is_stranded = store_list[0].is_stranded
        for st in store_list[1:]:
            if st.is_stranded != is_stranded:
                raise ValueError("Cannot merge stranded and unstranded FragmentStores.")

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

        return cls(starts=starts, ends=ends, assay=assay, strands=strands, sort=True, dedup=dedup)
