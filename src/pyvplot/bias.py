"""Optional sequence k-mer cleavage bias modeling and naked-DNA control normalization."""

from __future__ import annotations

import gzip
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.matrices import SignalMatrix
from pyvplot.store import FragmentStore
from pyvplot.vplot import VPlot

_BASE_TO_INT = np.full(256, -1, dtype=np.int8)
_BASE_TO_INT[ord("A")] = 0
_BASE_TO_INT[ord("a")] = 0
_BASE_TO_INT[ord("C")] = 1
_BASE_TO_INT[ord("c")] = 1
_BASE_TO_INT[ord("G")] = 2
_BASE_TO_INT[ord("g")] = 2
_BASE_TO_INT[ord("T")] = 3
_BASE_TO_INT[ord("t")] = 3

_RC_TABLE = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence string."""
    return seq.translate(_RC_TABLE)[::-1]


def _encode_kmer(kmer: str) -> int:
    codes = _BASE_TO_INT[np.frombuffer(kmer.encode("ascii"), dtype=np.uint8)]
    if np.any(codes < 0):
        return -1
    idx = 0
    for c in codes:
        idx = (idx << 2) | int(c)
    return idx


def _encode_sequence_kmers(seq: str, k: int) -> np.ndarray:
    """Encode all overlapping k-mers of ``seq`` into integer codes (``-1`` if ambiguous base)."""
    n = len(seq)
    if n < k:
        return np.empty(0, dtype=np.int64)
    raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    bases = _BASE_TO_INT[raw].astype(np.int64)
    n_kmers = n - k + 1
    codes = np.zeros(n_kmers, dtype=np.int64)
    valid = np.ones(n_kmers, dtype=np.bool_)
    for offset in range(k):
        b = bases[offset : offset + n_kmers]
        valid &= b >= 0
        codes = (codes << 2) | np.where(b >= 0, b, 0)
    codes[~valid] = -1
    return codes


def _read_fasta(path: Path) -> dict[str, str]:
    """Read a plain or gzipped FASTA into ``{name: sequence}`` (name = first word of header)."""
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(chunks)
                name = line[1:].split()[0] if line[1:].strip() else ""
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        out[name] = "".join(chunks)
    return out


def _open_genome(sequences: Any) -> Any:
    """Open a FASTA path (via ``pysam.FastaFile`` when available); pass other inputs through."""
    if not isinstance(sequences, (str, Path)):
        return sequences
    path = Path(sequences)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    try:
        import pysam

        return pysam.FastaFile(str(path))
    except (ImportError, OSError, ValueError):
        # No pysam, plain-gzip FASTA, or no write access for the .fai index.
        return _read_fasta(path)


def _get_chrom_seq(sequences: Any, chrom: str) -> str | None:
    """Fetch a chromosome DNA sequence from a dict, pyfaidx.Fasta, or pysam.FastaFile."""
    if isinstance(sequences, dict):
        return sequences.get(chrom)
    if hasattr(sequences, "fetch"):
        # pysam.FastaFile
        try:
            return str(sequences.fetch(chrom))
        except KeyError:
            return None
    if chrom in sequences:
        return str(sequences[chrom][:].seq if hasattr(sequences[chrom][:], "seq") else sequences[chrom])
    return None


class KmerBiasTable:
    """K-mer cleavage bias table for MNase, Tn5, or DNase cut-site correction.

    Parameters
    ----------
    k:
        K-mer length (typically 6 or 8).
    weights:
        1D ``float64`` array of length ``4**k`` giving relative cleavage
        propensity for each 2-bit encoded k-mer.
    cut_offset:
        0-based offset within the k-mer aligned to the cleavage position
        (default ``k // 2``, centering the k-mer symmetrically around the cut).
    """

    def __init__(
        self,
        k: int,
        weights: np.ndarray,
        cut_offset: int | None = None,
    ) -> None:
        if k < 1 or k > 12:
            raise ValueError(f"k must be between 1 and 12, got {k}.")
        expected_size = 4**k
        w = np.ascontiguousarray(weights, dtype=np.float64)
        if w.shape != (expected_size,):
            raise ValueError(f"Expected weights of shape ({expected_size},), got {w.shape}.")
        self.k: int = int(k)
        self.weights: np.ndarray = w
        self.cut_offset: int = k // 2 if cut_offset is None else int(cut_offset)
        if not (0 <= self.cut_offset < self.k):
            raise ValueError(f"cut_offset must be in [0, {self.k - 1}], got {self.cut_offset}.")

    def __repr__(self) -> str:
        return (
            f"KmerBiasTable(k={self.k}, cut_offset={self.cut_offset}, "
            f"weight_range=[{float(self.weights.min()):.3g}, {float(self.weights.max()):.3g}])"
        )

    def __getitem__(self, kmer: str) -> float:
        if len(kmer) != self.k:
            raise KeyError(f"Expected {self.k}-mer, got {kmer!r}.")
        idx = _encode_kmer(kmer)
        if idx < 0:
            return 1.0
        return float(self.weights[idx])

    @classmethod
    def from_dict(
        cls,
        kmer_scores: dict[str, float],
        cut_offset: int | None = None,
        normalize: bool = True,
    ) -> KmerBiasTable:
        """Build a [`KmerBiasTable`][pyvplot.KmerBiasTable] from a mapping ``{kmer: propensity}``."""
        if not kmer_scores:
            raise ValueError("kmer_scores dictionary cannot be empty.")
        first_kmer = next(iter(kmer_scores))
        k = len(first_kmer)
        weights = np.ones(4**k, dtype=np.float64)
        seen_mask = np.zeros(4**k, dtype=np.bool_)
        for kmer, val in kmer_scores.items():
            if len(kmer) != k:
                raise ValueError(f"Inconsistent k-mer length: expected {k}, got {kmer!r}.")
            idx = _encode_kmer(kmer)
            if idx >= 0:
                weights[idx] = float(val)
                seen_mask[idx] = True
        if normalize and np.any(seen_mask):
            mean_val = float(np.mean(weights[seen_mask]))
            if mean_val > 0.0:
                weights[seen_mask] /= mean_val
        return cls(k=k, weights=weights, cut_offset=cut_offset)

    @classmethod
    def from_table(
        cls,
        table: str | Path | pd.DataFrame,
        kmer_col: str | None = None,
        score_col: str | None = None,
        cut_offset: int | None = None,
        normalize: bool = True,
    ) -> KmerBiasTable:
        """Load a k-mer bias table from a TSV/CSV file or `pandas.DataFrame`."""
        if isinstance(table, (str, Path)):
            p = Path(table)
            sep = "," if p.suffix.lower() == ".csv" else "\t"
            df = pd.read_csv(p, sep=sep)
        else:
            df = table

        if len(df.columns) < 2:
            raise ValueError("Bias table must have at least two columns (kmer and score).")
        k_col = df.columns[0] if kmer_col is None else kmer_col
        s_col = df.columns[1] if score_col is None else score_col
        scores = dict(zip(df[k_col].astype(str), df[s_col].astype(float), strict=True))
        return cls.from_dict(scores, cut_offset=cut_offset, normalize=normalize)

    @classmethod
    def from_cuts(
        cls,
        store: FragmentStore,
        sequences: Any,
        k: int = 6,
        cut_offset: int | None = None,
        chroms: Collection[str] | None = None,
        pseudocount: float = 1.0,
    ) -> KmerBiasTable:
        """Estimate empirical k-mer cleavage bias from naked-DNA control cuts and sequences.

        Computes ``(obs_freq + pseudocount) / (bg_freq + pseudocount)`` normalized
        to unit mean across all ``4**k`` k-mers. Right-end (minus-strand) cuts
        are reverse-complemented into the same 5'->3' cleavage frame as left-end cuts.

        Parameters
        ----------
        store:
            Control store (naked DNA digested with the same enzyme).
        sequences:
            Genome sequence: a FASTA path (plain, or bgzipped with an index),
            a ``pysam.FastaFile`` / ``pyfaidx.Fasta``, or a ``{chrom: str}`` dict.
            Chromosome names must match the store's.
        k:
            K-mer length. ``4**k`` weights are estimated, so k=6 needs on the
            order of millions of cuts.
        cut_offset:
            Position of the cut inside the k-mer (default ``k // 2``).
        chroms:
            Restrict estimation to these chromosomes.
        pseudocount:
            Added to observed and background k-mer counts.

        Raises
        ------
        ValueError
            If none of the store's chromosomes are found in ``sequences``.
        """
        sequences = _open_genome(sequences)
        offset = k // 2 if cut_offset is None else int(cut_offset)
        n_kmers = 4**k
        obs_counts = np.zeros(n_kmers, dtype=np.float64)
        bg_counts = np.zeros(n_kmers, dtype=np.float64)

        target_chroms = list(store.chroms) if chroms is None else [c for c in store.chroms if c in set(chroms)]
        n_used = 0
        for chrom in target_chroms:
            seq = _get_chrom_seq(sequences, chrom)
            if seq is None or len(seq) < k:
                continue
            n_used += 1
            codes_fwd = _encode_sequence_kmers(seq, k)
            valid_bg = codes_fwd[codes_fwd >= 0]
            if len(valid_bg) > 0:
                bg_counts += np.bincount(valid_bg, minlength=n_kmers)

            # Reverse-complement codes for minus-strand cuts
            rc_seq = reverse_complement(seq)
            codes_rc = _encode_sequence_kmers(rc_seq, k)
            valid_rc_bg = codes_rc[codes_rc >= 0]
            if len(valid_rc_bg) > 0:
                bg_counts += np.bincount(valid_rc_bg, minlength=n_kmers)

            starts = store.starts[chrom]
            ends = store.ends[chrom]
            n_seq = len(seq)

            if store.strands is not None:
                st = store.strands[chrom]
                plus_pos = starts[st >= 0]
                minus_pos = starts[st < 0]
            else:
                plus_pos = starts
                minus_pos = ends

            # Forward / left cuts: k-mer starts at pos - offset
            k_starts_fwd = plus_pos - offset
            ok_f = (k_starts_fwd >= 0) & (k_starts_fwd < len(codes_fwd))
            if np.any(ok_f):
                c_f = codes_fwd[k_starts_fwd[ok_f]]
                c_f = c_f[c_f >= 0]
                if len(c_f) > 0:
                    obs_counts += np.bincount(c_f, minlength=n_kmers)

            # Reverse / right cuts: position p on forward maps to (n_seq - 1 - p) on rc_seq
            rc_pos = (n_seq - 1) - minus_pos
            k_starts_rc = rc_pos - offset
            ok_r = (k_starts_rc >= 0) & (k_starts_rc < len(codes_rc))
            if np.any(ok_r):
                c_r = codes_rc[k_starts_rc[ok_r]]
                c_r = c_r[c_r >= 0]
                if len(c_r) > 0:
                    obs_counts += np.bincount(c_r, minlength=n_kmers)

        if n_used == 0:
            raise ValueError(
                f"None of the store's chromosomes ({', '.join(target_chroms[:5])}"
                f"{' ...' if len(target_chroms) > 5 else ''}) were found in `sequences`. "
                "Check chromosome naming ('chr1' vs '1')."
            )
        obs_freq = (obs_counts + pseudocount) / np.sum(obs_counts + pseudocount)
        bg_freq = (bg_counts + pseudocount) / np.sum(bg_counts + pseudocount)
        weights = obs_freq / bg_freq
        mean_w = float(np.mean(weights))
        if mean_w > 0.0:
            weights /= mean_w
        return cls(k=k, weights=weights, cut_offset=offset)

    def score_sequence(self, seq: str, strand: str | int = "+") -> np.ndarray:
        """Compute per-base expected cleavage propensity along ``seq``.

        If ``strand`` is ``"-"`` or ``-1``, the sequence is reverse-complemented
        prior to scoring so the returned profile is oriented 5'->3' along the anchor.
        """
        s = reverse_complement(seq) if strand in {"-", "-1", -1} else seq
        n = len(s)
        scores = np.ones(n, dtype=np.float64)
        if n < self.k:
            return scores
        codes = _encode_sequence_kmers(s, self.k)
        valid = codes >= 0
        # k-mer starting at i maps to cut position i + cut_offset
        pos_idx = np.arange(len(codes), dtype=np.int64) + self.cut_offset
        in_bounds = valid & (pos_idx >= 0) & (pos_idx < n)
        scores[pos_idx[in_bounds]] = self.weights[codes[in_bounds]]
        return scores

    def expected_matrix(
        self,
        sequences: Sequence[str],
        strands: Sequence[str | int] | None = None,
    ) -> np.ndarray:
        """Compute a 2D ``(n_anchors, seq_len)`` matrix of expected per-base cleavage bias."""
        if len(sequences) == 0:
            return np.empty((0, 0), dtype=np.float64)
        seq_len = len(sequences[0])
        out = np.ones((len(sequences), seq_len), dtype=np.float64)
        for i, seq in enumerate(sequences):
            if len(seq) != seq_len:
                raise ValueError(
                    f"All anchor sequences must have equal length ({seq_len}), got {len(seq)} at index {i}."
                )
            st = "+" if strands is None else strands[i]
            out[i] = self.score_sequence(seq, strand=st)
        return out

    @staticmethod
    def correct_profile(
        observed: np.ndarray,
        expected: np.ndarray,
        pseudocount: float = 1e-6,
        preserve_scale: bool = True,
    ) -> np.ndarray:
        """Normalize an observed 1D cut-site profile by its expected sequence bias."""
        obs = np.asarray(observed, dtype=np.float64)
        exp = np.asarray(expected, dtype=np.float64)
        if obs.shape != exp.shape:
            raise ValueError(f"Shape mismatch: observed {obs.shape} vs expected {exp.shape}.")
        corrected = obs / (exp + pseudocount)
        if preserve_scale:
            obs_sum = float(np.sum(obs))
            cor_sum = float(np.sum(corrected))
            if obs_sum > 0.0 and cor_sum > 0.0:
                corrected = corrected * (obs_sum / cor_sum)
        return corrected

    def correct_matrix(
        self,
        signal_matrix: SignalMatrix,
        sequences: Sequence[str],
        strands: Sequence[str | int] | None = None,
        pseudocount: float = 1e-6,
        preserve_scale: bool = True,
    ) -> SignalMatrix:
        """Bias-correct a 1-bp [`SignalMatrix`][pyvplot.SignalMatrix] using anchor sequences.

        Parameters
        ----------
        signal_matrix:
            1-bp resolution ``(n_anchors, n_bins)`` cut-site matrix.
        sequences:
            DNA sequences spanning each anchor window (length ``n_bins``, or
            padded by ``cut_offset`` on the left and ``k - 1 - cut_offset`` on
            the right for edge-exact k-mer scoring).
        strands:
            Optional per-anchor strands (``"+"`` / ``"-"``). If sequences are
            already oriented 5'->3' along each anchor, leave as ``None``.
        """
        if signal_matrix.bin_size != 1:
            raise ValueError(
                f"Sequence bias correction requires bin_size=1, got bin_size={signal_matrix.bin_size}."
            )
        if len(sequences) != signal_matrix.n_anchors:
            raise ValueError(
                f"Expected {signal_matrix.n_anchors} sequences, got {len(sequences)}."
            )
        exp_full = self.expected_matrix(sequences, strands=strands)
        n_bins = signal_matrix.n_bins
        pad_left = self.cut_offset
        pad_right = self.k - 1 - self.cut_offset

        if exp_full.shape[1] == n_bins:
            exp = exp_full
        elif exp_full.shape[1] == n_bins + pad_left + pad_right:
            exp = exp_full[:, pad_left : pad_left + n_bins]
        else:
            raise ValueError(
                f"Sequence length ({exp_full.shape[1]}) must equal n_bins ({n_bins}) "
                f"or n_bins + k - 1 ({n_bins + pad_left + pad_right})."
            )

        corrected = signal_matrix.values / (exp + pseudocount)
        if preserve_scale:
            obs_sum = float(np.sum(signal_matrix.values))
            cor_sum = float(np.sum(corrected))
            if obs_sum > 0.0 and cor_sum > 0.0:
                corrected = corrected * (obs_sum / cor_sum)

        return SignalMatrix(
            values=np.ascontiguousarray(corrected, dtype=np.float64),
            region_index=signal_matrix.region_index.copy(),
            upstream=signal_matrix.upstream,
            downstream=signal_matrix.downstream,
            bin_size=signal_matrix.bin_size,
            feature=signal_matrix.feature,
            len_range=signal_matrix.len_range,
            n_fragments=signal_matrix.n_fragments,
            assay=signal_matrix.assay,
            normalization=f"{signal_matrix.normalization}+kmer_bias",
            metadata={**signal_matrix.metadata, "bias_k": self.k},
            motif_overlap=signal_matrix.motif_overlap,
        )


def normalize_by_control(
    signal: SignalMatrix | VPlot | np.ndarray,
    control: SignalMatrix | VPlot | np.ndarray,
    mode: str = "ratio",
    pseudocount: float = 1.0,
) -> SignalMatrix | VPlot | np.ndarray:
    """Normalize a signal matrix, V-plot, or array against a naked-DNA / IgG control.

    Parameters
    ----------
    signal, control:
        Matching [`SignalMatrix`][pyvplot.SignalMatrix], [`VPlot`][pyvplot.VPlot],
        or NumPy arrays of identical shape.
    mode:
        ``"ratio"`` (``(signal + pc) / (control + pc)``), ``"log2fc"``
        (``log2((signal + pc) / (control + pc))``), or ``"subtract"``
        (``signal - control``).
    pseudocount:
        Pseudocount for ``"ratio"`` and ``"log2fc"``.
    """
    m = mode.lower()
    if m not in {"ratio", "log2fc", "subtract"}:
        raise ValueError(f"Invalid mode {mode!r}; expected 'ratio', 'log2fc', or 'subtract'.")

    if isinstance(signal, VPlot) and isinstance(control, VPlot):
        return signal.diff(control, mode=m, pseudocount=pseudocount, normalize_first=None)

    if isinstance(signal, SignalMatrix) and isinstance(control, SignalMatrix):
        if signal.shape != control.shape:
            raise ValueError(f"Shape mismatch: {signal.shape} vs {control.shape}.")
        if signal.feature != control.feature or signal.motif_overlap != control.motif_overlap:
            raise ValueError(
                f"Signal and control were built differently: feature {signal.feature!r} vs "
                f"{control.feature!r}, motif_overlap {signal.motif_overlap!r} vs {control.motif_overlap!r}."
            )
        a, b = signal.values, control.values
        if m == "ratio":
            vals = (a + pseudocount) / (b + pseudocount)
        elif m == "log2fc":
            vals = np.log2((a + pseudocount) / (b + pseudocount))
        else:
            vals = a - b
        return SignalMatrix(
            values=np.ascontiguousarray(vals, dtype=np.float64),
            region_index=signal.region_index.copy(),
            upstream=signal.upstream,
            downstream=signal.downstream,
            bin_size=signal.bin_size,
            feature=signal.feature,
            len_range=signal.len_range,
            n_fragments=signal.n_fragments,
            assay=signal.assay,
            normalization=f"control_{m}",
            metadata=dict(signal.metadata),
            motif_overlap=signal.motif_overlap,
        )

    a = np.asarray(signal, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    if m == "ratio":
        return (a + pseudocount) / (b + pseudocount)
    if m == "log2fc":
        return np.log2((a + pseudocount) / (b + pseudocount))
    return a - b
