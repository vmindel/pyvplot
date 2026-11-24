"""Optional sequence k-mer cleavage bias modeling and naked-DNA control normalization."""

from __future__ import annotations

import gzip
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.assays import get_assay
from pyvplot.extract import normalize_anchors
from pyvplot.matrices import CutSiteFootprint, SignalMatrix
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


_KMER_CHUNK = 4_000_000


def _count_kmers(
    seq: str, k: int, k_starts: np.ndarray, chunk: int = _KMER_CHUNK
) -> tuple[np.ndarray, np.ndarray]:
    """Count every unambiguous k-mer of ``seq`` and those starting at ``k_starts``.

    Works in windows of ``chunk`` k-mers so a whole chromosome is never encoded at once
    (a human chr1 would need several GB). Returns ``(background, observed)`` counts.
    """
    n_codes = 4**k
    background = np.zeros(n_codes, dtype=np.float64)
    observed = np.zeros(n_codes, dtype=np.float64)
    n_kmers = len(seq) - k + 1
    if n_kmers <= 0:
        return background, observed
    k_starts = np.sort(k_starts[(k_starts >= 0) & (k_starts < n_kmers)])
    for a in range(0, n_kmers, chunk):
        b = min(a + chunk, n_kmers)
        codes = _encode_sequence_kmers(seq[a : b + k - 1], k)
        valid = codes[codes >= 0]
        if len(valid):
            background += np.bincount(valid, minlength=n_codes)
        lo, hi = np.searchsorted(k_starts, [a, b])
        if hi > lo:
            hit = codes[k_starts[lo:hi] - a]
            hit = hit[hit >= 0]
            if len(hit):
                observed += np.bincount(hit, minlength=n_codes)
    return background, observed


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


def _fetch_subseq(genome: Any, chrom_cache: dict[str, str | None], chrom: str, start: int, end: int) -> str:
    """Fetch genomic interval ``[start, end)`` padding out-of-bounds or missing chromosomes with ``'N'``."""
    width = end - start
    if width <= 0:
        return ""
    if hasattr(genome, "fetch") and not isinstance(genome, dict):
        try:
            chrom_len = int(genome.get_reference_length(chrom)) if hasattr(genome, "get_reference_length") else None
            s0 = max(0, start)
            e0 = min(chrom_len, end) if chrom_len is not None else max(0, end)
            if e0 <= s0:
                return "N" * width
            core = str(genome.fetch(chrom, s0, e0))
            pad_l = s0 - start
            pad_r = width - pad_l - len(core)
            return ("N" * pad_l) + core + ("N" * max(0, pad_r))
        except (KeyError, ValueError):
            return "N" * width

    if chrom not in chrom_cache:
        chrom_cache[chrom] = _get_chrom_seq(genome, chrom)
    seq = chrom_cache[chrom]
    if seq is None:
        return "N" * width
    s0 = max(0, start)
    e0 = min(len(seq), end)
    if e0 <= s0:
        return "N" * width
    core = seq[s0:e0]
    pad_l = s0 - start
    pad_r = width - pad_l - len(core)
    return ("N" * pad_l) + core + ("N" * max(0, pad_r))


def extract_anchor_sequences(
    anchors: Any,
    sequences: Any,
    flank: int = 100,
    upstream: int | None = None,
    downstream: int | None = None,
    k: int | None = None,
    cut_offset: int | None = None,
    orient: bool = True,
    stranded: bool = False,
) -> list[str]:
    """Extract 5'->3' anchor-oriented DNA sequences from a genome FASTA or ``{chrom: seq}`` dict.

    When ``k`` is given, each sequence is padded by ``cut_offset`` bases on the
    upstream edge and ``k - 1 - cut_offset`` bases on the downstream edge so
    every 1-bp bin in ``[-upstream, +downstream)`` has a complete k-mer for
    [`KmerBiasTable.correct_matrix`][pyvplot.KmerBiasTable.correct_matrix].
    Positions extending beyond chromosome bounds (or on chromosomes absent from
    ``sequences``) are padded with ``'N'`` (neutral bias weight ``1.0``).

    Parameters
    ----------
    anchors:
        Anchor table (DataFrame, dict, or BED path).
    sequences:
        Genome sequence: FASTA path (plain or ``.gz``), ``pysam.FastaFile``,
        ``pyfaidx.Fasta``, or ``{chrom: str}`` dict.
    flank:
        Symmetric half-window (bp) around each anchor center.
    upstream, downstream:
        Optional asymmetric window bounds (bp), overriding ``flank``.
    k:
        Optional k-mer length for edge-exact padding (adds ``k - 1`` bases).
    cut_offset:
        Cut offset within the k-mer (default ``k // 2`` when ``k`` is provided).
    orient:
        If ``True`` (default), reverse-complement ``-`` strand anchors so the
        returned strings are already in 5'->3' anchor orientation. If ``False``,
        ``-`` anchors cover the same genomic window but are returned as the
        genomic ``+`` strand; pass ``strands=`` to ``correct_matrix`` to orient them.
    stranded:
        Set ``True`` for stranded single-end 5'-end stores (e.g. ChIP-exo), where
        each read is stored at its 1-bp 5' base ``[s, s + 1)`` rather than a
        paired-end fragment interval ``[start, end)``.

    Notes
    -----
    For paired-end data, right-end (reverse-frame) cuts need one more base of left
    padding than ``cut_offset`` provides when ``k`` is odd, so with odd ``k`` the
    first bin of a ``cuts_minus`` (or the right-end half of a ``cuts``) expectation
    keeps the neutral weight ``1.0``.
    """
    up = int(flank if upstream is None else upstream)
    dn = int(flank if downstream is None else downstream)
    if k is not None:
        offset = k // 2 if cut_offset is None else int(cut_offset)
        pad_l = offset
        pad_r = int(k) - 1 - offset
    else:
        pad_l = pad_r = 0

    norm = normalize_anchors(anchors)
    if norm.n_anchors == 0:
        return []

    genome = _open_genome(sequences)
    chrom_cache: dict[str, str | None] = {}
    out: list[str] = []

    width = up + dn + pad_l + pad_r
    for i in range(norm.n_anchors):
        chrom = str(norm.chroms[i])
        c = float(norm.centers[i])
        if int(norm.strands[i]) >= 0:
            # extract_events puts the cut at boundary b in bin floor(b - c + up), so bin 0 is
            # boundary ceil(c - up). Centres are half-integers for odd-width motifs, where
            # floor(c) - up would shift every k-mer by one base.
            g_start = int(np.ceil(c - up)) - pad_l
            sub = _fetch_subseq(genome, chrom_cache, chrom, g_start, g_start + width)
        else:
            # '-' anchors: bin floor(c - b + up), so bin 0 is boundary floor(c + up), and the
            # window is read reverse-complemented. For stranded 5'-end stores b is a base index
            # [b, b + 1) rather than a fragment right boundary, shifting the upper slice bound by +1.
            g_end = int(np.floor(c + up)) + int(bool(stranded)) + pad_l
            sub = _fetch_subseq(genome, chrom_cache, chrom, g_end - width, g_end)
            if orient:
                sub = reverse_complement(sub)
        out.append(sub)
    return out


def _check_anchor_rows(anchors: Any, signal_matrix: SignalMatrix) -> None:
    """Raise unless ``anchors`` lists the matrix's rows in the same order."""
    idx = normalize_anchors(anchors).region_index
    if len(idx) != signal_matrix.n_anchors or not np.array_equal(
        np.asarray(idx).astype(str), np.asarray(signal_matrix.region_index).astype(str)
    ):
        raise ValueError(
            "`anchors` must be the anchor table the matrix was built from, in the same order: "
            f"got {len(idx)} anchors for a matrix with {signal_matrix.n_anchors} rows, or the "
            "anchor IDs differ. Sequences are matched to matrix rows by position."
        )


_FEATURE_FRAME = {
    "cuts_plus": "forward",
    "cuts_minus": "reverse",
    "cuts": "both",
    "cuts_stranded": "both",
}


def _is_stranded_assay(name: str) -> bool:
    """Whether the named assay stores one 5' end per read (unknown names count as paired-end)."""
    try:
        return not get_assay(name).paired_end
    except ValueError:
        return False


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
            n_seq = len(seq)

            starts = store.starts[chrom]
            ends = store.ends[chrom]
            if store.strands is not None:
                # One 5' base per read: the cut sits just before the base itself.
                st = store.strands[chrom]
                plus_pos = starts[st >= 0]
                minus_pos = starts[st < 0]
            else:
                # Cuts are the boundaries `start` and `end`. The base just after the left
                # boundary is `start`; in the reverse-complement reading the base just after
                # the right boundary is `end - 1`.
                plus_pos = starts
                minus_pos = ends - 1

            # Left cuts: the k-mer starts `offset` bases before the cut base.
            bg, obs = _count_kmers(seq, k, plus_pos.astype(np.int64) - offset)
            bg_counts += bg
            obs_counts += obs
            # Right cuts: position p on the forward strand is (n_seq - 1 - p) on the
            # reverse complement, scored in that frame.
            rc_cut = (n_seq - 1) - minus_pos.astype(np.int64)
            bg, obs = _count_kmers(reverse_complement(seq), k, rc_cut - offset)
            bg_counts += bg
            obs_counts += obs

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

    def score_sequence_reverse(
        self, seq: str, strand: str | int = "+", stranded: bool = False
    ) -> np.ndarray:
        """Expected propensity of cuts made in the reverse-complement reading frame.

        Returns one value per bin, oriented like [`score_sequence`][pyvplot.KmerBiasTable.score_sequence]
        (5'->3' along the anchor). Bin ``i`` is the boundary just before ``seq[i]``. For
        paired-end fragments (``stranded=False``) the right-end cut at that boundary has
        ``seq[i - 1]`` as the first base after the cut when reading the opposite strand. For
        stranded 5'-end data (``stranded=True``) the cut is just before base ``seq[i]`` itself.
        """
        s = reverse_complement(seq) if strand in {"-", "-1", -1} else seq
        n = len(s)
        out = np.ones(n, dtype=np.float64)
        if n < self.k:
            return out
        sc = self.score_sequence(reverse_complement(s), strand="+")
        if stranded:
            out[:] = sc[::-1]
        elif n > 1:
            out[1:] = sc[:0:-1]
        return out

    def expected_matrix(
        self,
        sequences: Sequence[str],
        strands: Sequence[str | int] | None = None,
        frame: str = "forward",
        stranded: bool = False,
    ) -> np.ndarray:
        """Compute a 2D ``(n_anchors, seq_len)`` matrix of expected per-base cleavage bias.

        ``frame`` selects which cuts the expectation is for: ``"forward"`` (left ends in
        the anchor's orientation), ``"reverse"`` (right ends, read on the opposite strand)
        or ``"both"`` (their mean, for a combined-ends matrix). ``stranded`` is for
        single-end 5'-end stores such as ChIP-exo.
        """
        if frame not in {"forward", "reverse", "both"}:
            raise ValueError(f"Invalid frame {frame!r}; expected 'forward', 'reverse' or 'both'.")
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
            if frame == "forward":
                out[i] = self.score_sequence(seq, strand=st)
            elif frame == "reverse":
                out[i] = self.score_sequence_reverse(seq, strand=st, stranded=stranded)
            else:
                out[i] = 0.5 * (
                    self.score_sequence(seq, strand=st)
                    + self.score_sequence_reverse(seq, strand=st, stranded=stranded)
                )
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
        sequences: Any,
        strands: Sequence[str | int] | None = None,
        pseudocount: float = 1e-6,
        preserve_scale: bool = True,
        anchors: Any = None,
    ) -> SignalMatrix:
        """Bias-correct a 1-bp [`SignalMatrix`][pyvplot.SignalMatrix] using anchor sequences or a genome FASTA.

        The expectation follows ``signal_matrix.feature``: ``"cuts_plus"`` (left ends) is
        scored on the forward strand, ``"cuts_minus"`` (right ends) on the reverse strand
        and ``"cuts"`` as the mean of the two, since a combined matrix holds both kinds of
        cut. Other features (midpoint, coverage, proximal/distal) are rejected.

        Parameters
        ----------
        signal_matrix:
            1-bp resolution ``(n_anchors, n_bins)`` cut-site matrix.
        sequences:
            Either per-anchor DNA sequence strings (length ``n_bins``, or padded
            by ``cut_offset`` on the left and ``k - 1 - cut_offset`` on the right
            for edge-exact k-mer scoring), **or** a genome FASTA path /
            ``pysam.FastaFile`` / ``{chrom: seq}`` dict when ``anchors`` is also
            provided.
        strands:
            Optional per-anchor strands (``"+"`` / ``"-"``). If sequences are
            already oriented 5'->3' along each anchor (or extracted via
            ``anchors=``), leave as ``None``.
        anchors:
            Optional anchor table (DataFrame, dict, or BED path). When provided,
            ``sequences`` is treated as a genome reference and per-anchor
            sequences are extracted automatically with [`extract_anchor_sequences`][pyvplot.extract_anchor_sequences].
        """
        if signal_matrix.bin_size != 1:
            raise ValueError(
                f"Sequence bias correction requires bin_size=1, got bin_size={signal_matrix.bin_size}."
            )
        stranded = _is_stranded_assay(signal_matrix.assay)
        if anchors is not None:
            _check_anchor_rows(anchors, signal_matrix)
            seq_list: Sequence[str] = extract_anchor_sequences(
                anchors,
                sequences,
                upstream=signal_matrix.upstream,
                downstream=signal_matrix.downstream,
                k=self.k,
                cut_offset=self.cut_offset,
                orient=True,
                stranded=stranded,
            )
            strands = None
        else:
            if isinstance(sequences, (str, Path, dict)) or hasattr(sequences, "fetch"):
                raise ValueError(
                    "When passing a genome FASTA or {chrom: seq} dict to `sequences`, "
                    "also pass `anchors=` so per-anchor windows can be extracted."
                )
            seq_list = sequences

        if len(seq_list) != signal_matrix.n_anchors:
            raise ValueError(
                f"Expected {signal_matrix.n_anchors} sequences, got {len(seq_list)}."
            )
        frame = _FEATURE_FRAME.get(signal_matrix.feature)
        if frame is None:
            raise ValueError(
                f"Sequence bias correction applies to cut-site matrices; got feature="
                f"{signal_matrix.feature!r}. Use 'cuts', 'cuts_plus' or 'cuts_minus'."
            )
        exp_full = self.expected_matrix(seq_list, strands=strands, frame=frame, stranded=stranded)
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

    def correct_footprint(
        self,
        footprint: CutSiteFootprint,
        sequences: Any,
        strands: Sequence[str | int] | None = None,
        pseudocount: float = 1e-6,
        preserve_scale: bool = True,
        anchors: Any = None,
    ) -> CutSiteFootprint:
        """Bias-correct a 1-bp [`CutSiteFootprint`][pyvplot.CutSiteFootprint] (strand-split or combined).

        Requires ``footprint`` to have been computed with ``keep_matrices=True``
        (the default) and ``split_by="strand"`` or ``None``. Corrects the
        underlying per-anchor matrices in their respective cleavage frames
        (``+`` on forward, ``-`` on reverse) and re-aggregates the profiles.
        """
        if footprint.split_by == "proximity":
            raise ValueError(
                "Sequence bias correction applies to strand-split ('cuts_plus'/'cuts_minus') "
                "or unsplit ('cuts') footprints, not split_by='proximity'."
            )
        if footprint.matrix_combined is None:
            raise ValueError(
                "correct_footprint requires per-anchor matrices on the CutSiteFootprint; "
                "build it with keep_matrices=True (the default)."
            )

        if anchors is not None:
            m_ref = footprint.matrix_combined
            _check_anchor_rows(anchors, m_ref)
            seq_list: Sequence[str] = extract_anchor_sequences(
                anchors,
                sequences,
                upstream=m_ref.upstream,
                downstream=m_ref.downstream,
                k=self.k,
                cut_offset=self.cut_offset,
                orient=True,
                stranded=_is_stranded_assay(m_ref.assay),
            )
            strands = None
        else:
            seq_list = sequences

        reducer = np.mean if footprint.agg == "mean" else np.sum

        def _prof(m: SignalMatrix) -> np.ndarray:
            if m.n_anchors == 0:
                return np.zeros(m.n_bins, dtype=np.float64)
            return np.asarray(reducer(m.values, axis=0), dtype=np.float64)

        if footprint.matrix_plus is not None and footprint.matrix_minus is not None:
            m_plus = self.correct_matrix(
                footprint.matrix_plus, seq_list, strands=strands,
                pseudocount=pseudocount, preserve_scale=preserve_scale,
            )
            m_minus = self.correct_matrix(
                footprint.matrix_minus, seq_list, strands=strands,
                pseudocount=pseudocount, preserve_scale=preserve_scale,
            )
            m_comb = SignalMatrix(
                values=m_plus.values + m_minus.values,
                region_index=m_plus.region_index.copy(),
                upstream=m_plus.upstream,
                downstream=m_plus.downstream,
                bin_size=m_plus.bin_size,
                feature="cuts",
                len_range=m_plus.len_range,
                n_fragments=m_plus.n_fragments,
                assay=m_plus.assay,
                normalization=m_plus.normalization,
                metadata=dict(m_plus.metadata),
                motif_overlap=footprint.motif_overlap,
            )
            p_plus = _prof(m_plus)
            p_minus = _prof(m_minus)
            return CutSiteFootprint(
                pos=footprint.pos.copy(),
                combined=p_plus + p_minus,
                plus=p_plus,
                minus=p_minus,
                matrix_combined=m_comb,
                matrix_plus=m_plus,
                matrix_minus=m_minus,
                n_anchors=footprint.n_anchors,
                agg=footprint.agg,
                split_by=footprint.split_by,
                motif_overlap=footprint.motif_overlap,
            )

        m_comb = self.correct_matrix(
            footprint.matrix_combined, seq_list, strands=strands,
            pseudocount=pseudocount, preserve_scale=preserve_scale,
        )
        return CutSiteFootprint(
            pos=footprint.pos.copy(),
            combined=_prof(m_comb),
            matrix_combined=m_comb,
            n_anchors=footprint.n_anchors,
            agg=footprint.agg,
            split_by=footprint.split_by,
            motif_overlap=footprint.motif_overlap,
        )


def normalize_by_control(
    signal: SignalMatrix | VPlot | np.ndarray,
    control: SignalMatrix | VPlot | np.ndarray,
    mode: str = "ratio",
    pseudocount: float | None = None,
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
        Pseudocount for ``"ratio"`` and ``"log2fc"``, in the units of the inputs.
        ``None`` (default) adds half a raw count: for a [`VPlot`][pyvplot.VPlot] or
        [`SignalMatrix`][pyvplot.SignalMatrix] that is scaled to its normalisation
        (CPM, per anchor, ...), for plain arrays the arrays are taken to be raw counts.
        A number is added as given, so a value such as ``1.0`` swamps CPM-scaled signal.
    """
    m = mode.lower()
    if m not in {"ratio", "log2fc", "subtract"}:
        raise ValueError(f"Invalid mode {mode!r}; expected 'ratio', 'log2fc', or 'subtract'.")

    if isinstance(signal, VPlot) and isinstance(control, VPlot):
        return signal.diff(control, mode=m, pseudocount=pseudocount, normalize_first=None)

    if isinstance(signal, SignalMatrix) and isinstance(control, SignalMatrix):
        if signal.shape != control.shape:
            raise ValueError(f"Shape mismatch: {signal.shape} vs {control.shape}.")
        if signal.normalization != control.normalization:
            raise ValueError(
                f"Signal and control are in different units: {signal.normalization!r} vs "
                f"{control.normalization!r}. Normalise both the same way first."
            )
        if signal.feature != control.feature or signal.motif_overlap != control.motif_overlap:
            raise ValueError(
                f"Signal and control were built differently: feature {signal.feature!r} vs "
                f"{control.feature!r}, motif_overlap {signal.motif_overlap!r} vs {control.motif_overlap!r}."
            )
        a, b = signal.values, control.values
        if pseudocount is None:
            sa, sb = signal.raw_count_scale(), control.raw_count_scale()
            pseudocount = 0.5 * float(np.sqrt(sa * sb)) if sa is not None and sb is not None else 0.5
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
    if pseudocount is None:
        pseudocount = 0.5
    if m == "ratio":
        return (a + pseudocount) / (b + pseudocount)
    if m == "log2fc":
        return np.log2((a + pseudocount) / (b + pseudocount))
    return a - b
