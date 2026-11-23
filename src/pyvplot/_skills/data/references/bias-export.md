# Cleavage Sequence Bias Correction & Track Export

## Empirical K-mer Bias Table (`vp.KmerBiasTable`)

Model MNase, Tn5, or DNase sequence cleavage preference from a naked-DNA control or load a precomputed table:

```python
import pyvplot as vp

# Estimate 6-mer cleavage propensity from naked-DNA fragments + FASTA/dict
bias = vp.KmerBiasTable.from_cuts(
    naked_dna_store,
    "genome.fa",            # FASTA path (plain/gzipped), dict[chrom, str], pysam.FastaFile, or pyfaidx.Fasta
    k=6,
    cut_offset=3,           # default k // 2
    pseudocount=1.0,
)

# Or load an existing TSV/CSV/DataFrame (kmer, score)
bias = vp.KmerBiasTable.from_table("tn5_6mer_bias.tsv")

# Bias-correct a 1-bp cut-site SignalMatrix. With anchors= the per-anchor sequences are
# extracted from the genome for you (oriented 5'->3', padded for edge k-mers).
mat_cut = vp.to_matrix(frags, anchors, flank=100, bin_size=1, feature="cuts", len_range=(20, 120))
mat_corr = bias.correct_matrix(mat_cut, "genome.fa", anchors=anchors, preserve_scale=True)

# Same for a strand-split (or unsplit) CutSiteFootprint: corrects matrix_plus on the
# forward frame and matrix_minus on the reverse frame, then re-aggregates the profiles.
fp = vp.cut_site_footprint(frags, anchors, flank=100, len_range=(20, 120))   # keep_matrices=True (default)
fp_corr = bias.correct_footprint(fp, "genome.fa", anchors=anchors)

# Or extract the sequences once and reuse them (e.g. for several matrices on the same anchors)
seqs = vp.extract_anchor_sequences(anchors, "genome.fa", flank=100, k=bias.k, cut_offset=bias.cut_offset)
mat_corr = bias.correct_matrix(mat_cut, seqs)
```

## Control-Track Normalization

Divide or subtract a matched control (naked DNA, free MNase, IgG) on a `vp.SignalMatrix`, `vp.VPlot`, or NumPy array. Both inputs must be in the same units. The default pseudocount is half a raw count in those units, so CPM-scaled inputs are not swamped; an explicit `pseudocount=1.0` is only sensible for raw counts:

```python
mat_norm = vp.normalize_by_control(mat_signal, mat_control, mode="log2fc")
```

## Exporting Tracks (`bedGraph`, `BigWig`, `pymisha`)

```python
# Non-zero binned track DataFrame (chrom, start, end, value)
df = vp.to_binned_track(frags, feature="midpoint", len_range="tf", bin_size=10, normalize="cpm")

# Sorted bedGraph (plain or .gz, zero extra dependencies)
vp.to_bedgraph(frags, "tf_midpoints.bedGraph.gz", feature="midpoint", len_range="tf", bin_size=10)

# BigWig (requires pyvplot[bigwig] -> pyBigWig)
vp.to_bigwig(frags, "tf_midpoints.bw", chrom_sizes={"chr1": 248956422}, feature="midpoint", len_range="tf")

# pymisha sparse or dense track (requires pymisha initialized). Calls signature-checked
# against pymisha 0.20.1 but not exercised against a live database in the test suite.
vp.to_pymisha(frags, "chec.tf_cuts", feature="cuts", len_range="tf", bin_size=10, mode="sparse")
vp.to_pymisha(frags, "chec.nuc_mid", feature="midpoint", len_range="mononucleosome", bin_size=10, mode="dense")
```

## Footguns

- **1-bp requirement for `correct_matrix`:** `bias.correct_matrix` requires `signal_matrix.bin_size == 1`. Sequence length must equal `n_bins` (or `n_bins + k - 1` padded by `cut_offset` left and `k - 1 - cut_offset` right for edge-exact k-mers).
- **Prefer `anchors=` / `vp.extract_anchor_sequences` to slicing the genome yourself.** For a `+` anchor bin 0's cut is at `ceil(center - upstream)`; for a `-` anchor at `floor(center + upstream)`, read reverse-complemented. Odd-width motifs have half-integer centres, so `int(center) - upstream` puts every k-mer one base off on `+` anchors.
- **`anchors=` must be the table the matrix was built from, in the same order.** Rows are matched to sequences by position; a reordered or subset table raises `ValueError`.
- **Orientation of hand-extracted sequences:** Pass `strands=None` if they are already oriented 5'->3' along each motif (as `extract_anchor_sequences` returns them); pass `strands=anchors["strand"]` if they were taken from the genomic `+` strand.
- **`correct_footprint` needs per-anchor matrices** (`keep_matrices=True`, the default) and rejects `split_by="proximity"`.
- **BigWig chromosome clipping:** Always pass accurate `chrom_sizes` to `vp.to_bigwig` so trailing bin ends are clipped cleanly to chromosome boundaries.
