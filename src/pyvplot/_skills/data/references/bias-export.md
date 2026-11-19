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

# Bias-correct a 1-bp cut-site SignalMatrix
mat_cut = vp.to_matrix(frags, anchors, flank=100, bin_size=1, feature="cuts", len_range=(20, 120))
mat_corr = bias.correct_matrix(mat_cut, anchor_seqs, strands=anchor_strands, preserve_scale=True)
```

## Control-Track Normalization

Divide or subtract a matched control (naked DNA, free MNase, IgG) on a `vp.SignalMatrix`, `vp.VPlot`, or NumPy array:

```python
mat_norm = vp.normalize_by_control(mat_signal, mat_control, mode="log2fc", pseudocount=1.0)
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
- **Orientation of `anchor_seqs`:** Pass `strands=None` if `anchor_seqs` are already oriented 5'->3' along each motif; pass `strands=anchors["strand"]` if `anchor_seqs` were extracted directly from the genomic `+` strand.
- **BigWig chromosome clipping:** Always pass accurate `chrom_sizes` to `vp.to_bigwig` so trailing bin ends are clipped cleanly to chromosome boundaries.
