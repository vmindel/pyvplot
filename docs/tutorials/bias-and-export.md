# Cleavage bias & track export

## Modeling & correcting k-mer cleavage bias

Enzymes like MNase and Tn5 exhibit intrinsic DNA sequence cleavage preferences. [`KmerBiasTable`](../api/bias.md) learns empirical k-mer propensities directly from a naked-DNA control `FragmentStore` (or loads a precomputed table) and normalizes 1-bp cut-site matrices.

```python
import numpy as np
import pyvplot as vp

rng = np.random.default_rng(42)
seq = "".join(rng.choice(list("ACGT"), size=10_000))

starts, ends = [], []
for i in range(50, len(seq) - 200):
    if seq[i : i + 4] == "GATC":
        for _ in range(12):
            starts.append(i + 2)
            ends.append(i + 82)
bg = rng.integers(50, len(seq) - 200, size=500)
starts.extend(bg.tolist())
ends.extend((bg + 80).tolist())

naked_dna = vp.FragmentStore(
    {"chr1": np.asarray(starts, dtype=np.int32)},
    {"chr1": np.asarray(ends, dtype=np.int32)},
    assay="chec",
)
bias = vp.KmerBiasTable.from_cuts(naked_dna, {"chr1": seq}, k=4, cut_offset=2)
print(f"Learned bias at GATC: {bias['GATC']:.2f}x (mean across 4-mers: {bias.weights.mean():.2f}x)")
```

```text
Learned bias at GATC: 54.28x (mean across 4-mers: 1.00x)
```

Correct a 1-bp cut-site matrix using anchor sequences:

```python
anchor_seq = "AAAAAAAA" + "GATC" + "AAAAAAAA"
expected = bias.score_sequence(anchor_seq)
raw_mat = vp.SignalMatrix(
    values=expected[None, :].copy(),
    region_index=np.array(["site_0"]),
    upstream=10,
    downstream=10,
    bin_size=1,
    feature="cuts",
)
corrected = bias.correct_matrix(raw_mat, [anchor_seq], preserve_scale=True)
print(f"Std before correction: {raw_mat.values.std():.3f}, after correction: {corrected.values.std():.6f}")
```

```text
Std before correction: 11.611, after correction: 0.000001
```

## Exporting size-stratified tracks

Export binned midpoint, cleavage-cut, or exact average-coverage tracks to a DataFrame, `bedGraph` / `.bedGraph.gz`, `BigWig` (`pyBigWig`), or `pymisha` (sparse/dense):

```python
track_df = vp.to_binned_track(naked_dna, feature="midpoint", len_range="tf", bin_size=100, normalize="cpm")
bg_path = vp.to_bedgraph(naked_dna, "tf_midpoints.bedGraph.gz", feature="midpoint", len_range="tf", bin_size=100)
print("Exported bins:", len(track_df), "file:", bg_path.name)
```

```text
Exported bins: 98 file: tf_midpoints.bedGraph.gz
```

For BigWig or `pymisha`:

```py
vp.to_bigwig(frags, "tf_midpoints.bw", chrom_sizes={"chr1": 248_956_422}, feature="midpoint", len_range="tf")
vp.to_pymisha(frags, "chec.tf_cuts", feature="cuts", len_range="tf", bin_size=10, mode="sparse")
```
