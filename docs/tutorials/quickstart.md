# Quick start

`pyvplot` follows an **extract once, analyze everything** workflow: convert alignments or fragment files into a sorted per-chromosome [`FragmentStore`](../api/store.md) once, cache it to `.npz` or `.parquet`, and run V-plots, matrices, footprints, and QC in milliseconds.

## Loading from BAM or BED

For real alignments, parse the BAM once and cache it:

```py
import pyvplot as vp

# Paired-end ChEC-seq or ATAC-seq BAM (ATAC automatically applies +4 / -5 Tn5 shift)
frags = vp.FragmentStore.from_bam("sample.bam", assay="chec", min_mapq=20, min_len=10, max_len=1000)
frags.save("sample.frags.npz")

# Reload in future sessions in milliseconds
frags = vp.FragmentStore.load("sample.frags.npz")

# Or ingest 10x / ArchR fragment files or BEDPE directly
atac = vp.FragmentStore.from_fragments_bed("fragments.tsv.gz", assay="atac")
```

## Runnable walkthrough

Here we construct a synthetic ChEC-seq `FragmentStore` with TF-bound fragments (~60 bp) centered at 10 motif sites and flanking nucleosomes (~150 bp), save and reload it, and run our first V-plot and signal matrix.

```python
import numpy as np
import pandas as pd
import pyvplot as vp

rng = np.random.default_rng(42)
centers = np.arange(5_000, 25_000, 2_000)
anchors = pd.DataFrame(
    {
        "chrom": ["chr1"] * len(centers),
        "center": centers,
        "strand": ["+", "-"] * (len(centers) // 2),
    },
    index=[f"motif_{i}" for i in range(len(centers))],
)

starts, ends = [], []
for c in centers:
    for _ in range(25):
        fl = int(rng.integers(50, 70))
        mid = int(c + rng.normal(0, 6))
        starts.append(mid - fl // 2)
        ends.append(mid - fl // 2 + fl)
    for _ in range(50):
        fl = int(rng.integers(145, 160))
        mid = int(c + rng.integers(-250, 250))
        starts.append(mid - fl // 2)
        ends.append(mid - fl // 2 + fl)

frags = vp.FragmentStore(
    starts={"chr1": np.asarray(starts, dtype=np.int32)},
    ends={"chr1": np.asarray(ends, dtype=np.int32)},
    assay="chec",
)
print(frags)
```

```text
FragmentStore(n_fragments=750, n_chroms=1, assay='chec')
```

Persist to `.npz` and reload cleanly:

```python
cache_path = frags.save("chec_demo.frags.npz")
reloaded = vp.FragmentStore.load(cache_path)
print("Reloaded:", reloaded.n_fragments, "fragments,", "assay =", reloaded.assay.name)
```

```text
Reloaded: 750 fragments, assay = chec
```

Compute a V-plot and a size-stratified TF matrix around the same anchors:

```python
v = vp.vplot(reloaded, anchors, flank=250, min_len=20, max_len=200, pos_bin=5, len_bin=5)
mat_tf = reloaded.to_matrix(anchors, flank=250, bin_size=10, feature="midpoint", len_range="tf")
print("VPlot shape:", v.shape, "total hits:", int(v.counts.sum()))
print("TF matrix shape:", mat_tf.shape, "len_range:", mat_tf.len_range)
```

```text
VPlot shape: (36, 100) total hits: 750
TF matrix shape: (10, 50) len_range: (20, 120)
```
