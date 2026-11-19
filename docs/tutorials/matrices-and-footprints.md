# Matrices and cut-site footprints

A V-plot sums over anchors. The tools on this page keep one row per anchor (`to_matrix`, `footprint_score`) or go to base resolution (`cut_site_footprint`), usually for one fragment size class at a time.

The toy data below has two motifs. At the bound one, `motif_bound`, short fragments span the site with ends 12–25 bp from the centre, so the central ±10 bp is never cut. At the unbound one, `motif_unbound`, cuts are uniform.

## Per-anchor matrices

`to_matrix()` returns a `(n_anchors, n_bins)` [`SignalMatrix`](../api/matrices.md). Choose the fragments with `len_range`: an inclusive `(min_len, max_len)` tuple or a size-bin name such as `"tf"`. Choose what is counted with `feature` (see [features](../concepts.md#4-feature-which-point-of-each-fragment-is-counted)). Pass `feature` explicitly: the default is the assay's, which for ChEC is `cuts`.

```python
import numpy as np
import pandas as pd
import pyvplot as vp

anchors = pd.DataFrame(
    {"chrom": ["chr1", "chr1"], "center": [2000, 6000], "strand": ["+", "-"]},
    index=["motif_bound", "motif_unbound"],
)

starts, ends = [], []
# Bound motif at 2000: cleavage shoulders at +/-12..25 bp, protected center [-10, +10)
for offset in range(12, 26):
    for _ in range(5):
        starts.append(2000 - offset)
        ends.append(2000 + offset)
# Unbound motif at 6000: uniform background cuts
for pos in range(5965, 6035):
    starts.append(pos)
    ends.append(pos + 60)

frags = vp.FragmentStore(
    {"chr1": np.asarray(starts, dtype=np.int32)},
    {"chr1": np.asarray(ends, dtype=np.int32)},
    assay="chec",
)

mat_tf = frags.to_matrix(anchors, flank=100, bin_size=10, feature="midpoint", len_range="tf")
print(mat_tf)
print(mat_tf.region_index.tolist())
```

```text
SignalMatrix(shape=(2, 20), window=[-100, +100) bp, bin_size=10, feature='midpoint', lengths=20-120 bp, normalization='raw', assay='chec')
['motif_bound', 'motif_unbound']
```

Rows follow the input order and labels (`region_index`). `mat.values` is a plain NumPy array. `mat.mean_profile(ci=0.95)` gives the average profile with a normal-approximation interval. `mat.subset(idx)` selects rows, for example after sorting by a score. `mat.to_dataframe()` gives a long table (anchor, pos, value) for ggplot2.

For heatmap tools, `mat.to_dict()` returns `values`, `region_index`, `upstream`, `downstream` and `bin_size`. `mat.to_enriched_heatmap()` returns a `NormalizedMatrix` if a Python package exposing `NormalizedMatrix.from_array` is importable, and that same dict otherwise. No such package was found on PyPI when this was written, so treat the dict as the interface.

## Base-resolution cut-site footprints

`cut_site_footprint()` counts fragment ends at 1 bp. With `split_strands=True` it separates each fragment's upstream end (`plus`) from its downstream end (`minus`), in anchor orientation. For unstranded paired-end data these are fragment ends, not read strands. Fragments that contain a protected site put `plus` ends upstream of it and `minus` ends downstream. Fragments that abut it do the reverse.

```python
fp = frags.cut_site_footprint(anchors.loc[["motif_bound"]], flank=40, bin_size=1, agg="sum")
print(fp)
print("plus ends span", fp.pos[fp.plus > 0].min(), "to", fp.pos[fp.plus > 0].max(), "bp")
print("minus ends span", fp.pos[fp.minus > 0].min(), "to", fp.pos[fp.minus > 0].max(), "bp")
print("cuts within 10 bp of the centre:", fp.combined[np.abs(fp.pos) < 10].sum())
```

```text
CutSiteFootprint(pos=[-39.5, 39.5] bp, n_bins=80, n_anchors=1, split_strands=True, agg='sum', has_matrices=True)
plus ends span -24.5 to -11.5 bp
minus ends span 12.5 to 25.5 bp
cuts within 10 bp of the centre: 0.0
```

`fp.plot()` draws both strands; `fp.to_dataframe()` gives a long table (pos, strand, signal). The figure below shows a strand-split footprint next to TF-sized and nucleosome-sized midpoint profiles (`mean_profile(ci=0.95)` of two matrices with different `len_range`):

![1-bp strand-separated cleavage cut-site footprint and size-stratified midpoint profiles](../img/cut_site_footprint.png)

## Per-anchor protection scores

`footprint_score()` compares cut density in the centre `[-motif_half_width, +motif_half_width)` with the two shoulders of width `flank_width` on either side. `protection_score = log2((flank_density + pc) / (center_density + pc))` is positive when the centre is protected; `accessibility` is the mean density over centre and shoulders. Choose `motif_half_width` to cover the expected footprint, not just the motif. The score is per anchor, so with sparse data it is noisy: rank or group anchors with it rather than reading individual values.

```python
scores = frags.footprint_score(anchors, motif_half_width=10, flank_width=25, feature="cuts")
print(scores[["center_cuts", "flank_cuts", "protection_score", "footprint_depth"]].round(2))
```

```text
               center_cuts  flank_cuts  protection_score  footprint_depth
motif_bound            0.0       140.0              5.83             -2.8
motif_unbound         20.0        60.0              0.25             -0.2
```
