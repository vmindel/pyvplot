# Matrices & cut-site footprints

## Size-stratified signal matrices & `EnrichedHeatmap` handoff

`to_matrix()` extracts a `(n_anchors, n_bins)` [`SignalMatrix`](../api/matrices.md) filtered by an explicit inclusive `(min_len, max_len)` range or a named assay size band (e.g., `"tf"`, `"mononucleosome"`, `"nfr"`).

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
print("Shape:", mat_tf.shape, "keys:", mat_tf.keys())
eh_payload = mat_tf.normalize("cpm").to_enriched_heatmap(name="TF_midpoints")
print("Handoff type:", type(eh_payload).__name__)
```

```text
Shape: (2, 20) keys: ('values', 'region_index', 'upstream', 'downstream', 'bin_size')
Handoff type: dict
```

## 1-bp strand-separated cleavage cut-site footprints

`cut_site_footprint()` extracts `+` (sense 5') and `-` (antisense 5') cleavage ends separately with exact 5'→3' reflection on `-` strand anchors:

```python
fp = frags.cut_site_footprint(anchors, flank=40, bin_size=1, split_strands=True, agg="sum")
df_fp = fp.to_dataframe()
print(df_fp.groupby("strand")["signal"].sum().to_dict())
```

```text
{'+': 86.0, '-': 140.0}
```

## Per-anchor motif protection & footprint depth scores

`footprint_score()` summarizes each anchor's flanking-vs-center cleavage ratio, producing a per-anchor table ready to sort heatmaps or attach as a 1D covariate / `BarTrack`:

```python
scores = frags.footprint_score(anchors, motif_half_width=10, flank_width=25, feature="cuts")
print(scores[["center_cuts", "flank_cuts", "protection_score", "footprint_depth"]].round(2))
```

```text
               center_cuts  flank_cuts  protection_score  footprint_depth
motif_bound            0.0       140.0              5.83             -2.8
motif_unbound         20.0        60.0              0.25             -0.2
```
