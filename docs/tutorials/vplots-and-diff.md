# V-plots & differential V-plots

A **V-plot** projects fragment midpoints (or cleavage ends) onto relative distance from an anchor center (x-axis) versus fragment length (y-axis). Tightly protected TF binding sites form a sharp V-shaped vertex at short fragment lengths (~40–100 bp), while flanking nucleosomes appear as horizontal clouds around ~150 bp.

## Expected/background normalization

In raw or CPM V-plots, abundant ~150 bp nucleosomal fragments can visually drown out fainter 50–80 bp TF signatures. Dividing each fragment-length row by its expected background rate (`normalize("expected", expected_from="flank")` or `expected_from="genome"`) reveals both regimes simultaneously.

```python
import numpy as np
import pandas as pd
import pyvplot as vp

def simulate_condition(tf_strength: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = np.arange(5_000, 45_000, 2_000)
    anchors = pd.DataFrame(
        {"chrom": ["chr1"] * len(centers), "center": centers, "strand": rng.choice(["+", "-"], len(centers))}
    )
    starts, ends = [], []
    for c in centers:
        for _ in range(tf_strength):
            fl = int(rng.integers(50, 70))
            mid = int(c + rng.normal(0, 5))
            starts.append(mid - fl // 2)
            ends.append(mid - fl // 2 + fl)
        for _ in range(15):
            fl = int(rng.integers(50, 70))
            mid = int(c + rng.integers(-280, 280))
            starts.append(mid - fl // 2)
            ends.append(mid - fl // 2 + fl)
        for _ in range(80):
            fl = int(rng.integers(145, 160))
            mid = int(c + rng.integers(-280, 280))
            starts.append(mid - fl // 2)
            ends.append(mid - fl // 2 + fl)
    store = vp.FragmentStore(
        {"chr1": np.asarray(starts, dtype=np.int32)},
        {"chr1": np.asarray(ends, dtype=np.int32)},
        assay="chec",
    )
    return store, anchors

wt, anchors = simulate_condition(tf_strength=35, seed=1)
mut, _ = simulate_condition(tf_strength=0, seed=2)

vp_wt = vp.vplot(wt, anchors, flank=250, min_len=20, max_len=200, pos_bin=5, len_bin=5)
vp_mut = vp.vplot(mut, anchors, flank=250, min_len=20, max_len=200, pos_bin=5, len_bin=5)

vp_oe = vp_wt.normalize("expected", expected_from="flank")
tf_row = int(np.argmin(np.abs(vp_oe.len_centers - 60)))
center_col = int(np.argmin(np.abs(vp_oe.pos_centers - 0)))
print(f"Expected-normalized enrichment at (pos=0, len=60): {vp_oe.counts[tf_row, center_col]:.2f}x")
```

```text
Expected-normalized enrichment at (pos=0, len=60): 88.70x
```

## Differential V-plots (mutant vs. WT)

`VPlot.diff()` automatically CPM-normalizes raw V-plots before differencing, applies optional 2D Gaussian smoothing, and marks the result with `is_diff=True` so plotting helpers automatically use a symmetric diverging colormap centered at 0.

```python
vp_diff = vp_mut.diff(vp_wt, mode="log2fc", pseudocount=0.5, smooth_sigma=(1.5, 1.5))
print("Normalization:", vp_diff.normalization, "is_diff:", vp_diff.is_diff)
print(f"Mut vs WT log2FC at TF center: {vp_diff.counts[tf_row, center_col]:.2f}")
```

```text
Normalization: diff:log2fc is_diff: True
Mut vs WT log2FC at TF center: -5.37
```

## Multi-condition tidy export for `plotnine`

Bundle labeled V-plots into a [`VPlotCollection`](../api/vplot.md) to export one tidy DataFrame for `plotnine` `facet_grid`:

```python
coll = (
    vp.VPlotCollection()
    .add(vp_wt.normalize("cpm"), tf="CTCF", condition="WT")
    .add(vp_mut.normalize("cpm"), tf="CTCF", condition="Mut")
)
tidy = coll.to_dataframe()
print(tidy.columns.tolist(), "rows:", len(tidy))
```

```text
['pos', 'length', 'value', 'tf', 'condition'] rows: 7200
```

Render with `plotnine` or `matplotlib`:

```py
p = vp_oe.plot_plotnine(cmap="magma", title="WT ChEC-seq (O/E)")
p_facet = coll.plot_plotnine(facet="tf ~ condition")
fig, ax = vp_diff.plot(title="Mutant vs WT (log2FC)")
```
