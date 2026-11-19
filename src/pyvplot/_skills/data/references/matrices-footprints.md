# Size-Stratified Matrices, Cut-Site Footprints & Protection Scores

## Size-Stratified `SignalMatrix`

```python
import pyvplot as vp

# Accepts explicit inclusive (min_len, max_len) tuples OR named assay size bins
mat_tf  = frags.to_matrix(anchors, flank=1000, bin_size=10, feature="midpoint", len_range="tf")
mat_nuc = frags.to_matrix(anchors, flank=1000, bin_size=10, feature="midpoint", len_range=(135, 180))
mat_cov = frags.to_matrix(anchors, flank=1000, bin_size=10, feature="coverage", len_range=(135, 180))
mat_cut = frags.to_matrix(anchors, flank=100,  bin_size=1,  feature="cuts",     len_range=(20, 120))
```

`vp.SignalMatrix` supports:
- `.normalize(method)` — `"cpm"`, `"rpkm"`, `"zscore"`, `"row_zscore"`, `"row_sum"`
- `.mean_profile(ci=0.95)` — DataFrame (`pos`, `mean`, `std`, `sem`, `ci_low`, `ci_high`)
- `.subset(indices)` — filter/reorder anchor rows
- `.to_dataframe()` — tidy DataFrame (`anchor`, `pos`, `value`)
- Duck-typed dict access (`mat["values"]`, `mat["region_index"]`, `mat["upstream"]`, `mat["downstream"]`, `mat["bin_size"]`) and `np.asarray(mat)`.

## Handoff to `EnrichedHeatmap`

```python
# Returns NormalizedMatrix.from_array(...) only if a module named enrichedheatmap /
# EnrichedHeatmap / pyenrichedheatmap is importable; none exists on PyPI, so in
# practice this is the dict {"values", "region_index", "upstream", "downstream", "bin_size"}.
# Prefer mat.to_dict() or mat.to_dataframe() (long format, for R/ggplot2).
eh_mat = mat_tf.normalize("cpm").to_enriched_heatmap(name="TF_midpoints")
```

## 1-bp Cleavage Cut-Site Footprints

```python
fp = vp.cut_site_footprint(
    frags,
    anchors,
    flank=100,
    bin_size=1,
    len_range=(20, 120),
    split_strands=True,     # extracts cuts_plus & cuts_minus separately + their sum
    normalize="cpm",
    agg="mean",             # "mean" or "sum"
)
df = fp.to_dataframe()      # columns: pos, strand, signal
fig, ax = fp.plot()
p = fp.plot_plotnine()
```

## Per-Anchor Motif Footprint & Protection Scores

```python
scores = vp.footprint_score(
    frags,
    anchors,
    motif_half_width=10,    # protected center: [-10, +10)
    flank_width=30,         # flanking shoulders: [-40, -10) and [+10, +40)
    len_range=(20, 120),
    feature="cuts",
    pseudocount=0.05,
)
```

Returned DataFrame is indexed by `region_index` with columns:
- `center_cuts`, `flank_cuts`, `center_density`, `flank_density`
- `protection_score`: `log2((flank_density + pseudocount) / (center_density + pseudocount))` (**higher / positive = stronger protection**)
- `footprint_depth`: `center_density - flank_density` (**more negative = deeper protection dip**)
- `accessibility`: combined per-bp density across center + flanks.

## Footguns

- **Sign conventions on `footprint_score`:** `protection_score` is positive when the motif center is protected (`flank > center`), while `footprint_depth` is negative when the motif center dips below the flanks. Check which sign your downstream sort or `BarTrack` expects.
- **`coverage` units:** `feature="coverage"` computes exact **average base coverage per bin** (covered bp / `bin_size`). At `bin_size=1` this equals exact fragment pileup depth; at `bin_size=10` a fragment covering 5 bp of a bin contributes `0.5`.
- **Feature default:** When `feature=None`, `vp.to_matrix` uses `store.assay.default_feature` (e.g., `"cuts"` for `chec`/`atac`/`cuttag`, `"midpoint"` for `mnase`/`cutrun`).
