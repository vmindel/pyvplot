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

## Heatmaps

pyvplot plots its own heatmaps and depends on no heatmap package.

```python
# One row per anchor, mean profile (with 95% CI band) on top, colour scale 0..99th percentile.
fig, (ax_prof, ax_heat) = mat_tf.normalize("cpm").plot()           # order="mean" (default): strongest rows on top
mat_tf.plot(order=np.argsort(-scores["protection_score"].to_numpy()))   # any row order, or a subset of rows
mat_tf.normalize("row_zscore").plot()                               # negative values -> RdBu_r centred on 0
p = mat_tf.plot_plotnine()                                          # geom_raster; slow beyond ~1e6 cells

# Plain-dict export for any other tool (no pyvplot import needed on the other side)
d = mat_tf.to_dict()   # values, region_index, upstream, downstream, bin_size
```

`to_enriched_heatmap()` was removed; it never found a package to hand off to.

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

# Near vs far end, only fragments overlapping the motif (give anchors start/end for the motif width)
motifs = anchors.assign(start=anchors["center"] - 5, end=anchors["center"] + 5)
fp_nf = vp.cut_site_footprint(frags, motifs, flank=100, len_range="tf",
                              split_by="proximity", motif_overlap="any")
fp_nf.proximal, fp_nf.distal, fp_nf.matrix_proximal   # to_dataframe() columns: pos, end, signal
mat_near = frags.to_matrix(motifs, flank=100, bin_size=1, feature="cuts_proximal", motif_overlap="full")
```

`split_by="proximity"` needs an unstranded store. `cuts_proximal + cuts_distal == cuts` for any `motif_overlap`.

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

## De-Novo Footprint Scanning

To call bound sites without known motif positions (cut pillars and pillar-pair motifs), read `references/pillar-scanning.md`.

## Footguns

- **Sign conventions on `footprint_score`:** `protection_score` is positive when the motif center is protected (`flank > center`), while `footprint_depth` is negative when the motif center dips below the flanks. Check which sign your downstream sort or `BarTrack` expects.
- **`footprint_score` pseudocount units:** `pseudocount` is in raw cuts per bp. With `normalize="cpm"`/`"rpkm"` it is converted to the matrix's units, so `protection_score` equals the raw-count score; it still depends on depth, because the pseudocount weighs more against sparse data.
- **`coverage` units:** `feature="coverage"` computes exact **average base coverage per bin** (covered bp / `bin_size`). At `bin_size=1` this equals exact fragment pileup depth; at `bin_size=10` a fragment covering 5 bp of a bin contributes `0.5`.
- **Feature default:** When `feature=None`, `vp.to_matrix` uses `store.assay.default_feature` (e.g., `"cuts"` for `chec`/`atac`/`cuttag`, `"midpoint"` for `mnase`/`cutrun`).
