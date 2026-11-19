# pyvplot

Fast, assay-aware Python toolkit for **paired-end fragment geometry, V-plots, 1-bp cleavage cut-site footprints, size-stratified signal matrices, and nucleosome phasing** tailored to **ChEC-seq, ATAC-seq, CUT&RUN / CUT&Tag, MNase-seq, and ChIP-exo**.

## Design Principles

- **Extract once, analyze everything:** Convert BAM, BED/BEDPE, or 10x/ArchR `.frag.tsv.gz` files into a compact, sorted per-chromosome `FragmentStore` (`.npz` or `.parquet`). Repeated queries across 10k–100k motif hits (`moodspy`) or TSSs run in milliseconds via binary search + Numba (`@njit(parallel=True)`), with an exact pure-NumPy fallback.
- **Assay physics built in:** Explicit presets for Tn5 shifts (`+4 / -5`), MNase/ChEC cleavage ends, fragment midpoints, strand-separated ChIP-exo 5' ends, and exact 5'→3' strand flipping around `-` strand motifs and TSSs.
- **Clean downstream handoffs:**
  - Tidy DataFrames & `plotnine` / `matplotlib` helpers for single, differential, and faceted multi-condition V-plots.
  - Size-stratified `(n_anchors, n_bins)` `SignalMatrix` objects directly compatible with `EnrichedHeatmap` (`NormalizedMatrix.from_array(...)` or duck-typed dict).
  - Track export to `bedGraph`, `BigWig`, and `pymisha` (sparse/dense).

## Installation

```bash
pip install pyvplot

# With optional extras:
pip install "pyvplot[numba,bam,io,plot]"
# Or everything (excluding optional pyBigWig):
pip install "pyvplot[all]"
```

## Quickstart

### 1. Ingest once & cache

```python
from pyvplot import FragmentStore

# From BAM (applies assay shift once, e.g. +4/-5 for ATAC, 0/0 for ChEC)
frags = FragmentStore.from_bam("sample.bam", assay="chec", min_mapq=20, min_len=10, max_len=1000)
frags.save("sample.frags.npz")          # or "sample.frags.parquet"

# Reload instantaneously in future sessions
frags = FragmentStore.load("sample.frags.npz")

# Or ingest 10x / ArchR fragment files or BEDPE directly
atac = FragmentStore.from_fragments_bed("fragments.tsv.gz", assay="atac")
```

### 2. V-Plots & Differential V-Plots

```python
import pandas as pd
from pyvplot import VPlotCollection, vplot

anchors = pd.DataFrame({
    "chrom": ["chr1", "chr1"],
    "center": [10500, 24800],
    "strand": ["+", "-"],
})

vp_wt = vplot(frags_wt, anchors, flank=300, min_len=20, max_len=250, pos_bin=1, len_bin=1)
vp_mut = vplot(frags_mut, anchors, flank=300, min_len=20, max_len=250, pos_bin=1, len_bin=1)

# CPM or row-wise expected/background normalization (prevents nucleosomes from drowning out TF Vs)
vp_wt_cpm = vp_wt.normalize("cpm")
vp_wt_oe = vp_wt.normalize("expected", expected_from="flank")   # or expected_from="genome"

# Differential V-plot (log2FC, smoothed)
vp_diff = vp_mut.diff(vp_wt, mode="log2fc", pseudocount=0.5, smooth_sigma=(1.5, 1.5))

# Plot with plotnine (returns customizable ggplot) or matplotlib
p = vp_wt_oe.plot_plotnine(cmap="magma")
fig, ax = vp_diff.plot()

# Faceted multi-condition export for plotnine
coll = VPlotCollection().add(vp_wt_cpm, tf="CTCF", condition="WT").add(vp_mut.normalize("cpm"), tf="CTCF", condition="Mut")
p_facet = coll.plot_plotnine(facet="tf ~ condition")
```

### 3. Size-Stratified Matrices, Cut-Site Footprints & Protection Scores

```python
# Named assay size bins ("tf", "mononucleosome", "nfr", ...) or explicit (min_len, max_len) tuples
mat_tf  = frags.to_matrix(anchors, flank=1000, bin_size=10, feature="midpoint", len_range="tf")
mat_nuc = frags.to_matrix(anchors, flank=1000, bin_size=10, feature="midpoint", len_range=(135, 180))
mat_cut = frags.to_matrix(anchors, flank=100,  bin_size=1,  feature="cuts",     len_range=(20, 120))

# Drop-in handoff to EnrichedHeatmap (or duck-typed dict)
eh_mat = mat_tf.to_enriched_heatmap(name="TF_fragments")

# 1-bp strand-separated cleavage cut-site footprint
fp = frags.cut_site_footprint(anchors, flank=100, len_range=(20, 120), split_strands=True)
fig, ax = fp.plot()

# Per-anchor protection depth / footprint score (great as a 1D covariate or BarTrack)
scores = frags.footprint_score(anchors, motif_half_width=10, flank_width=30, len_range=(20, 120))
```

### 4. Fragment QC, Nucleosome Phasing (NRL), Bias Correction & Export

```python
from pyvplot import KmerBiasTable, to_bedgraph, to_bigwig, to_pymisha

# Insert-size distribution, FRiP, and nucleosome repeat length (NRL)
qc = frags.qc_summary(peaks="peaks.bed")
ph = frags.phasogram(max_lag=1000, len_range=(135, 165))
print(f"Estimated NRL: {ph.nrl:.1f} bp (R²={ph.r_squared:.3f})")

# Empirical k-mer cleavage bias estimation & correction (e.g. from naked-DNA control)
bias = KmerBiasTable.from_cuts(naked_dna_store, genome_fasta, k=6)
mat_cut_corrected = bias.correct_matrix(mat_cut, anchor_sequences)

# Export size-stratified tracks
to_bedgraph(frags, "tf_midpoints.bedGraph.gz", feature="midpoint", len_range="tf", bin_size=10)
# to_bigwig(frags, "tf_midpoints.bw", chrom_sizes={"chr1": 248956422}, feature="midpoint", len_range="tf")
# to_pymisha(frags, "chec.tf_cuts", feature="cuts", len_range="tf", bin_size=10, mode="sparse")
```

## Built-in Assay Presets

| Preset | Endpoint Shift | Default `feature` | Default Size Bins |
| :--- | :--- | :--- | :--- |
| `chec` | `(0, 0)` | `cuts` | `tf`: `20–120`, `mononucleosome`: `130–200` |
| `atac` | `(+4, -5)` | `cuts` | `nfr`: `10–100`, `mononucleosome`: `180–247`, `dinucleosome`: `315–473` |
| `mnase` | `(0, 0)` | `midpoint` | `subnucleosome`: `20–120`, `nucleosome`: `135–165` |
| `cutrun` | `(0, 0)` | `midpoint` | `short`: `20–120`, `nucleosome`: `150–200` |
| `cuttag` | `(0, 0)` | `cuts` | `short`: `20–120`, `nucleosome`: `150–200` |
| `chip_exo` | `(0, 0)` | `cuts_stranded` | Single-end stranded 5' stop sites |

## Development

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check src tests
```
