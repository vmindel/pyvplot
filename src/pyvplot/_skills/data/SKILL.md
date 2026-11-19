---
name: pyvplot
description: Fast, assay-aware Python toolkit for paired-end fragment geometry, V-plots, differential V-plots, 1-bp cleavage cut-site footprints, size-stratified EnrichedHeatmap matrices, nucleosome phasograms/NRL, k-mer bias correction, and bedGraph/BigWig/pymisha track export for ChEC-seq, ATAC-seq, CUT&RUN, CUT&Tag, MNase-seq, and ChIP-exo.
---

# pyvplot

Use this skill whenever the user analyzes paired-end chromatin fragments, builds V-plots or differential V-plots, computes 1-bp cut-site footprints or TF protection scores, extracts size-stratified matrices for `EnrichedHeatmap`, estimates nucleosome repeat length (NRL), or normalizes cleavage sequence bias with `pyvplot`.

## Router

Read only the topic reference file needed for the task:

| Task | Read |
|---|---|
| **Ingesting BAM / BED / `.frag.tsv.gz` / BEDPE**, caching `.npz`/`.parquet`, deduplication, filtering, merging | `references/store-ingestion.md` |
| **V-plots & differential V-plots**, CPM / expected background normalization, 2D smoothing, `plotnine` / `matplotlib`, faceted `VPlotCollection` | `references/vplots.md` |
| **Size-stratified matrices**, `EnrichedHeatmap` handoff, 1-bp cut-site footprints (`+`/`-`), motif `footprint_score` | `references/matrices-footprints.md` |
| **Assay presets & coordinate geometry**, Tn5 `+4/-5` shift, 5'->3' strand flipping, ChIP-exo, asymmetric windows, anchor tables | `references/assays-coordinates.md` |
| **Fragment QC & nucleosome phasing**: insert-size histograms, FRiP, phasograms, NRL (peak fit + FFT), `qc_summary` | `references/qc-phasing.md` |
| **K-mer cleavage bias correction**, naked-DNA control normalization, exporting `bedGraph` / `BigWig` / `pymisha` tracks | `references/bias-export.md` |
| **"Are these V-plot / matrix counts exact?"**: independent brute-force check of strand-aware extraction | `scripts/check_extract.py` |

`scripts/check_extract.py` (path relative to this skill's root) compares `vp.vplot` and `vp.to_matrix` against an independent brute-force O(N*M) reference on `+` and `-` strand anchors. Run `python scripts/check_extract.py --help`.

## Core Conventions

- Import as `import pyvplot as vp`.
- **Extract once, query many:** Build a `vp.FragmentStore` from BAM/BED once and `.save("sample.frags.npz")` (or `.parquet`). Reload with `vp.FragmentStore.load(...)` so queries over 10k–100k anchors run in milliseconds via binary search + Numba.
- **Coordinates are 0-based, half-open `[start, end)`** everywhere, stored as sorted `int32` per-chromosome arrays (`length = end - start`).
- **Assay endpoint shifts are applied once during ingestion:** `vp.FragmentStore.from_bam(..., assay="atac")` defaults to `apply_shift=True` (`+4 / -5` Tn5 shift), whereas `vp.FragmentStore.from_fragments_bed(...)` defaults to `apply_shift=False` because 10x/ArchR `.frag.tsv.gz` files are typically already shifted. Never double-shift!
- **Strand flipping around `-` anchors:** Reflections follow `x -> 2 * center - x`, so `[start, end)` maps to `[center - end, center - start)`. Midpoints negate, and left (`+`) and right (`-`) cut sites swap and negate.
- **Bin vs. filter bounds:**
  - `vp.vplot(..., min_len=20, max_len=250)` uses **half-open** length bins `[min_len, max_len)` so every bin has uniform width `len_bin` (no double-counting the top row).
  - `vp.to_matrix(..., len_range=(min_len, max_len))` and `store.filter(min_len=..., max_len=...)` use **inclusive** bounds `min_len <= length <= max_len` (and accept named assay size bins like `len_range="tf"` or `len_range="mononucleosome"`).
- **Both functional and method APIs exist:** `vp.vplot(store, anchors, ...)` is identical to `store.vplot(anchors, ...)`, and `vp.to_matrix(store, anchors, ...)` is identical to `store.to_matrix(anchors, ...)`.
