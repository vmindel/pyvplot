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
- **Extract once, query many:** Build a `vp.FragmentStore` from BAM/BED once and `.save("sample.frags.npz")` (or `.parquet`), then reload with `vp.FragmentStore.load(...)`. Measured on 4 cores with Numba warm: 50k anchors on 20M fragments take ~0.3 s for a V-plot and ~0.1 s for a matrix. The first call in a fresh environment compiles kernels (~4 s, cached afterwards).
- **No data?** `store, anchors = vp.datasets.simulate_chec(seed=0)` gives a simulated ChEC store and anchor table; `tf_fragments=0` gives a matched unbound condition with identical anchors. Use it to prototype and to sanity-check an analysis before running on real data.
- **Cluster script:** `examples/bam_to_vplot.py` in the repo goes BAM + motif BED to cached store, QC JSON, V-plot/footprint PNGs and tidy TSVs (for ggplot2). Adapt it rather than writing ingestion from scratch.
- **Coordinates are 0-based, half-open `[start, end)`** everywhere, stored as sorted `int32` per-chromosome arrays (`length = end - start`).
- **Assay endpoint shifts are applied once during ingestion:** `vp.FragmentStore.from_bam(..., assay="atac")` defaults to `apply_shift=True` (`+4 / -5` Tn5 shift), whereas `vp.FragmentStore.from_fragments_bed(...)` defaults to `apply_shift=False` because 10x/ArchR `.frag.tsv.gz` files are typically already shifted. Never double-shift!
- **Strand flipping around `-` anchors:** Reflections follow `x -> 2 * center - x`, so `[start, end)` maps to `[center - end, center - start)`. Midpoints negate, and left (`+`) and right (`-`) cut sites swap and negate.
- **Bin vs. filter bounds:**
  - `vp.vplot(..., min_len=20, max_len=250)` uses **half-open** length bins `[min_len, max_len)` so every bin has uniform width `len_bin` (no double-counting the top row).
  - Everywhere length is a filter it is **inclusive** `min_len <= length <= max_len` and accepts a named assay size bin: `len_range=` in `to_matrix`, `cut_site_footprint`, `footprint_score`, `extract_events`, `phasogram`, `VPlot.marginal_position` and all track exporters; also `store.filter(min_len=..., max_len=...)`. `vplot(min_len="tf")` raises a `TypeError` by design.
  - So `len_range=(20, 120)` selects the same fragments as V-plot rows from `min_len=20, max_len=121`.
- **Feature defaults differ:** `vplot` defaults to `feature="midpoint"`; `to_matrix` and track export default to `store.assay.default_feature` (`"cuts"` for chec/atac/cuttag). Always pass `feature=` explicitly. `"cuts_plus"`/`"cuts_minus"` on unstranded paired-end data mean the fragment's upstream/downstream end in anchor orientation, not read strands.
- **Silent-failure guards:** anchors on chromosomes missing from the store (`chr1` vs `1`) raise a `UserWarning` and give zero rows: fix naming, never ignore it. `normalize("expected")` warns when a length row has no flank background. `KmerBiasTable.from_cuts` raises if no store chromosome is in the sequences.
- **Both functional and method APIs exist:** `vp.vplot(store, anchors, ...)` is identical to `store.vplot(anchors, ...)`, and `vp.to_matrix(store, anchors, ...)` is identical to `store.to_matrix(anchors, ...)`.
