# pyvplot

Fast, assay-aware Python toolkit for **paired-end fragment geometry, V-plots, 1-bp cleavage cut-site footprints, size-stratified signal matrices, and nucleosome phasing** tailored to **ChEC-seq, ATAC-seq, CUT&RUN / CUT&Tag, MNase-seq, and ChIP-exo**.

## Core Principles

- **Extract once, analyze everything:** Convert BAM, BED/BEDPE, or 10x/ArchR `.frag.tsv.gz` files into a compact, sorted per-chromosome [`FragmentStore`](api/store.md) (`.npz` or `.parquet`). Repeated queries across 10k–100k motif hits (`moodspy`) or TSSs run in milliseconds via binary search + Numba (`@njit(parallel=True)`), with an exact pure-NumPy fallback.
- **Assay physics built in:** Explicit presets for Tn5 shifts (`+4 / -5`), MNase/ChEC cleavage ends, fragment midpoints, strand-separated ChIP-exo 5' ends, and exact 5'→3' strand flipping around `-` strand motifs and TSSs.
- **Clean downstream handoffs:**
  - Tidy DataFrames and `plotnine` / `matplotlib` helpers for single, differential, and faceted multi-condition V-plots.
  - Size-stratified `(n_anchors, n_bins)` [`SignalMatrix`](api/matrices.md) objects ready for `EnrichedHeatmap` (`NormalizedMatrix.from_array(...)` or duck-typed dict).
  - Track export to `bedGraph`, `BigWig`, and `pymisha` (sparse/dense).

## Built-in Assay Presets

| Preset | Endpoint Shift | Default `feature` | Default Size Bins |
| :--- | :--- | :--- | :--- |
| `chec` | `(0, 0)` | `cuts` | `tf`: `20–120`, `mononucleosome`: `130–200` |
| `atac` | `(+4, -5)` | `cuts` | `nfr`: `10–100`, `mononucleosome`: `180–247`, `dinucleosome`: `315–473` |
| `mnase` | `(0, 0)` | `midpoint` | `subnucleosome`: `20–120`, `nucleosome`: `135–165` |
| `cutrun` | `(0, 0)` | `midpoint` | `short`: `20–120`, `nucleosome`: `150–200` |
| `cuttag` | `(0, 0)` | `cuts` | `short`: `20–120`, `nucleosome`: `150–200` |
| `chip_exo` | `(0, 0)` | `cuts_stranded` | Single-end stranded 5' stop sites |

## Where to Go Next

- [Installation](installation.md) — core package and optional extras (`numba`, `bam`, `io`, `plot`, `bigwig`).
- [Quick start](tutorials/quickstart.md) — build a `FragmentStore`, compute a V-plot, and extract matrices.
- [V-plots & differential V-plots](tutorials/vplots-and-diff.md) — CPM & expected/background normalization, smoothing, `diff()`, and faceted `VPlotCollection`.
- [Matrices & cut-site footprints](tutorials/matrices-and-footprints.md) — size-stratified `EnrichedHeatmap` handoffs, 1-bp `+`/`-` footprints, and motif protection scores.
- [Fragment QC & nucleosome phasing](tutorials/qc-and-phasing.md) — insert-size distributions, FRiP, phasograms, and NRL estimation.
- [Cleavage bias & track export](tutorials/bias-and-export.md) — empirical k-mer bias modeling and bedGraph / BigWig / `pymisha` export.
- [Claude Code skill](claude-skill.md) — install the bundled Agent Skill so Claude Code uses `pyvplot` idiomatically.
