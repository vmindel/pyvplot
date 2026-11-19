# How pyvplot works

Every analysis in pyvplot is the same short pipeline, and every function takes the same handful of decisions as arguments. Most V-plots that look wrong come down to one of these six choices, so this page goes through them in order.

```text
BAM / fragments file
   │  (1) ingest: assay preset, endpoint shift, read filters
   ▼
FragmentStore  ── save() / load() ──▶  sample.frags.npz
   │  (2) anchors   (3) window   (4) feature   (5) length range
   ▼
VPlot · SignalMatrix · CutSiteFootprint · footprint_score table
   │  (6) normalise
   ▼
plot · tidy DataFrame for ggplot2/plotnine · bedGraph / BigWig / misha track
```

## Which function do I want?

| Question | Call | Returns |
| :--- | :--- | :--- |
| Where, relative to my motifs, do fragments of each length sit? | `store.vplot(anchors, ...)` | [`VPlot`](api/vplot.md): 2D counts, length × position |
| Did that change between conditions? | `v_mut.diff(v_wt)` | `VPlot` with `is_diff=True` |
| Per-motif signal from one size class, as a heatmap | `store.to_matrix(anchors, len_range="tf").plot()` | [`SignalMatrix`](api/matrices.md): anchors × bins |
| Base-resolution cleavage around a motif, by strand | `store.cut_site_footprint(anchors)` | [`CutSiteFootprint`](api/matrices.md) |
| One protection number per motif | `store.footprint_score(anchors)` | DataFrame indexed by anchor |
| Every fragment near every anchor, for custom analysis | `store.extract_events(anchors)` | DataFrame, one row per fragment |
| Library QC: lengths, size classes, FRiP | `store.qc_summary(peaks=...)` | dict |
| Nucleosome repeat length | `store.phasogram(len_range=...)` | [`PhasogramResult`](api/qc.md) |
| Genome-wide track of one size class | `vp.to_bedgraph(store, ...)`, `to_bigwig`, `to_pymisha` | file / misha track |

Every `store.method(...)` is the same as the module function `vp.method(store, ...)`, with the same arguments.

## 1. Ingestion: the store and the assay preset

A [`FragmentStore`](api/store.md) holds one interval per fragment, sorted per chromosome, in 0-based half-open coordinates `[start, end)`, so `length = end - start`. Reading a BAM is a single-threaded pass over every read, so build the store once per sample and cache it:

```py
store = vp.FragmentStore.from_bam("sample.bam", assay="chec")
store.save("sample.frags.npz")
store = vp.FragmentStore.load("sample.frags.npz")   # later sessions
```

The `assay` preset sets three things:

| Preset | Endpoint shift at ingestion | Default `feature` | Named size bins (inclusive, bp) |
| :--- | :--- | :--- | :--- |
| `chec` | none | `cuts` | `tf` 20–120, `mononucleosome` 130–200 |
| `atac` | `+4` start / `−5` end (Tn5) | `cuts` | `nfr` 10–100, `mononucleosome` 180–247, `dinucleosome` 315–473 |
| `mnase` | none | `midpoint` | `subnucleosome` 20–120, `nucleosome` 135–165 |
| `cutrun` | none | `midpoint` | `short` 20–120, `nucleosome` 150–200 |
| `cuttag` | none | `cuts` | `short` 20–120, `nucleosome` 150–200 |
| `chip_exo` | none; one 1-bp event per read 5′ end, with strand | `cuts_stranded` | none |
| `generic` | none | `midpoint` | none |

These bins are starting points, not calibrated values. Check them against your own insert-size distribution (`store.insert_size_distribution()`), and define your own with `vp.register_assay(vp.get_assay("chec").with_overrides(name="chec_mine", size_bins={...}))`.

**The Tn5 shift is applied once, at ingestion.** `from_bam` applies it by default (`apply_shift=True`). `from_fragments_bed` does not, because 10x and ArchR fragment files are already shifted. Shifting twice moves every ATAC cut by 4–5 bp and shortens every fragment by 9 bp.

**Read filters in `from_bam`.** The defaults are MAPQ ≥ 20 and fragment length 10–1000 bp. Unmapped, mate-unmapped, secondary, supplementary and QC-fail reads are dropped. **Duplicates are kept** unless you pass `dedup=True` (drop flagged duplicates and identical coordinates), `"flag"` or `"coords"`. `pairing="read1"` (the default) takes each fragment from read 1's template length. `pairing="mate"` pairs mates by name and requires both to pass MAPQ, which costs memory proportional to unpaired reads in flight.

## 2. Anchors

Anchors are the positions you aggregate around: motif hits, TSSs, peak summits. Every analysis function accepts any of these:

- a BED path: strand from column 6, and column 4 (name) becomes the row label;
- a DataFrame with `chrom` plus `center` (or `pos`/`midpoint`/`summit`), or `chrom` plus `start` and `end`, and an optional `strand` column;
- a dict of those columns.

The centre of a `start`/`end` interval is `(start + end) / 2`. For an even-length motif that is the boundary between its two middle bases; for an odd-length one it is the middle of the central base.

**Minus-strand anchors are mirrored**, so all outputs are in anchor orientation: negative positions are upstream (5′ of the motif or TSS), and `upstream=`/`downstream=` mean the same thing for every anchor. Unstranded anchors are treated as `+`.

Row order and labels are kept: `SignalMatrix.region_index` and the `footprint_score` index match your input. If an anchor's chromosome is not in the store (most often `chr1` versus `1`), you get a warning and that anchor's row is zero.

## 3. Window

`flank=300` means positions `[-300, +300)` around each centre. Use `upstream=`/`downstream=` for asymmetric windows, such as `upstream=1000, downstream=3000` around TSSs. The window must be a whole number of bins (`pos_bin` for V-plots, `bin_size` for matrices).

## 4. Feature: which point of each fragment is counted

| `feature` | What is placed on the position axis | Counts per fragment | Typical use |
| :--- | :--- | :--- | :--- |
| `midpoint` | `(start + end) / 2` | 1 | V-plots; nucleosome positions; TF occupancy from short fragments |
| `cuts` | both ends, `start` and `end` | 2 | Cleavage footprints (ChEC, MNase, ATAC) |
| `cuts_plus` | the upstream end, in anchor orientation | 1 | Strand-split footprints |
| `cuts_minus` | the downstream end, in anchor orientation | 1 | Strand-split footprints |
| `cuts_proximal` | the end nearer the anchor centre | 1 | Near/far-split footprints; unstranded stores only |
| `cuts_distal` | the end farther from the anchor centre | 1 | Near/far-split footprints; unstranded stores only |
| `coverage` | every base the fragment covers (fraction of each bin) | — | Matrices and tracks only |
| `cuts_stranded` | the read 5′ end of a stranded store | 1 | ChIP-exo; same as `cuts` on such a store |

Cut positions are the fragment boundaries `start` and `end`: the gaps between bases where the enzyme cut. With unstranded paired-end data such as ChEC-seq, "plus" and "minus" mean the fragment's upstream and downstream ends relative to the anchor, not read strands.

Both kinds of fragment around a protected site put their ends at its edges, but they differ in which end lands where:

- **Fragments that contain the site**, such as the short TF-protected particle, have their upstream end (`cuts_plus`) before the site and their downstream end (`cuts_minus`) after it.
- **Fragments that abut the site** from either side do the opposite: those downstream start at its right edge (`cuts_plus`), and those upstream end at its left edge (`cuts_minus`).

The combined `cuts` profile shows a dip between two edge peaks either way. The strand split tells you which kind of fragment the signal comes from.

`cuts_proximal` and `cuts_distal` split the two ends by distance to the anchor centre instead of by orientation. For a fragment that contains the site, the proximal end is the cut at the nearer edge of the protected particle and the distal end is the other edge; for a fragment off to one side, the proximal end is the one facing the site. A fragment centred exactly on the anchor counts its upstream end as proximal, so `cuts_proximal + cuts_distal == cuts` always. Stranded stores (ChIP-exo) have one end per read, so these two features raise an error there.

### Filtering by motif overlap

`motif_overlap=` (on `vplot`, `to_matrix`, `cut_site_footprint` and `extract_events`) keeps fragments by how they sit on each anchor's motif interval, independently of `feature`:

| `motif_overlap` | Keeps fragments that |
| :--- | :--- |
| `None` (default) | all fragments |
| `"any"` | overlap the motif by at least 1 bp |
| `"full"` | contain the whole motif |
| `"none"` | do not overlap the motif |

The motif interval is the BED `start`/`end`, or the `start`/`end` columns of a DataFrame (kept even when a `center` column sets the window centre). Anchors given only as a centre use the 1-bp interval `[center - 0.5, center + 0.5)`. `"any"` and `"none"` partition the fragments, so their counts add up to the unfiltered result.

Restricting to overlapping fragments confines the signal to the region within half a fragment length of the motif, by construction. Normalise such V-plots with `"cpm"`, not `"expected"` (whose flank background is then depleted; pyvplot warns), and do not feed them to `footprint_score`, whose centre-versus-shoulder contrast becomes circular. `VPlot.diff` refuses to compare V-plots built with different `feature` or `motif_overlap`.

**Defaults differ**: `vplot` always defaults to `midpoint`, while `to_matrix` and the track exporters default to the assay's `default_feature` (`cuts` for ChEC and ATAC). In analysis code, pass `feature=` explicitly.

## 5. Fragment length: axis versus filter

Length plays two different roles.

**V-plot length axis.** `vplot(min_len=20, max_len=250, len_bin=5)` defines bin edges: half-open `[20, 250)`, in 5 bp rows. It takes numbers only, and `max_len - min_len` must be a multiple of `len_bin`.

**Length filter, everywhere else.** `len_range=(20, 120)` keeps fragments with `20 <= length <= 120`. Both bounds are inclusive. You can also pass a size-bin name of the store's assay, such as `len_range="tf"`. This works in `to_matrix`, `cut_site_footprint`, `footprint_score`, `extract_events`, `phasogram`, `VPlot.marginal_position` and all track exporters. `store.filter(min_len=, max_len=)` is inclusive too.

So `to_matrix(len_range=(20, 120))` selects the same fragments as V-plot rows from `min_len=20, max_len=121`.

## 6. Normalisation

Everything starts as raw counts. CPM always divides by the **total fragments in the store**, not those in the window or length range, so it corrects for sequencing depth, not for enrichment.

### V-plots: `VPlot.normalize(method)`

| `method` | Definition | Use it to |
| :--- | :--- | :--- |
| `cpm` | counts / (store fragments / 10⁶) / n_anchors | compare samples at the same anchors (the default before `diff`) |
| `cpm_total` | counts / (store fragments / 10⁶) | compare anchor sets of different size by total signal |
| `per_anchor` | counts / n_anchors | compare within one library |
| `density` | counts / grid total | compare shape only |
| `expected` | each length row / its mean in the outer 20% of positions | see the TF-sized V next to the much more abundant nucleosome band. 1 = flank level |
| `log2_oe` | log2((obs + pc) / (expected + pc)) | the same on a log scale |
| `row_zscore` | z-score each length row | shape per length class |

`expected` needs signal in the flanks. With few anchors or narrow windows, some length rows have no flank counts. Those rows are divided by the tiny pseudocount and swamp the colour scale, and pyvplot warns when it happens. Widen `flank`, coarsen `len_bin`, add anchors, or use `expected_from="genome"`. That option takes the row expectation from the store's genome-wide length distribution, scaled to this V-plot's total. It corrects for length composition but not for local accessibility.

### Differential V-plots: `v_a.diff(v_b)`

This computes `a` relative to `b`: pass the treatment first and the reference as the argument. If both V-plots are raw, both are first CPM-normalised. The default `mode="log2fc"` adds half a raw count per cell, expressed in the normalised units and shared by both samples (the geometric mean of the two samples' values). The log2FC therefore does not depend on sequencing depth, and a cell empty in both samples is exactly 0. Pass `pseudocount=` to set it yourself, in normalised units. `smooth_sigma=(rows, cols)` applies a Gaussian blur, in bins, before taking the ratio.

### Matrices: `SignalMatrix.normalize(method)`

| `method` | Definition |
| :--- | :--- |
| `cpm` | values / (store fragments / 10⁶). Rows are already per anchor, so `mat.normalize("cpm").mean_profile()` is in the same units as a CPM V-plot's `marginal_position()`. |
| `rpkm` | CPM / (bin size in kb) |
| `row_sum` | each anchor row sums to 1 (shape only) |
| `row_zscore`, `zscore` | per row, or over the whole matrix |

## Numba and speed

With `numba` installed, extraction runs in parallel compiled kernels; without it, an exact NumPy path gives identical results. On a 4-core machine, 50,000 anchors against 20 million fragments take about 0.3 s for a V-plot and 0.1 s for a matrix, once compiled. The NumPy path takes about 2 s for the same V-plot. The first call in a fresh environment compiles the kernels, which takes several seconds; the result is cached on disk for later sessions.
