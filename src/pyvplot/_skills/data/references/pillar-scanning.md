# De-novo footprints: cut pillars and protected motifs

Use when motif positions are unknown and the user wants bound sites called from paired-end cleavage geometry. For scoring known motif sites use `footprint_score` (`references/matrices-footprints.md`).

**Model.** A bound protein makes fragment ends of many lengths stop at the same boundary on each side of it (**cut pillars**) and leaves the DNA between uncut. A 1-bp **protection profile** scores, at every boundary and footprint width, an uncut core against the strongest flank pillar within `flank_pad` of each edge. `caller="profile"` (default) calls the profile's peaks, each spanning its best width with edges snapped to the flank pillars; `caller="pairs"` scores only pairs of called pillars `(l, r)` with `r - l` in `motif_width_range` (exact pillar boundaries, lower sensitivity). Both report protected bases `[l, r)`. Paired-end stores only (stranded/ChIP-exo stores raise).

**Coordinates.** Index `p` of every 1-bp array is the cut boundary at genomic `L + p` (between bases `L + p - 1` and `L + p`). A fragment `[s, e)` adds a left cut at `s`, a right cut at `e`. A call's `start`/`end` are its flank boundaries; `center` is the profile peak (profile caller) or `(l + r) // 2` (pairs).

## Workflow

```python
import pyvplot as vp

store = vp.FragmentStore.load("chec.frags.npz")

# 1. One locus: profiles + calls; re-threshold without re-scanning
scan = store.scan_locus_pillars("chr1:10000-10600")          # or vp.scan_locus_pillars(store, ...)
scan.motifs[["chrom", "start", "end", "width", "protection_score", "composite_score"]]
scan.pillars[["pos", "pillar_score", "n_len_bins", "left_frac", "strand"]]
strict = scan.call(min_protection_score=1.5)                 # re-threshold; scan.call(caller="pairs") switches caller
fig, axes = scan.plot(store=store)                           # 4 panels: fragment ends, profile + calls vs threshold, pillars, coverage
tracks = scan.to_pygenomic_tracks()                          # dict of chrom/start/end DataFrames

# 2. Sequence bias (MNase AT preference): k-mer table from a naked-DNA / free-MNase control
table = vp.KmerBiasTable.from_cuts(control_store, "genome.fa", k=6)
expected = vp.pillar_expected_cuts(table, "genome.fa")       # f(chrom, start, end) -> per-boundary rate

# 3. Calibrate min_protection_score on the control (downsampled to the sample's depth first)
loose = dict(min_protection_score=0.0, min_raw_protection=0.0, expected_cuts=expected)
s = store.scan_genome_pillars(**loose)
c = control_store.scan_genome_pillars(**loose)
for thr in (0.5, 0.75, 1.0, 1.5, 2.0):
    print(thr, (s.protection_score >= thr).sum(), (c.protection_score >= thr).sum())  # control/sample ~ FDR

# 4. Genome-wide calls at the chosen threshold, sorted by composite_score
motifs, pillars = store.scan_genome_pillars(min_protection_score=1.0, expected_cuts=expected, return_pillars=True)
```

## Outputs

`motifs`: `start`, `end`, `width` (`r - l`), `center`, `raw_log2_protection` (`log2((flank + pseudocount) / (core + pseudocount))`), `protection_score` (raw times a 0–1 short-fragment coverage weight), `left_pillar_score` / `right_pillar_score` / `core_pillar_score`, `tf_coverage`, `composite_score` (protection times log flank height times log coverage), `occupancy_score` (`scan.occupancy_score` at `center`; not used for calling).

`scan.occupancy_score` (1 bp, same indexing as `protection_score`): 20–120 bp fragments (`tf_len_range`) with start in `[c - 60, c - 5]` and end in `[c + 5, c + 60]`, as a signed Poisson LLR against their mean at `c ± [40, 150]` (grows with depth). `vp.occupancy_profile(store, region, inner=5, outer=60, background=(40, 150), score="llr" | "log2" | "counts")` gives it without a scan; use `"log2"` to compare loci of different depth. Exported by `to_track_df("occupancy_score")`, drawn by `scan.plot(ax=ax, what="occupancy")`.

`pillars`: `pos` (boundary, placed on the centroid of its raw cut stack), `start`/`end` (`pos` ± `pillar_window // 2`), `pillar_score` (sum over 10-bp length bins of `log2(1 + cuts within pillar_window)`), `n_len_bins`, `left_cuts`, `right_cuts`, `left_frac`, `strand` (`+` when fragment starts dominate).

## Thresholds, in the order they act

`min_pillar_score` (8.0), `min_len_bins` (4), `min_pillar_prominence` (2.0) define pillars; `footprint_widths` (8–35) and `flank_pad` (2) shape the profile; `motif_width_range` ((10, 24), inclusive) limits call widths (pairs: exact; profile: the best width, then edge snapping can change it by up to `2 * flank_pad`); `min_raw_protection` (0.20) and `min_protection_score` (1.0 profile / 0.75 pairs) filter calls; `min_motif_distance` (12) suppresses close calls, keeping the best.

## Footguns

- **Thresholds depend on depth.** `pillar_score` sums log counts, so defaults call more at high depth and fewer at low depth. Calibrate on a control per dataset; do not reuse thresholds across libraries.
- **Depletion is not occupancy.** Part of a GC box's depletion is MNase preference: 10-bp GC alone ranked FiNeMo hit centres better than the profile on KLF5 ChEC-seq (AUROC 0.85 vs 0.77; partly circular, as hits are GC boxes). Compare footprints against a sequence-only baseline (GC, k-mer expected cuts) or a free-MNase control before calling them binding.
- **"Off-peak" is not "unbound".** In KLF5 ChEC-seq DNA ≥2 kb from peak tiles and outside active windows, KLF-like 9-mers were ~2× more depleted and had ~2× more centred short-fragment midpoints than other 9-mers of the same GC: weak binding. A `KmerBiasTable.from_cuts` fitted on such cuts absorbs it; mask the factor's motif matches (`from_cuts(..., exclude=...)`), or use a free-MNase / naked-DNA store.
- **Short-fragment midpoints are the motif-specific readout.** At identical 9-mers, centred 20–120 bp fragment midpoints were ~4× local background at hits vs ~2× at non-hits, where depletion did not differ (0.013 vs 0.014). `occupancy_score` (bracketing fragments) ranked KLF hit centres best within coverage × GC bins (AUROC 0.74 vs 0.69 for protection), but only ~25% of its peaks are within 3 bp of the site (protection ~64%): rank/weight with occupancy, place with protection. Reranking profile calls by occupancy lowered precision at matched call density.
- **Use `scan.motifs`, not a threshold on the array.** With the default caller calls are peaks of `scan.protection_score` with NMS (`min_motif_distance`) and both thresholds; thresholding the array gives runs of positions, not sites. With `caller="pairs"` the array is context only.
- **`expected_cuts` shapes.** `scan_locus_pillars` takes an array of length `end - start` or a function `(chrom, start, end) -> array`; `scan_genome_pillars` takes only the function (a genome-length array would not fit in memory). Any scale; rescaled to mean 1 per window. A wrong length raises. It corrects the profile and the call scores; pillars are still called on the uncorrected score. `scan.call()` reuses the scan's array; `scan.call(expected_cuts=...)` recomputes the profile.
- **Packed footprints.** The stretch between two neighbouring sites is bounded by pillars on both sides; if it is also uncut it is called as its own motif. Check widths and motif content.
- **One-sided sites (pairs caller).** With `caller="pairs"` both flanks must be called pillars, even with `flank_mode="either"`; the profile caller needs only flank scores.
- **Profile edges are approximate** (within ~1 bp on clean simulations); use `caller="pairs"` when exact boundaries matter.
- **`min_pillar_distance` is fixed at 5 bp** in both scans; only `scan.call(min_pillar_distance=...)` changes it.
- **Validate on real data** before trusting calls: enrichment of the expected motif in calls versus matched flanks, and a few known sites by eye with `scan.plot()`.
