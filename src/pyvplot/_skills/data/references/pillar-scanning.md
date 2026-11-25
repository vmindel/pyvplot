# De-novo footprints: cut pillars and protected motifs

Use when motif positions are unknown and the user wants bound sites called from paired-end cleavage geometry. For scoring known motif sites use `footprint_score` (`references/matrices-footprints.md`).

**Model.** A bound protein makes fragment ends of many lengths stop at the same boundary on each side of it (**cut pillars**) and leaves the DNA between uncut. Pillars are called first; motifs are scored only between pairs of called pillars `(l, r)` with `r - l` in `motif_width_range`, and reported as the protected bases `[l, r)`. Paired-end stores only (stranded/ChIP-exo stores raise).

**Coordinates.** Index `p` of every 1-bp array is the cut boundary at genomic `L + p` (between bases `L + p - 1` and `L + p`). A fragment `[s, e)` adds a left cut at `s`, a right cut at `e`. Motif `start` is the left pillar, `end` the right pillar.

## Workflow

```python
import pyvplot as vp

store = vp.FragmentStore.load("chec.frags.npz")

# 1. One locus: profiles + calls; re-threshold without re-scanning
scan = store.scan_locus_pillars("chr1:10000-10600")          # or vp.scan_locus_pillars(store, ...)
scan.motifs[["chrom", "start", "end", "width", "protection_score", "composite_score"]]
scan.pillars[["pos", "pillar_score", "n_len_bins", "left_frac", "strand"]]
strict = scan.call(min_protection_score=1.0)
fig, axes = scan.plot()                                      # QC figure; scan.plot(ax=ax, what="protection") for one track
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

`motifs`: `start`, `end`, `width` (`r - l`), `center`, `raw_log2_protection` (`log2((flank + pseudocount) / (core + pseudocount))` over the pair), `protection_score` (raw times a 0–1 short-fragment coverage weight), `left_pillar_score` / `right_pillar_score` / `core_pillar_score`, `tf_coverage`, `composite_score` (protection times log flank height times log coverage).

`pillars`: `pos` (boundary, placed on the centroid of its raw cut stack), `start`/`end` (`pos` ± `pillar_window // 2`), `pillar_score` (sum over 10-bp length bins of `log2(1 + cuts within pillar_window)`), `n_len_bins`, `left_cuts`, `right_cuts`, `left_frac`, `strand` (`+` when fragment starts dominate).

## Thresholds, in the order they act

`min_pillar_score` (8.0), `min_len_bins` (4), `min_pillar_prominence` (2.0) define pillars; `motif_width_range` ((10, 24), inclusive, exact) selects pairs; `min_raw_protection` (0.20) and `min_protection_score` (0.75) filter pairs; `min_motif_distance` (12) suppresses overlapping or close calls, keeping the best.

## Footguns

- **Thresholds depend on depth.** `pillar_score` sums log counts, so defaults call more at high depth and fewer at low depth. Calibrate on a control per dataset; do not reuse thresholds across libraries.
- **`protection_score` the array is not the caller.** `scan.protection_score` / `protection_2d` / `best_widths` are an exploratory sliding-window profile (shaped by `footprint_widths`, `flank_pad`). Do not threshold it to get motifs; use `scan.motifs` / `scan.call(...)`. The motif column `protection_score` is the pair score and is what calls use.
- **`expected_cuts` shapes.** `scan_locus_pillars` takes an array of length `end - start` or a function `(chrom, start, end) -> array`; `scan_genome_pillars` takes only the function (a genome-length array would not fit in memory). Any scale; rescaled to mean 1 per window. A wrong length raises. It corrects pair scoring only: pillars are still called on the uncorrected score. `scan.call()` reuses the scan's array.
- **Packed footprints.** The stretch between two neighbouring sites is bounded by pillars on both sides; if it is also uncut it is called as its own motif. Check widths and motif content.
- **One-sided sites are not called.** Both flanks must be called pillars, even with `flank_mode="either"`.
- **`min_pillar_distance` is fixed at 5 bp** in both scans; only `scan.call(min_pillar_distance=...)` changes it.
- **Validate on real data** before trusting calls: enrichment of the expected motif in calls versus matched flanks, and a few known sites by eye with `scan.plot()`.
