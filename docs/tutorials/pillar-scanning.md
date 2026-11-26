# De-novo footprints from cut pillars

When you do not know where a factor binds, the fragments can still show it. Around a bound protein, fragment ends of many different lengths stop at the same two boundaries, one either side of it, and the DNA between them is not cut. pyvplot calls those boundaries **cut pillars**, scores every position for an uncut core between strong flanks (the 1-bp **protection profile**), and calls **protected footprints** at the peaks of that profile. The scanner needs paired-end fragments; stranded stores (ChIP-exo) raise an error.

## Coordinates

Every 1-bp array of a scan is indexed by cut boundary: index `p` is the boundary at genomic `L + p`, between bases `L + p - 1` and `L + p`. A fragment `[s, e)` adds a left cut at `s` and a right cut at `e`. A footprint `[l, r)` covers the protected bases between its flank boundaries `l` and `r`.

## One locus

The toy store has uniform background fragments over chr1:4000-6000 and two bound sites, `[5000, 5016)` and `[5300, 5312)`: fragments end at the left boundary, start at the right one, and short fragments span each site.

```python
import numpy as np
import pandas as pd
import pyvplot as vp

rng = np.random.default_rng(3)
starts, ends = [], []
for _ in range(3000):  # background
    s = int(rng.integers(4000, 6000))
    starts.append(s)
    ends.append(s + int(rng.integers(20, 250)))
for l, r in [(5000, 5016), (5300, 5312)]:  # bound sites
    for _ in range(80):
        starts.append(l - int(rng.integers(25, 220)))
        ends.append(l)
        starts.append(r)
        ends.append(r + int(rng.integers(25, 220)))
    for _ in range(40):
        starts.append(l)
        ends.append(r + int(rng.integers(20, 80)))
order = np.argsort(starts)
frags = vp.FragmentStore.from_arrays(
    {"chr1": np.asarray(starts)[order].astype(np.int32)},
    {"chr1": np.asarray(ends)[order].astype(np.int32)},
    assay="chec",
)

scan = frags.scan_locus_pillars("chr1:4800-5500")
print(scan)
print(scan.motifs[["start", "end", "width", "protection_score", "raw_log2_protection", "composite_score"]].round(2).to_string(index=False))
```

```text
LocusPillarScan(chr1:4800-5500, n_pillars=85, n_motifs=2)
 start  end  width  protection_score  raw_log2_protection  composite_score
  5000 5015     15              1.83                 2.26            69.51
  5300 5313     13              2.11                 2.70            76.11
```

`caller="pairs"` is the original caller: it scores only pairs of called pillars `r - l` apart with `r - l` in `motif_width_range`, so on clean data like this it returns the pillar boundaries exactly. On KLF5 ChEC-seq, scored against FiNeMo hits from a sequence model, profile peaks had higher precision and recall at every call density (see *Real data* below), which is why `"profile"` is the default.

```python
pairs = frags.scan_locus_pillars("chr1:4800-5500", caller="pairs")
print(pairs.motifs[["start", "end", "width", "protection_score"]].round(2).to_string(index=False))
```

```text
 start  end  width  protection_score
  5000 5016     16              1.81
  5300 5312     12              1.99
```

Both sites are found, centred on the planted ones; each edge lands on the tallest pillar within `flank_pad` (2 bp) of the profile's best width, here within 1 bp of the planted boundary. Most of the 85 pillars are background stacks with no depleted core next to them; the two flanks of the first site stand out by `pillar_score` and by how many length bins they span:

```python
near = scan.pillars[(scan.pillars["pos"] - 5150).abs() < 200]
print(near[["pos", "pillar_score", "n_len_bins", "left_frac", "strand"]].round(2).head(8).to_string(index=False))
```

```text
 pos  pillar_score  n_len_bins  left_frac strand
4957         13.34          11       0.33      -
4964         16.00          15       0.47      -
4973         15.17          13       0.53      +
4991         16.75          15       0.44      -
5000         56.08          22       0.36      -
5005         11.17           9       0.69      +
5016         44.80          19       0.94      +
5030          9.75           8       0.27      -
```

## Looking at a scan

`scan.plot(store=frags)` draws the evidence for every call in four aligned panels:

```python
fig, axes = scan.plot(store=frags)
```

![Pillar scan of chr1:4800-5500: fragment ends by length, the protection profile with the two calls against the threshold, pillar score, and short-fragment coverage](../img/pillar_scan_locus.png)

- **Fragment ends by length.** Each fragment contributes its start (red) and end (dark blue). A pillar is a vertical stack: ends of every length stopping at one boundary. Diagonals are fragments sharing their other end.
- **Protection.** The grey 1-bp profile, and each call as a dark bar spanning `[start, end)` at its `protection_score`, above the red `min_protection_score` line. With the default caller the bars sit on the profile's peaks; the profile dips below zero on the pillars themselves.
- **Pillar score.** Red dots are called pillars. Only a depleted core between tall flanks scores high, which is why the 81 background pillars produce nothing.
- **TF-size coverage.** Short (20–120 bp) fragments covering each position, which weights the protection score.

Without `store=`, the top panel shows the length-resolved cut matrix instead. `scan.plot(ax=ax)` draws the protection track alone into an existing axis, for multi-track figures; `scan.plot(mode="multiscale")` adds the width-resolved protection profile.

## What the tables hold

`scan.motifs`, one row per call (the same columns for both callers):

| column | meaning |
|---|---|
| `start`, `end`, `width` | protected interval `[l, r)`; `width = r - l`. Profile caller: the peak's best width, each edge moved to the tallest pillar within `flank_pad`. Pair caller: the two pillars |
| `center` | profile caller: the profile peak; pair caller: `(l + r) // 2` |
| `raw_log2_protection` | `log2((flank + pseudocount) / (core + pseudocount))`: flank combines the two flank pillar scores (`flank_mode`), core is the mean of the average and the maximum `pillar_score` inside |
| `protection_score` | `raw_log2_protection` times a 0–1 weight for short-fragment (TF-size) coverage at `center` |
| `left_pillar_score`, `right_pillar_score`, `core_pillar_score` | `pillar_score` at `l`, `r` and `center` (divided by the expected rate with `expected_cuts`) |
| `tf_coverage` | short fragments covering `center` |
| `composite_score` | `protection_score * log2(1 + mean flank) * log2(1 + tf_coverage)`; the genome scan sorts by it |

`scan.pillars`, one row per called pillar:

| column | meaning |
|---|---|
| `pos` | the boundary, placed on the centroid of its raw cut stack |
| `start`, `end` | `pos` plus or minus `pillar_window // 2` |
| `pillar_score` | sum over 10-bp length bins of `log2(1 + cuts within pillar_window)` |
| `n_len_bins` | length bins with at least one cut there |
| `left_cuts`, `right_cuts`, `left_frac`, `strand` | fragment starts and ends at the pillar; `strand` is `+` when starts dominate, as on the right flank of a bound site |

`scan.protection_score` is the 1-bp profile: at each boundary, the best over `footprint_widths` (restricted to `motif_width_range`) of the coverage-weighted depletion score; `protection_2d` holds every width and `best_widths` the winner. The default caller calls its peaks; with `caller="pairs"` it is context only.

## Thresholds and re-calling

`call()` re-calls pillars and footprints from the stored arrays without touching the fragments, and `call(caller="pairs")` switches caller:

```python
for thr in (1.0, 1.9, 2.5):
    print(thr, scan.call(min_protection_score=thr).motifs[["start", "end"]].values.tolist())
```

```text
1.0 [[5000, 5015], [5300, 5313]]
1.9 [[5300, 5313]]
2.5 []
```

The thresholds, in the order they act:

- `min_pillar_score` (8.0), `min_len_bins` (4), `min_pillar_prominence` (2.0): what counts as a pillar.
- `footprint_widths` (8–35) and `flank_pad` (2): the widths the profile tries and how far from each edge it looks for the flank pillar.
- `motif_width_range` ((10, 24), inclusive): pairs: which pillar pairs are scored at all, so call widths are exact; profile: which widths compete for the best, after which snapping each edge can change a call's width by up to `2 * flank_pad`.
- `min_raw_protection` (0.20): minimum `raw_log2_protection`.
- `min_protection_score` (1.0 for `caller="profile"`, 0.75 for `"pairs"`): minimum after coverage weighting.
- `min_motif_distance` (12): calls closer than this keep only the best (pairs also drop overlapping calls).

`pillar_score` is a sum of log counts, so the pillar thresholds mean less at higher depth and more at lower depth. Calibrate them on each dataset (below) rather than reusing numbers between libraries.

`scan.to_track_df("protection_score")`, `scan.to_pygenomic_tracks()` and `scan.plot()` export and draw the profiles; `to_track_df("motifs")` and `to_track_df("pillars")` return the tables.

## Genome-wide

`scan_genome_pillars()` finds active windows (50-bp bins with at least `min_bin_frags` short-fragment midpoints, padded and merged) and scans each with the same profile and calling as `scan_locus_pillars` (`caller`, `footprint_widths` and `flank_pad` included). It returns motifs sorted by `composite_score`, and pillars too with `return_pillars=True`:

```python
motifs, pillars = frags.scan_genome_pillars(return_pillars=True)
print(motifs[["chrom", "start", "end", "width", "composite_score"]].round(1).to_string(index=False))
```

```text
chrom  start  end  width  composite_score
 chr1   5300 5313     13             76.1
 chr1   5000 5015     15             69.5
```

## Sequence bias

MNase prefers AT-rich DNA, so a GC-rich stretch between AT-rich flanks has few cuts with nothing bound. Estimate a k-mer table on a naked-DNA or free-MNase control and pass the expected cut rate:

```python
genome = {"chr1": "".join(rng.choice(list("ACGT"), 10_000))}
# With real data: table = vp.KmerBiasTable.from_cuts(control_store, "genome.fa", k=6)
table = vp.KmerBiasTable(k=4, weights=rng.uniform(0.5, 2.0, 4**4))
expected = vp.pillar_expected_cuts(table, genome)

scan_b = frags.scan_locus_pillars("chr1:4800-5500", expected_cuts=expected)
motifs_b = frags.scan_genome_pillars(expected_cuts=expected)
print(expected("chr1", 5000, 5004).round(2), len(scan_b.motifs), len(motifs_b))
```

```text
[1.18 2.47 2.46 2.44] 5 5
```

`expected(chrom, start, end)` gives, at each boundary, the propensity of a fragment starting there plus that of one ending there. The scanner rescales it to mean 1 per window and divides `pillar_score` by it before computing the profile and scoring calls; pillars themselves are still called on the uncorrected score. `scan.call()` reuses the scan's correction, and `scan.call(expected_cuts=...)` recomputes the profile with a new one. Here the table is random, so the correction mostly adds noise: three weak extra calls appear. On a simulated naked-DNA store with strong 4-mer bias and no bound sites, the correction cut false calls of the pair caller from 58 to 9.

A short k-mer table does not capture all of the enzyme's preference. On KLF5 ChEC-seq, a 6-mer table fitted on the library's own cuts away from peaks predicted cores at the commonest KLF 9-mers cut at about 0.6–0.7 of their flanks; the same 9-mers far from any peak were cut at 0.07–0.12. MNase barely cuts long GC runs, bound or not.

## Calibrating on a control

A free-MNase or naked-DNA control has no footprints, so its calls estimate false positives. Downsample the control to the sample's fragment count first (for example `samtools view -s` on the BAM), scan both permissively, and take the lowest threshold at which the control makes few calls relative to the sample:

```python
rng_c = np.random.default_rng(12)  # control: background only, at the sample's background depth
c_starts = np.sort(rng_c.integers(4000, 6000, 3000))
control = vp.FragmentStore.from_arrays(
    {"chr1": c_starts.astype(np.int32)},
    {"chr1": (c_starts + rng_c.integers(20, 250, 3000)).astype(np.int32)},
    assay="chec",
)
loose = dict(min_protection_score=0.0, min_raw_protection=0.0)
sample_calls = frags.scan_genome_pillars(**loose, min_bin_frags=5)
control_calls = control.scan_genome_pillars(**loose, min_bin_frags=5)
for thr in (0.25, 0.5, 0.75, 1.0, 1.5):
    n_s = int((sample_calls["protection_score"] >= thr).sum())
    n_c = int((control_calls["protection_score"] >= thr).sum())
    print(f"{thr:.2f}  sample {n_s:3d}  control {n_c:3d}")
```

```text
0.25  sample  87  control  91
0.50  sample  13  control  14
0.75  sample   5  control   1
1.00  sample   2  control   0
1.50  sample   2  control   0
```

![Motif calls at or above each min_protection_score for the sample and a background-only control](../img/pillar_calibration.png)

Below 0.75 the control calls about as much as the sample, so those calls are noise; at 0.75 the control still makes one; from 1.0, the default, only the two real sites remain. The control's count at a threshold divided by the sample's estimates the false discovery rate there. Use the same `expected_cuts` and window settings for both scans.

## Limits

- Closely packed footprints: the stretch between two neighbouring sites is bounded by pillars on both sides. If it is also uncut it is called as a footprint of its own. Check widths and motif content before treating every call as a binding site.
- With `caller="pairs"` a bound site with cuts on one side only is not called; `flank_mode="either"` relaxes how the two flanks combine but still needs a called pillar on each side. The profile caller only needs the flank scores, not called pillars.
- Profile-caller edges are approximate (here within 1 bp); use `caller="pairs"` when exact pillar boundaries matter more than sensitivity.
- `min_pillar_distance` is 5 bp in both scans; only `call()` changes it.

## Real data

On KLF5 ChEC-seq (64 M fragments), scored inside 64k ISM tiles against FiNeMo hits called on a sequence model's attributions (`benchmarks/footprints` in the repository):

- Profile peaks sat on hit centres (median offset −0.5 bp) and beat the pair caller at every call density: at about 10 calls/kb, 29% precision and 34% recall against 22% and 20% for pairs at 7.7/kb (random placement: 8%).
- Much of the signal is sequence. The GC fraction of 10 bp alone separated hit centres from other positions better than the profile (AUROC 0.85 against 0.74–0.77). Within bins matched for both coverage and GC the profile still reached 0.70, so it carries information beyond composition.

Read a footprint as occupancy only after comparing it with a sequence-only baseline (GC or k-mer expected cuts) or a free-MNase control.
