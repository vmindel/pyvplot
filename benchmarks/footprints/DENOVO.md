# De novo motifs from KLF5 ChEC-seq footprints

The question: which motifs do footprint calls from the KLF5 fragments carry? Discovery starts
from fragments only; ISM, the model and FiNeMo are used only to compare the results.

## Inputs

- **Calls.** `scan_genome_pillars` on all 63.9 M KLF5FULL fragments, with the profile caller
  and the Fig2 settings: `len_range=(20, 300)`, `flank_mode="either"`,
  `footprint_widths=range(6, 21)`, `motif_width_range=(6, 20)`. This gives 1.07 M calls,
  median width 11 bp (`05_shipped.py`, ~17 min on 1 CPU).
- **Discovery set.** The top 20 k calls by `composite_score`; 96% of them fall inside the
  ISM tiles, so these are mostly the strongest peaks. Each call gives the 30-bp window at its
  profile peak.

## STREME-style discovery (pystreme, `06_denovo.py`)

Job 852949, A40, ~8 min for 3 controls. Motifs that are hold-out significant:

| motif | best HOCOMOCO v12 match | model MoDISco (PCC) | sites / 20 k | controls where found |
|---|---|---|---|---|
| `GCCCCACCC` | KLF3 / KLF5 | pos_0 (0.98) | 6,550–6,900, central | shuffle, GC-matched |
| `GCCACACCC`, `CACACCC` | KLF5 / KLF10 | pos_19 (0.99) | ~4,800 | local flanks |
| `GTGACTCA` | JUNB / FOS (AP-1) | pos_1 (0.97–0.99) | ~1,050 | all three |
| `ACATTCC(A/T)` | TEAD1 / TEAD4 | pos_4 / pos_9 (0.96–0.997) | 170–600 | shuffle, GC-matched |
| `TAAACA` | FOX | pos_2 (0.92) | ~520 | shuffle |
| `CAAAGGGC` | PPARG-like | pos_3 (0.77) | ~670 | GC-matched |
| `CCGCCC` | SP3 | pos_0 (0.83) | ~1,360 | GC-matched |
| `CTAG` (`CCTAGG`, `CTAGGG`, `CCCTAG`) | weak zinc-finger matches | pos_11 (0.87–0.89) | ~13,000 | all three |

**Reading.**

- KLF tops the list against GC-matched windows, so it is not just GC content.
- The AT-containing cofactors (AP-1, TEAD, FOX) are recovered as well.
- `CTAG` is MNase cleavage preference: `CCTAGG` is the most-cut 6-mer in the KLF5 bias table,
  and the user confirms it. The model's MoDISco pos_11 matches it, so the sequence model
  learned the enzyme bias too. Hits of that pattern are not TF sites.

## Occupancy separates binding from bias (`11_denovo_occupancy.py`, `12_occ_stratified.py`)

- **Score.** Occupancy at each call centre is `occupancy_profile(score="log2")`: short
  fragments bracketing the site against local background (feat/occupancy-score).
- **Calls.** The top 60 k calls.
- **Depth control.** Calls are split into occupancy tertiles inside each TF-coverage
  quintile, then the tertiles are pooled, so coverage is matched.

Job 854584 (`12`). Pooled tertiles:

| matched coverage (median 330–332) | low | mid | high |
|---|---|---|---|
| median protection | 2.98 | 3.09 | 3.15 |
| GC of the 30-bp window | 0.536 | 0.542 | 0.561 |
| within 4 bp of a FiNeMo hit | 16% | 28% | 40% |
| KLF `GGGYGKGG` | 4.5% | 8.0% | 16.0% |
| SP `GGGGCGGG` | 1.2% | 1.5% | 2.5% |
| AP-1 `TGASTCA` | 3.0% | 4.7% | 4.5% |
| MNase `CTAG` | 21.8% | 18.2% | 17.1% |

The pattern holds in every coverage quintile: KLF 3.6% → 14.7% in the lowest, 6.3% → 18.4% in
the highest, and CTAG falls in each.

De novo discovery on the unstratified tertiles (job 854374, H100):

- **High tertile:** the top motif is KLF5 `GCCACACCC` (~9.8 k sites, hold-out log p −450),
  then AP-1, TEAD and SP.
- **Low tertile:** the top motif is `CCCTAG` (13 k sites), with no KLF5 motif, only SP-like
  `GGGCGGGGC`.

So two calls can have equal protection and GC, yet one is a KLF-bound footprint and the other
an enzyme-preference footprint. Occupancy tells them apart. Use protection to place a site
and `occupancy_score` to rank or filter calls.

A 6-mer bias table does not explain the split away (`10_masked_bias.py`, bias-ms):

- Every table, masked or not, predicts a GC-rich core/flank cut ratio of about 0.7, against
  an observed 0.24–0.45.
- Bias correction moves call precision only from 0.295 to 0.308.

## MoDISco-style clustering (`07_fp_modisco.py`)

- **Method.** modiscolite on 10 k windows of 200 bp. Contributions are one-hot × the signed
  1-bp protection profile, positive on footprints and negative on pillars; the clipped
  profile crashed modiscolite, which needs both signs.
- **Cost.** Job 853358 took 2.7 h on 1 CPU, against minutes for the STREME route, and adds
  little beyond it.
- **Patterns.** 35, ranked by number of seqlets:

| rank | consensus | seqlets | what it is |
|---|---|---|---|
| 0 | `GCCACWCCC` | 1,395 | KLF |
| 1 | `GGGGTGGGG` | 728 | KLF |
| 2 | `GCCCCGCCC` | 725 | SP / KLF |
| 3 | `GRTGASTCA` | 561 | AP-1 |

- G-tracts and many `YTARG`- / `CTAG`-containing patterns follow (ranks 10, 12, 19, 22, 23,
  25, 27, 31, 33): MNase cut preference at footprint edges.
- PCC with the model's patterns is 0.6–0.9; it is diluted by the 26-bp windows with flat flanks.

## Reproduce

```bash
bsub < run_05.sh   # genome-wide calls (CPU, ~25 min incl. benchmark)
bsub < run_06.sh   # pystreme x3 controls + model comparison (GPU, ~8 min)
bsub < run_11.sh   # occupancy tertiles + pystreme (GPU, ~10 min; needs occupancy_profile)
bsub < run_12.sh   # coverage-matched tertiles (CPU, ~3 min)
bsub < run_07.sh   # optional: MoDISco-style (CPU, ~3 h)
```

The non-MoDISco route takes about 25–30 min end to end.
