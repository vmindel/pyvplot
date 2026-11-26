# KLF5 ChEC-seq footprint benchmark

Scores the pillar scanner's 1-bp protection profile and its calls against model-derived
ground truth: FiNeMo hits on ISM attributions of the AlphaGenome KLF5FULL fine-tune, and
the ISM track. Both exist only inside the ISM tiles (summit +/- 200 bp of 64,404 tiles,
merged into 53,485 regions, 23.8 Mb), so every metric is restricted to them. Tuning is on
even chromosomes + chrX ("dev"), reporting on odd ones ("test"). Paths in `common.py` are
lab-specific; every script runs through `bsub` (`run_*.sh`), never on the login node.

| script | what it measures |
|---|---|
| `01_baseline.py` | oriented cut aggregate around pattern_0 hits; per-bp AUPRC, offset, Spearman vs ISM and pair-caller PR for the original scanner |
| `02_scorers.py` | 18 count-based binomial-LLR / ratio depletion scorers against the baseline |
| `03_checks.py` | same-9-mer cut depth at hits / tile no-hit / off-peak; GC-only scorer; site AUROC in coverage deciles; per-site core/flank ceiling |
| `04_bias.py` | k=6/8 `KmerBiasTable` from off-peak KLF5 cuts; observed/expected at the same 9-mers; bias-corrected scorers; AUROC within coverage x GC bins |
| `08_occupancy.py` | short-fragment midpoint / bracket occupancy scores vs a +-25 bp summit baseline; offsets; same-9-mer check; re-ranking profile peaks |
| `09_kmer_binding_vs_bias.py` | every 9-mer in strict off-peak DNA: depletion and midpoint enrichment vs KLF PWM score at fixed GC |
| `10_masked_bias.py` | k=6 tables fitted off-peak with KLF-PWM 9-mers masked (`from_cuts(exclude=)`) vs a GC-matched placebo mask, on chromosomes other than chr15-19; held-out 9-mer depletion; profile benchmark with each table as `expected_cuts`; shipped `occupancy_score` site AUROC |

Findings (test split):

- No systematic shift: profile peaks sit on hit centres (median offset -0.5 bp).
- Peaks of the profile beat the pillar-pair caller at every call density (~10 calls/kb:
  precision 0.29, recall 0.34; pairs at 7.7/kb: 0.22 / 0.20; random placement 0.08).
- Raw-count LLR scorers did not beat the log-pillar profile.
- GCCCCGCCC cores are cut at 0.013 of flank at hits, 0.014 in tiles without hits and 0.075
  off-peak; 10-bp GC alone gives site AUROC 0.85 vs 0.74-0.77 for the profile (partly
  circular: hits are GC boxes). A 6-mer table predicts only ~0.6-0.7.
- Off-peak is not unbound (`09_kmer_binding_vs_bias.py`): >= 2 kb from tiles and outside
  active windows, top-1% KLF-PWM 9-mers vs below-median ones of the same GC count are ~2x
  more depleted (GC=8: 0.24 vs 0.45), ~2x more midpoint-enriched (2.8x vs 1.7x) and carry
  ~1.6x more cuts. Depletion = GC-dependent cleavage preference + motif-specific binding.
- Occupancy (`08_occupancy.py`): 20-120 bp fragments bracketing the site, against local
  background, reach AUROC 0.74 within coverage x GC bins (profile 0.70, a +-25 bp summit
  baseline 0.71, GC 0.52) and separate hits from non-hits at identical 9-mers (midpoints
  ~4x vs ~2x local background) where depletion cannot. They are not bp-precise (24% of hits
  within +-3 bp vs 64% for the profile) and re-ranking profile peaks by them lowers
  precision, so the profile stays the positioner and caller. The shipped
  `LocusPillarScan.occupancy_score` reproduces it (0.735 vs 0.737).
- Masking motif matches in the bias fit (`10_masked_bias.py`): with the top-1% KLF-PWM
  9-mers (+-10 bp) excluded, the 6-mer table predicts less depletion at held-out KLF-like
  9-mers (GC=8: 0.77 vs 0.68 expected, observed 0.24), about twice its shift at other 9-mers
  of the same GC and five times a GC-matched placebo mask's; all-GC 6-mers gain 16% weight
  (placebo 7%). So the off-peak table had absorbed some binding, but little: no table comes
  close to the observed GC-rich depletion, even at below-median 9-mers (GC=8: 0.45 vs
  0.69-0.74). As `expected_cuts`, every k=6 table helps the profile slightly and equally
  (~10 calls/kb: precision 0.295 -> 0.308, recall 0.33 -> 0.35; per-bp AUPRC 0.202 -> 0.215;
  site AUROC 0.687 -> 0.694), masked or not. A top-5% mask removes 64% of the genome and
  inverts the GC trend of the weights; not usable.
- Signal beyond sequence: within coverage x GC bins the profile still reaches AUROC 0.70,
  and 9-mers with an A are 2-3x more depleted at hits than the same 9-mer without a hit.
