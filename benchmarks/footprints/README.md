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

Findings (test split):

- No systematic shift: profile peaks sit on hit centres (median offset -0.5 bp).
- Peaks of the profile beat the pillar-pair caller at every call density (~10 calls/kb:
  precision 0.29, recall 0.34; pairs at 7.7/kb: 0.22 / 0.20; random placement 0.08).
- Raw-count LLR scorers did not beat the log-pillar profile.
- The cut-free core is mostly MNase sequence preference: GCCCCGCCC cores are cut at 0.013
  of flank at hits, 0.014 in tiles without hits and 0.075 off-peak. 10-bp GC alone gives
  site AUROC 0.85 vs 0.74-0.77 for the profile. A 6-mer table predicts only ~0.6-0.7.
- Signal beyond sequence: within coverage x GC bins the profile still reaches AUROC 0.70,
  and 9-mers with an A are 2-3x more depleted at hits than the same 9-mer without a hit.
