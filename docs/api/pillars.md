# Cut pillars & de-novo motif footprints

De-novo scanning of multi-length cleavage pillars and protected motif footprints from paired-end cleavage geometry, with continuous 1-bp `chrom, start, end` tracks and BED interval tables ready for `pygenomic_tracks` and `moodspy`.

Index `p` of `left_cuts`, `right_cuts` and `pillar_score` is the cleavage boundary at genomic coordinate `L + p` (a fragment `[s, e)` contributes a left cut at `s - L` and a right cut at `e - L`). Candidate motifs are called by scoring pairs `(l, r)` of called cut pillars whose separation `r - l` falls in `motif_width_range` (inclusive), yielding exact protected intervals `[l, r)`. The continuous `protection_score` profile is exploratory and is not what motifs are called from. For a walk-through with calibration on a control and k-mer bias correction, see the [tutorial](../tutorials/pillar-scanning.md).

::: pyvplot.scan_locus_pillars

::: pyvplot.LocusPillarScan

::: pyvplot.scan_genome_pillars

::: pyvplot.pillar_expected_cuts

::: pyvplot.find_active_windows
