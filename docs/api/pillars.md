# Cut pillars & de-novo motif footprints

De-novo scanning of multi-length cleavage pillars and protected motif footprints from paired-end cleavage geometry, with continuous 1-bp `chrom, start, end` tracks and BED interval tables ready for `pygenomic_tracks` and `moodspy`.

Index `p` of `left_cuts`, `right_cuts`, `pillar_score`, `protection_score` and `occupancy_score` is the cleavage boundary at genomic coordinate `L + p` (a fragment `[s, e)` contributes a left cut at `s - L` and a right cut at `e - L`). With the default `caller="profile"`, footprints are the peaks of the 1-bp `protection_score` profile, each spanning its best width; with `caller="pairs"`, pairs `(l, r)` of called cut pillars whose separation `r - l` falls in `motif_width_range` (inclusive) are scored, yielding intervals `[l, r)`. `occupancy_score` (from [`occupancy_profile`](#pyvplot.occupancy_profile)) counts short fragments bracketing each position, a second signal that does not need an uncut core. For a walk-through with calibration on a control and k-mer bias correction, see the [tutorial](../tutorials/pillar-scanning.md).

::: pyvplot.scan_locus_pillars

::: pyvplot.LocusPillarScan

::: pyvplot.scan_genome_pillars

::: pyvplot.occupancy_profile

::: pyvplot.pillar_expected_cuts

::: pyvplot.find_active_windows
