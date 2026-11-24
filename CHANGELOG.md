# Changelog

## Unreleased

### Fixed (bias sequence extraction)
- `extract_anchor_sequences(..., orient=False)` cut `-` anchors with the `+` anchor window, so after `correct_matrix(strands=...)` their k-mers were shifted: by 1 bp at integer centres with even `k` or half-integer centres with odd `k`, and by `downstream - upstream` bp with asymmetric windows. `-` anchors now keep their own window and are only left un-reverse-complemented. The default `orient=True` and `correct_matrix(anchors=)` were not affected.
- `extract_anchor_sequences` takes `stranded: bool = False` (passed automatically by `correct_matrix` / `correct_footprint` for stranded single-end stores such as `chip_exo`), shifting `-` anchor windows by +1 bp because 5'-end reads are indexed by their base `[s, s + 1)` rather than a right fragment boundary.
- `locus_vplot(..., orient=False)` on a `-` strand motif keeps the motif's 5'->3' biological end selection (`upstream`/`downstream`/`proximal`/`distal`) while leaving the relative `x` axis and `xlim` unflipped.

### Added (bias correction helpers)
- `extract_anchor_sequences` extracts 5'->3' anchor-oriented DNA sequences (with edge-exact k-mer padding and `-` strand reverse-complementation) from a genome FASTA or `{chrom: seq}` dict. `KmerBiasTable.correct_matrix` accepts a genome FASTA/dict directly when `anchors=` is provided (it must be the table the matrix was built from, in the same order, or it raises), and `KmerBiasTable.correct_footprint` bias-corrects a `CutSiteFootprint` (strand-split or combined).

### Changed (locus end labels)
- With a motif, `locus_vplot(feature="cuts_plus")` / `"cuts_minus"` label their points `upstream` / `downstream` (in the motif's 5'->3' orientation) instead of `left` / `right`; without a motif they stay `left` / `right`. Use the new names as `palette=` keys for motif plots.

### Fixed (single-locus, footprint score, and BED anchors)
- `locus_vplot` with a `-` strand motif and `coords="genomic"` now preserves the motif's 5'->3' biological orientation for `cuts_plus`/`cuts_minus` and proximal/distal tie-breaking.
- `LocusVPlot.to_vplot` uses strict half-open length and position bins `[min_len, max_len)` matching `store.vplot` (instead of `np.histogram2d`'s closed upper bin) and populates `genome_len_counts` so `normalize("expected", expected_from="genome")` works. The store's fragment lengths are kept as a 1-bp histogram, not copied into every `LocusVPlot`.
- `footprint_score` scales `pseudocount` by `raw_count_scale()` when `normalize` is used (e.g. `"cpm"` or `"rpkm"`), so `protection_score` with `normalize=` equals the raw-count score instead of changing with the units. The pseudocount stays in raw cuts per bp, so scores still depend on depth as before.
- `_read_anchors_bed` deduplicates repeated column-4 motif names (`CTCF_0`, `CTCF_1`, ...) so `footprint_score` and `SignalMatrix.to_dataframe()` keep distinct per-locus identifiers.
- `VPlot.diff` warns when `smooth_sigma` is passed and one or both V-plots were already smoothed before `diff()`.

### Added (merge provenance)
- `FragmentStore.merge` keeps each input's metadata (including `ingest_stats` and `exclude_regions`) in `metadata["merged_inputs"]`, and with `dedup=True` records `metadata["removed_coord_dedup"]`.

### Added (locus plot colours)
- `LocusVPlot.plot(palette=)` / `plot_plotnine(palette=)`: a `{level: colour}` dict merged over the default colours for `color_by="end"` (midpoint/proximal/distal/left/right) or `"motif_overlap"` (full/partial/none). Unknown levels raise.

### Added (ingestion progress)
- `FragmentStore.from_bam(verbose=True)` prints progress to stderr (on each new chromosome and every 5M records) and a final summary of records read, skipped (SAM flags, pairing, MAPQ, chromosome, length), removed by coordinate dedup and blacklist, and kept. The same counts are always stored in `metadata["ingest_stats"]`.

### Added (blacklist)
- `FragmentStore.exclude_regions(regions, mode=)` drops fragments overlapping (or with their midpoint in) a BED file or DataFrame of regions; `from_bam` / `from_fragments_bed` take `exclude_regions=` and `exclude_mode=`; `pyvplot-report` takes `--blacklist` and `--blacklist-mode`. The removal is recorded in `metadata["exclude_regions"]`.

### Fixed
- `KmerBiasTable.from_cuts` placed right-end cuts one base out in the reverse-complement frame, so right ends taught the table nothing. `correct_matrix` now scores left-end (`cuts_plus`), right-end (`cuts_minus`) and combined (`cuts`) matrices in the matching frame, and rejects non-cut features.
- `frip(..., return_per_peak=True)` raised `IndexError` for peaks DataFrames without a default index.
- `FragmentStore.save(path, format="npz")` returned a path that did not exist when `path` lacked the `.npz` suffix.
- `VPlot.smooth` rejected NumPy integer `sigma`.
- `to_bigwig` silently dropped bins on chromosomes missing from `chrom_sizes`; it now warns.
- Single-locus view: `motif_overlap` is empty without a motif, and `to_vplot()` carries the store's fragment count so CPM matches the aggregate V-plots.
- BED-like readers name the file and line when a header line is not prefixed with `#`/`track`/`browser`.

### Changed (behaviour)
- `VPlot.normalize` and `SignalMatrix.normalize` raise on an already normalised object.
- `VPlot.diff` raises when the two V-plots are in different units.
- `normalize_by_control` defaults to `pseudocount=None` (half a raw count in the inputs' units) instead of `1.0`, and raises on mixed units.
- Stranded stores reduce read intervals to their 5' base in `from_arrays` / `from_dataframe`, and the constructor rejects longer intervals.
- `FragmentStore.merge` raises when assays differ (pass `assay=`), and records the inputs' sources.
- `cut_site_footprint(split_strands=False, split_by="proximity")` raises.
- `extract_events` adds an `anchor` column with the anchor's index label.

### Added
- `pyvplot-report` console command (the former `examples/bam_to_vplot.py`), `from_bam(threads=)`, `SignalMatrix.raw_count_scale`, `KmerBiasTable.score_sequence_reverse`, `LICENSE`, `py.typed`, single-sourced version.

### Performance
- BAM and BED ingestion keep coordinates in compact arrays (about 35 bytes per fragment instead of about 105). `KmerBiasTable.from_cuts` encodes chromosomes in windows instead of all at once.
