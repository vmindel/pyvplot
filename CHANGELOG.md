# Changelog

## Unreleased

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
