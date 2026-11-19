# Fragment QC & Nucleosome Phasing (NRL)

## Insert-Size Distribution & QC Summary

```python
import pyvplot as vp

dist = vp.insert_size_distribution(frags, min_len=20, max_len=600, normalize=True)
# Columns: length, count, fraction

summary = vp.qc_summary(frags, peaks="peaks.bed")
# Keys: assay, n_fragments, n_chroms, mean_length, median_length,
#       min_length, max_length, size_band_fractions, frip
```

## FRiP (Fraction of Fragments in Peaks)

```python
# Overlapping peak intervals are automatically merged before counting global FRiP
score = vp.frip(frags, peaks_df, mode="midpoint")                # or mode="overlap"
score, per_peak_counts = vp.frip(frags, peaks_df, mode="overlap", return_per_peak=True)
```

## Nucleosome Phasogram & NRL Estimation

```python
ph = vp.phasogram(
    frags,
    max_lag=1000,
    len_range=(135, 165),           # restrict to mononucleosomes if mixed assay
    mode="midpoint",                # "midpoint" (dyad-to-dyad), "start", or "end"
    smooth_sigma=5.0,
    min_peak_lag=120,
    max_peak_lag=850,
    expected_nrl_range=(140.0, 240.0),
)

print(ph.nrl, ph.nrl_peak_fit, ph.nrl_fft, ph.r_squared, ph.peak_positions)
df = ph.to_dataframe()              # columns: lag, count, smoothed
fig, ax = ph.plot()
p = ph.plot_plotnine()
```

`vp.estimate_nrl` detrends the smoothed phasogram with a wide Gaussian baseline, computes both an FFT periodogram (`nrl_fft`) and a harmonic peak-spacing linear regression (`nrl_peak_fit` + `r_squared`), and uses `nrl_peak_fit` when at least 2 consistent peaks fall inside `expected_nrl_range` (falling back to `nrl_fft`).

## Footguns

- **Filter to mononucleosomes on ATAC / ChEC:** Short TF fragments add a sharp low-lag peak (< 100 bp) and wash out nucleosome periodicity. Pass `len_range=(135, 165)` (MNase) or `len_range=(180, 247)` (ATAC) to `vp.phasogram`.
- **Deep datasets:** Pair counting scales with local fragment density within `max_lag`. On >50M fragment mammalian libraries, restrict `chroms={"chr1"}` or pass `max_positions_per_chrom=500_000` for an instant estimate.
- **Peak overlap double-counting:** Global `vp.frip` merges overlapping peaks so FRiP never exceeds `1.0`, whereas `per_peak_counts` reports raw counts per input peak row.
