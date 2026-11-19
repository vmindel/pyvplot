# Fragment QC and nucleosome phasing

## Insert sizes, size classes and FRiP

`qc_summary()` reports fragment count, length statistics, and the fraction of fragments in each of the assay's named size bins (inclusive bounds; bins may overlap or leave gaps). With `peaks=`, it also reports FRiP: the fraction of fragments whose midpoint falls in a peak, with overlapping peaks merged first. `store.frip(peaks, mode="overlap")` counts any overlap instead, and `return_per_peak=True` also returns per-peak counts.

Look at `store.insert_size_distribution()` before trusting the preset bins: they are generic starting points.

```python
import numpy as np
import pandas as pd
import pyvplot as vp

rng = np.random.default_rng(7)
nfr_lens = rng.integers(40, 80, size=300, dtype=np.int32)
nuc_lens = rng.integers(190, 220, size=200, dtype=np.int32)
lens = np.concatenate([nfr_lens, nuc_lens])
starts = np.arange(1_000, 1_000 + len(lens) * 50, 50, dtype=np.int32)

atac = vp.FragmentStore({"chr1": starts}, {"chr1": starts + lens}, assay="atac")
peaks = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [1_000, 10_000], "end": [8_500, 18_500]})

summary = atac.qc_summary(peaks=peaks)
print("n_fragments:", summary["n_fragments"])
print("bands:", {k: round(v, 2) for k, v in summary["size_band_fractions"].items()})
print(f"FRiP: {summary['frip']:.2f}")
```

```text
n_fragments: 500
bands: {'nfr': 0.6, 'mononucleosome': 0.4, 'dinucleosome': 0.0}
FRiP: 0.64
```

Plotting an ATAC-seq store with `plot_insert_size_matplotlib(..., show_bands=True)` automatically shades the assay's NFR, mononucleosome, and dinucleosome size regimes on both linear and log10 scales:

![ATAC-seq insert-size distribution with shaded nucleosome bands](../img/insert_size_qc.png)

## Nucleosome phasogram & NRL estimation

`phasogram()` histograms the distances between all pairs of fragment midpoints on the same chromosome, up to `max_lag` bp. Regularly spaced nucleosome arrays give peaks at multiples of the repeat length. NRL is estimated two ways. `nrl_peak_fit` is the slope of peak position against peak number, with `r_squared` from that fit. `nrl_fft` comes from the dominant period in the spectrum. If they disagree by more than a few bp, or `r_squared` is low, there is little phasing to measure.

Restrict to nucleosome-sized fragments, as a tuple or a size-bin name (`len_range="nucleosome"` for the MNase preset). The pair count grows with the square of local density, so on deep libraries use `chroms=` or `max_positions_per_chrom=` for a quick estimate.

```python
rng = np.random.default_rng(2026)
starts_list, ends_list = [], []
for array_idx in range(30):
    base = 5_000 + array_idx * 5_000
    for k in range(8):
        dyad_center = base + k * 185
        for _ in range(10):
            dyad = int(round(dyad_center + rng.normal(0.0, 8.0)))
            fl = int(rng.integers(145, 155))
            starts_list.append(dyad - fl // 2)
            ends_list.append(dyad - fl // 2 + fl)

mnase = vp.FragmentStore(
    {"chr1": np.asarray(starts_list, dtype=np.int32)},
    {"chr1": np.asarray(ends_list, dtype=np.int32)},
    assay="mnase",
)
ph = mnase.phasogram(max_lag=900, smooth_sigma=6.0)
print(f"Estimated NRL: {ph.nrl:.1f} bp (FFT: {ph.nrl_fft:.1f} bp, R^2: {ph.r_squared:.4f})")
```

```text
Estimated NRL: 185.0 bp (FFT: 186.2 bp, R^2: 1.0000)
```

Calling `ph.plot()` overlays the raw pair counts, smoothed phasogram, detected nucleosome harmonics, and estimated NRL:

![Nucleosome phasogram with detected periodic peaks and 185 bp NRL estimate](../img/phasogram_nrl.png)
