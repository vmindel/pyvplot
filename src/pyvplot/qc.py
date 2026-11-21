"""Fragment QC metrics: insert-size distributions, FRiP, nucleosome phasograms, and NRL."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
import pandas as pd

from pyvplot.assays import resolve_len_range
from pyvplot.extract import HAS_NUMBA, njit
from pyvplot.store import FragmentStore, load_intervals


def _gaussian_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1D Gaussian smoothing via SciPy or pure-NumPy fallback."""
    if sigma <= 0.0 or len(arr) == 0:
        return arr.astype(np.float64, copy=True)
    try:
        from scipy.ndimage import gaussian_filter1d

        return np.asarray(gaussian_filter1d(arr.astype(np.float64, copy=False), sigma=sigma, mode="nearest"), dtype=np.float64)
    except ImportError:
        radius = max(1, int(np.ceil(4.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(arr.astype(np.float64, copy=False), (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")


@njit(cache=True)
def _phasogram_kernel(positions: np.ndarray, max_lag: int) -> np.ndarray:
    hist = np.zeros(max_lag, dtype=np.int64)
    n = positions.shape[0]
    for i in range(n):
        pi = positions[i]
        for j in range(i + 1, n):
            d = positions[j] - pi
            if d > max_lag:
                break
            if d >= 1:
                hist[d - 1] += 1
    return hist


def _phasogram_numpy(positions: np.ndarray, max_lag: int) -> np.ndarray:
    hist = np.zeros(max_lag, dtype=np.int64)
    n = len(positions)
    if n < 2:
        return hist
    # Vectorized shift-and-accumulate up to the max index offset within max_lag
    k = 1
    while k < n:
        diffs = positions[k:] - positions[:-k]
        valid = (diffs >= 1) & (diffs <= max_lag)
        if not np.any(valid):
            if np.all(diffs > max_lag):
                break
        else:
            hist += np.bincount(diffs[valid] - 1, minlength=max_lag)
        k += 1
    return hist


def insert_size_distribution(
    store: FragmentStore,
    min_len: int | None = None,
    max_len: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute per-base-pair fragment insert-size histogram.

    Returns a `pandas.DataFrame` with columns ``length``, ``count``,
    and (when ``normalize=True``) ``fraction``.
    """
    lens = store.lengths()
    if len(lens) == 0:
        cols = ["length", "count", "fraction"] if normalize else ["length", "count"]
        return pd.DataFrame(columns=cols)

    lo = int(np.min(lens)) if min_len is None else int(min_len)
    hi = int(np.max(lens)) if max_len is None else int(max_len)
    if hi < lo:
        cols = ["length", "count", "fraction"] if normalize else ["length", "count"]
        return pd.DataFrame(columns=cols)

    mask = (lens >= lo) & (lens <= hi)
    sub = lens[mask]
    counts = np.bincount(sub - lo, minlength=hi - lo + 1).astype(np.int64)
    lengths = np.arange(lo, hi + 1, dtype=np.int32)
    df = pd.DataFrame({"length": lengths, "count": counts})
    if normalize:
        total = float(np.sum(counts))
        df["fraction"] = counts / total if total > 0.0 else 0.0
    return df


def _load_peaks_df(peaks: Any) -> pd.DataFrame:
    return load_intervals(peaks, what="Peaks")


def _merge_sorted_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(starts, kind="mergesort")
    s = starts[order]
    e = ends[order]
    m_starts = [int(s[0])]
    m_ends = [int(e[0])]
    for i in range(1, len(s)):
        if int(s[i]) <= m_ends[-1]:
            if int(e[i]) > m_ends[-1]:
                m_ends[-1] = int(e[i])
        else:
            m_starts.append(int(s[i]))
            m_ends.append(int(e[i]))
    return np.asarray(m_starts, dtype=np.int64), np.asarray(m_ends, dtype=np.int64)


@overload
def frip(
    store: FragmentStore, peaks: Any, mode: str = ..., return_per_peak: Literal[False] = ...
) -> float: ...


@overload
def frip(
    store: FragmentStore, peaks: Any, mode: str = ..., *, return_per_peak: Literal[True]
) -> tuple[float, np.ndarray]: ...


@overload
def frip(
    store: FragmentStore, peaks: Any, mode: str = ..., return_per_peak: bool = ...
) -> float | tuple[float, np.ndarray]: ...


def frip(
    store: FragmentStore,
    peaks: Any,
    mode: str = "midpoint",
    return_per_peak: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute Fraction of Fragments in Peaks (FRiP).

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to evaluate.
    peaks:
        Peak intervals (DataFrame with ``chrom``, ``start``, ``end`` or BED path).
    mode:
        ``"midpoint"`` (fragment midpoint falls within a peak) or ``"overlap"``
        (any base of ``[start, end)`` overlaps a peak).
    return_per_peak:
        If ``True``, return ``(frip_score, per_peak_counts)`` where
        ``per_peak_counts`` matches the row order of ``peaks``.
    """
    if mode not in {"midpoint", "overlap"}:
        raise ValueError(f"Invalid mode {mode!r}; expected 'midpoint' or 'overlap'.")

    peaks_df = _load_peaks_df(peaks)
    per_peak = np.zeros(len(peaks_df), dtype=np.int64)

    if store.n_fragments == 0 or len(peaks_df) == 0:
        return (0.0, per_peak) if return_per_peak else 0.0

    in_peaks_total = 0
    for chrom, grp in peaks_df.groupby("chrom", sort=False):
        c = str(chrom)
        if c not in store.starts or len(store.starts[c]) == 0:
            continue
        fs = store.starts[c].astype(np.int64, copy=False)
        fe = store.ends[c].astype(np.int64, copy=False)
        p_starts = grp["start"].to_numpy(dtype=np.int64)
        p_ends = grp["end"].to_numpy(dtype=np.int64)
        m_starts, m_ends = _merge_sorted_intervals(p_starts, p_ends)

        if mode == "midpoint":
            mids = (fs + fe) // 2
            idx = np.searchsorted(m_starts, mids, side="right") - 1
            valid = (idx >= 0) & (mids < m_ends[np.maximum(idx, 0)])
            in_peaks_total += int(np.sum(valid))

            if return_per_peak:
                sorted_mids = np.sort(mids)
                lo = np.searchsorted(sorted_mids, p_starts, side="left")
                hi = np.searchsorted(sorted_mids, p_ends, side="left")
                per_peak[grp.index.to_numpy()] = hi - lo
        else:
            # Overlap with merged intervals: a fragment [fs, fe) overlaps merged [ms, me)
            # iff there exists k with ms[k] < fe and me[k] > fs.
            # Because merged intervals are disjoint and sorted, the candidate interval
            # with the largest ms < fe is k = searchsorted(m_starts, fe, side="left") - 1.
            k = np.searchsorted(m_starts, fe, side="left") - 1
            valid = (k >= 0) & (m_ends[np.maximum(k, 0)] > fs)
            in_peaks_total += int(np.sum(valid))

            if return_per_peak:
                max_fl = store.max_len[c]
                orig_idx = grp.index.to_numpy()
                for pi_i in range(len(p_starts)):
                    ps = int(p_starts[pi_i])
                    pe = int(p_ends[pi_i])
                    lo = int(np.searchsorted(fs, ps - max_fl, side="left"))
                    hi = int(np.searchsorted(fs, pe, side="left"))
                    if lo < hi:
                        per_peak[orig_idx[pi_i]] = int(np.sum(fe[lo:hi] > ps))

    score = float(in_peaks_total) / float(store.n_fragments)
    return (score, per_peak) if return_per_peak else score


@dataclass(repr=False)
class PhasogramResult:
    """Nucleosome phasing pair-distance histogram and NRL estimate.

    Attributes
    ----------
    lags:
        1D integer array of lag distances ``1 .. max_lag`` (bp).
    counts:
        Raw pair counts at each lag distance.
    smoothed:
        Gaussian-smoothed pair counts.
    nrl:
        Primary Nucleosome Repeat Length (bp) estimate, or ``None`` if no
        periodic nucleosome phasing was detected.
    nrl_peak_fit:
        NRL estimated by linear regression across detected nucleosome peaks.
    nrl_fft:
        NRL estimated from the dominant peak in the detrended FFT power spectrum.
    r_squared:
        Coefficient of determination ($R^2$) of the peak-spacing linear fit.
    peak_positions:
        1D array of detected nucleosome peak lags (bp).
    power_spectrum:
        Optional ``(periods, power)`` tuple from FFT analysis.
    """

    lags: np.ndarray
    counts: np.ndarray
    smoothed: np.ndarray
    nrl: float | None = None
    nrl_peak_fit: float | None = None
    nrl_fft: float | None = None
    r_squared: float | None = None
    peak_positions: np.ndarray = None  # type: ignore[assignment]
    power_spectrum: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.peak_positions is None:
            self.peak_positions = np.empty(0, dtype=np.float64)

    def __repr__(self) -> str:
        def _f(x: float | None) -> str:
            return "None" if x is None else f"{x:.1f}"

        r2 = "None" if self.r_squared is None else f"{self.r_squared:.3f}"
        return (
            f"PhasogramResult(nrl={_f(self.nrl)}, nrl_peak_fit={_f(self.nrl_peak_fit)}, "
            f"nrl_fft={_f(self.nrl_fft)}, r_squared={r2}, n_peaks={len(self.peak_positions)}, "
            f"max_lag={int(self.lags[-1]) if len(self.lags) else 0}, "
            f"n_pairs={int(np.sum(self.counts)):,})"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return phasogram curve as a pandas DataFrame (``lag``, ``count``, ``smoothed``)."""
        return pd.DataFrame(
            {
                "lag": self.lags,
                "count": self.counts,
                "smoothed": self.smoothed,
            }
        )

    def plot(self, **kwargs: Any) -> Any:
        """Render phasogram with ``matplotlib``."""
        from pyvplot.plotting import plot_phasogram_matplotlib

        return plot_phasogram_matplotlib(self, **kwargs)

    def plot_plotnine(self, **kwargs: Any) -> Any:
        """Render phasogram with ``plotnine``."""
        from pyvplot.plotting import plot_phasogram_plotnine

        return plot_phasogram_plotnine(self, **kwargs)


def _find_local_peaks(signal: np.ndarray, min_distance: int) -> np.ndarray:
    """Detect local peak indices with minimum separation ``min_distance``."""
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(signal, distance=max(1, int(min_distance)), prominence=0.0)
        return np.asarray(peaks, dtype=np.int64)
    except ImportError:
        n = len(signal)
        if n < 3:
            return np.empty(0, dtype=np.int64)
        cand = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:]))[0] + 1
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        # Greedy non-maximum suppression by height
        order = cand[np.argsort(signal[cand])[::-1]]
        selected: list[int] = []
        for idx in order:
            if all(abs(int(idx) - s) >= min_distance for s in selected):
                selected.append(int(idx))
        selected.sort()
        return np.asarray(selected, dtype=np.int64)


def estimate_nrl(
    lags: np.ndarray,
    signal: np.ndarray,
    min_lag: int = 120,
    max_lag: int = 850,
    expected_nrl_range: tuple[float, float] = (140.0, 240.0),
) -> dict[str, Any]:
    """Estimate Nucleosome Repeat Length (NRL) from a smoothed phasogram curve.

    Combines two complementary estimators:
    1. **Peak-spacing linear regression:** detects periodic nucleosome peaks in
       ``[min_lag, max_lag]`` on the detrended phasogram and fits
       ``peak_lag = nrl * k + offset``.
    2. **FFT power spectrum:** computes the periodogram of the detrended signal
       and identifies the dominant period inside ``expected_nrl_range``.
    """
    lags_arr = np.asarray(lags, dtype=np.float64)
    sig_arr = np.asarray(signal, dtype=np.float64)
    nrl_lo, nrl_hi = float(expected_nrl_range[0]), float(expected_nrl_range[1])

    mask = (lags_arr >= min_lag) & (lags_arr <= max_lag)
    sub_lags = lags_arr[mask]
    sub_sig = sig_arr[mask]

    if len(sub_lags) < 20 or float(np.std(sub_sig)) == 0.0:
        return {
            "nrl": None,
            "nrl_peak_fit": None,
            "nrl_fft": None,
            "r_squared": None,
            "peak_positions": np.empty(0, dtype=np.float64),
            "power_spectrum": None,
        }

    # Remove broad background trend via wide Gaussian envelope
    baseline = _gaussian_1d(sub_sig, sigma=max(40.0, 0.45 * nrl_lo))
    detrended = sub_sig - baseline

    # 1. FFT power spectrum estimate (zero-padded for sub-bp frequency resolution)
    n_fft = max(4096, 1 << int(np.ceil(np.log2(len(detrended) * 4))))
    window = np.hanning(len(detrended))
    spec = np.abs(np.fft.rfft(detrended * window, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    valid_f = (freqs > 0.0) & (1.0 / np.where(freqs > 0.0, freqs, 1.0) >= nrl_lo) & (
        1.0 / np.where(freqs > 0.0, freqs, 1.0) <= nrl_hi
    )
    nrl_fft: float | None = None
    power_spectrum: tuple[np.ndarray, np.ndarray] | None = None
    if np.any(valid_f):
        periods = 1.0 / freqs[valid_f]
        powers = spec[valid_f]
        order = np.argsort(periods)
        periods = periods[order]
        powers = powers[order]
        power_spectrum = (periods, powers)
        best_idx = int(np.argmax(powers))
        if powers[best_idx] > 0.0:
            nrl_fft = float(periods[best_idx])

    # 2. Peak-spacing linear regression
    min_dist = max(10, int(0.75 * nrl_lo))
    peak_idxs = _find_local_peaks(detrended, min_distance=min_dist)
    # Keep positive peaks above background trend
    peak_idxs = peak_idxs[detrended[peak_idxs] > 0.0]
    peak_lags = sub_lags[peak_idxs]

    # Filter peaks to the dominant harmonic chain consistent with expected_nrl_range
    nrl_peak_fit: float | None = None
    r_squared: float | None = None
    used_peaks = peak_lags

    if len(peak_lags) >= 2:
        # Use FFT period (if available) or first peak to assign harmonic indices k = 1, 2, ...
        seed_period = nrl_fft if nrl_fft is not None else float(peak_lags[0])
        if nrl_lo <= seed_period <= nrl_hi:
            ks = np.maximum(1, np.round(peak_lags / seed_period).astype(np.int64))
            # Keep the strongest peak for each harmonic k
            best_by_k: dict[int, tuple[float, float]] = {}
            for idx_p, k_val in zip(peak_idxs, ks, strict=True):
                h = float(detrended[idx_p])
                lag_p = float(sub_lags[idx_p])
                if int(k_val) not in best_by_k or h > best_by_k[int(k_val)][1]:
                    best_by_k[int(k_val)] = (lag_p, h)
            sorted_ks = np.asarray(sorted(best_by_k.keys()), dtype=np.float64)
            sorted_lags = np.asarray([best_by_k[int(k)][0] for k in sorted_ks], dtype=np.float64)

            if len(sorted_ks) >= 2:
                used_peaks = sorted_lags
                slope, intercept = np.polyfit(sorted_ks, sorted_lags, deg=1)
                preds = slope * sorted_ks + intercept
                ss_res = float(np.sum((sorted_lags - preds) ** 2))
                ss_tot = float(np.sum((sorted_lags - np.mean(sorted_lags)) ** 2))
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
                if nrl_lo <= slope <= nrl_hi:
                    nrl_peak_fit = float(slope)
                    r_squared = float(r2)

    primary_nrl = nrl_peak_fit if nrl_peak_fit is not None else nrl_fft
    return {
        "nrl": primary_nrl,
        "nrl_peak_fit": nrl_peak_fit,
        "nrl_fft": nrl_fft,
        "r_squared": r_squared,
        "peak_positions": used_peaks,
        "power_spectrum": power_spectrum,
    }


def phasogram(
    store: FragmentStore,
    max_lag: int = 1000,
    chroms: Collection[str] | None = None,
    len_range: tuple[int, int] | str | None = None,
    mode: str = "midpoint",
    max_positions_per_chrom: int | None = None,
    smooth_sigma: float = 5.0,
    min_peak_lag: int = 120,
    max_peak_lag: int = 850,
    expected_nrl_range: tuple[float, float] = (140.0, 240.0),
    use_numba: bool | None = None,
) -> PhasogramResult:
    """Compute nucleosome phasogram (pair-distance histogram in ``[1, max_lag]`` bp) and NRL.

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to analyze.
    max_lag:
        Maximum pair distance in bp (default ``1000``).
    chroms:
        Optional subset of chromosomes to include.
    len_range:
        Optional inclusive ``(min_len, max_len)`` fragment-length filter (e.g.,
        ``(135, 165)`` for mononucleosomes), or the name of a size bin of the
        store's assay (``"nucleosome"``, ``"mononucleosome"``, ...).
    mode:
        Position used per fragment: ``"midpoint"`` (dyad-to-dyad, default),
        ``"start"`` (same-strand 5' left ends), or ``"end"`` (right ends).
    max_positions_per_chrom:
        Optional cap on positions per chromosome (takes first N sorted positions)
        for quick approximate estimates on very deep datasets.
    smooth_sigma:
        Gaussian smoothing sigma (bp) applied prior to NRL estimation.
    min_peak_lag, max_peak_lag:
        Lag window (bp) searched for nucleosome phasing peaks.
    expected_nrl_range:
        Physiologically plausible ``(min_nrl, max_nrl)`` range in bp.
    use_numba:
        Override automatic Numba dispatch.
    """
    if max_lag < 2:
        raise ValueError("`max_lag` must be >= 2.")
    if mode not in {"midpoint", "start", "end"}:
        raise ValueError(f"Invalid mode {mode!r}; expected 'midpoint', 'start', or 'end'.")

    len_range = resolve_len_range(store.assay, len_range)
    run_numba = HAS_NUMBA if use_numba is None else bool(use_numba and HAS_NUMBA)
    target_chroms = list(store.chroms) if chroms is None else [c for c in store.chroms if c in set(chroms)]

    total_hist = np.zeros(max_lag, dtype=np.int64)

    for chrom in target_chroms:
        s = store.starts[chrom]
        e = store.ends[chrom]
        if len(s) < 2:
            continue
        if len_range is not None:
            fl = e - s
            mask = (fl >= len_range[0]) & (fl <= len_range[1])
            s = s[mask]
            e = e[mask]
            if len(s) < 2:
                continue

        if mode == "midpoint":
            pos = np.sort((s.astype(np.int64) + e.astype(np.int64)) // 2)
        elif mode == "start":
            pos = s.astype(np.int64, copy=False)
        else:
            pos = np.sort(e.astype(np.int64, copy=False))

        if max_positions_per_chrom is not None and len(pos) > max_positions_per_chrom:
            pos = pos[:max_positions_per_chrom]

        pos = np.ascontiguousarray(pos, dtype=np.int64)
        if run_numba:
            total_hist += _phasogram_kernel(pos, int(max_lag))
        else:
            total_hist += _phasogram_numpy(pos, int(max_lag))

    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    smoothed = _gaussian_1d(total_hist.astype(np.float64), sigma=smooth_sigma)
    nrl_info = estimate_nrl(
        lags=lags,
        signal=smoothed,
        min_lag=min_peak_lag,
        max_lag=min(max_peak_lag, max_lag),
        expected_nrl_range=expected_nrl_range,
    )

    return PhasogramResult(
        lags=lags,
        counts=total_hist,
        smoothed=smoothed,
        nrl=nrl_info["nrl"],
        nrl_peak_fit=nrl_info["nrl_peak_fit"],
        nrl_fft=nrl_info["nrl_fft"],
        r_squared=nrl_info["r_squared"],
        peak_positions=nrl_info["peak_positions"],
        power_spectrum=nrl_info["power_spectrum"],
    )


def qc_summary(store: FragmentStore, peaks: Any = None) -> dict[str, Any]:
    """Compute a summary QC dictionary for a [`FragmentStore`][pyvplot.FragmentStore]."""
    lens = store.lengths()
    summary: dict[str, Any] = {
        "assay": store.assay.name,
        "n_fragments": store.n_fragments,
        "n_chroms": len(store.starts),
        "mean_length": float(np.mean(lens)) if len(lens) > 0 else 0.0,
        "median_length": float(np.median(lens)) if len(lens) > 0 else 0.0,
        "min_length": int(np.min(lens)) if len(lens) > 0 else 0,
        "max_length": int(np.max(lens)) if len(lens) > 0 else 0,
    }

    band_fractions: dict[str, float] = {}
    if len(lens) > 0:
        n_total = float(len(lens))
        for band_name, (lo, hi) in store.assay.size_bins.items():
            band_fractions[band_name] = float(np.sum((lens >= lo) & (lens <= hi))) / n_total
    else:
        for band_name in store.assay.size_bins:
            band_fractions[band_name] = 0.0
    summary["size_band_fractions"] = band_fractions

    if peaks is not None:
        summary["frip"] = frip(store, peaks)

    return summary
