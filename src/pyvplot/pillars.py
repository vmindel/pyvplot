"""Cut-pillar and de-novo protected motif footprint scanning from fragment cleavage geometry.

Identifies multi-length cleavage pillars (genomic boundaries where fragment ends accumulate
across many distinct fragment lengths) and protected motif footprints (intervals ``[l, r)``
between pairs of called cleavage pillars at boundaries ``l`` and ``r`` whose core interior is
depleted of cuts and spanned by short transcription-factor fragments).

Coordinate convention: index ``p`` of ``left_cuts`` / ``right_cuts`` / ``pillar_score`` is the
cut boundary at genomic coordinate ``L + p`` (a fragment ``[s, e)`` contributes a left cut at
``s - L`` and a right cut at ``e - L``). The protected bases between pillars at boundaries
``l`` and ``r`` are ``[l, r)``.

Every result exposes ``chrom, start, end`` pandas DataFrames ready for ``pygenomic_tracks``,
bedGraph/BED export, or downstream sequence analysis with ``moodspy``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pyvplot.extract import HAS_NUMBA
from pyvplot.locus import _parse_region

if TYPE_CHECKING:
    from pyvplot.bias import KmerBiasTable
    from pyvplot.store import FragmentStore

if HAS_NUMBA:
    import numba


# ---------------------------------------------------------------------------
# Fast region accumulation & unified profile kernels
# ---------------------------------------------------------------------------

def _accumulate_region_py(
    starts: np.ndarray,
    ends: np.ndarray,
    L: int,
    R: int,
    min_len: int,
    max_len: int,
    len_bin: int,
    tf_min_len: int,
    tf_max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W = int(R - L)
    n_bins = max(1, (int(max_len) - int(min_len)) // int(len_bin))
    cut_mat = np.zeros((n_bins, W), dtype=np.float32)
    left_cuts = np.zeros(W, dtype=np.float32)
    right_cuts = np.zeros(W, dtype=np.float32)
    cov_diff = np.zeros(W + 1, dtype=np.float32)

    if len(starts) == 0 or W <= 0:
        return cut_mat, left_cuts, right_cuts, np.zeros(W, dtype=np.float32)

    lookback = max(int(max_len), int(tf_max_len), 1000)
    i0 = int(np.searchsorted(starts, L - lookback))
    i1 = int(np.searchsorted(starts, R))

    s = starts[i0:i1].astype(np.int64)
    e = ends[i0:i1].astype(np.int64)
    keep = (e > L) & (s < R)
    s = s[keep]
    e = e[keep]
    fl = e - s

    tf_mask = (fl >= tf_min_len) & (fl <= tf_max_len)
    if np.any(tf_mask):
        cs = np.clip(s[tf_mask] - L, 0, W)
        ce = np.clip(e[tf_mask] - L, 0, W)
        ok = ce > cs
        np.add.at(cov_diff, cs[ok], 1.0)
        np.add.at(cov_diff, ce[ok], -1.0)

    len_mask = (fl >= min_len) & (fl < min_len + n_bins * len_bin)
    if np.any(len_mask):
        sl = s[len_mask]
        el = e[len_mask]
        b = ((fl[len_mask] - min_len) // len_bin).astype(np.int64)

        m_s = (sl >= L) & (sl < R)
        if np.any(m_s):
            idx_s = sl[m_s] - L
            np.add.at(cut_mat, (b[m_s], idx_s), 1.0)
            np.add.at(left_cuts, idx_s, 1.0)

        m_e = (el >= L) & (el < R)
        if np.any(m_e):
            idx_e = el[m_e] - L
            np.add.at(cut_mat, (b[m_e], idx_e), 1.0)
            np.add.at(right_cuts, idx_e, 1.0)

    cov_tf = np.cumsum(cov_diff[:-1]).astype(np.float32)
    return cut_mat, left_cuts, right_cuts, cov_tf


if HAS_NUMBA:

    @numba.njit(nogil=True, cache=True)
    def _accumulate_region_numba(
        starts: np.ndarray,
        ends: np.ndarray,
        L: int,
        R: int,
        min_len: int,
        max_len: int,
        len_bin: int,
        tf_min_len: int,
        tf_max_len: int,
    ):
        W = R - L
        n_bins = max(1, (max_len - min_len) // len_bin)
        cut_mat = np.zeros((n_bins, W), dtype=np.float32)
        left_cuts = np.zeros(W, dtype=np.float32)
        right_cuts = np.zeros(W, dtype=np.float32)
        cov_diff = np.zeros(W + 1, dtype=np.float32)

        if len(starts) == 0 or W <= 0:
            return cut_mat, left_cuts, right_cuts, np.zeros(W, dtype=np.float32)

        lookback = max(max_len, tf_max_len, 1000)
        i0 = np.searchsorted(starts, L - lookback)
        i1 = np.searchsorted(starts, R)
        max_eff = min_len + n_bins * len_bin

        for i in range(i0, i1):
            s = int(starts[i])
            e = int(ends[i])
            if e <= L or s >= R:
                continue
            fl = e - s
            if fl >= tf_min_len and fl <= tf_max_len:
                cs = s - L if s > L else 0
                ce = e - L if e < R else W
                if ce > cs:
                    cov_diff[cs] += 1.0
                    cov_diff[ce] -= 1.0
            if fl >= min_len and fl < max_eff:
                b = (fl - min_len) // len_bin
                if s >= L and s < R:
                    idx = s - L
                    cut_mat[b, idx] += 1.0
                    left_cuts[idx] += 1.0
                if e >= L and e < R:
                    idx = e - L
                    cut_mat[b, idx] += 1.0
                    right_cuts[idx] += 1.0

        cov_tf = np.zeros(W, dtype=np.float32)
        acc = 0.0
        for j in range(W):
            acc += cov_diff[j]
            cov_tf[j] = acc
        return cut_mat, left_cuts, right_cuts, cov_tf


def _accumulate_region(
    starts: np.ndarray,
    ends: np.ndarray,
    L: int,
    R: int,
    min_len: int,
    max_len: int,
    len_bin: int,
    tf_min_len: int,
    tf_max_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if HAS_NUMBA:
        return _accumulate_region_numba(
            starts, ends, int(L), int(R), int(min_len), int(max_len), int(len_bin), int(tf_min_len), int(tf_max_len)
        )
    return _accumulate_region_py(
        starts, ends, int(L), int(R), int(min_len), int(max_len), int(len_bin), int(tf_min_len), int(tf_max_len)
    )


def _box_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast 1D sliding sum with nearest-edge padding."""
    w = int(window)
    if w <= 1 or len(arr) == 0:
        return arr.astype(np.float64, copy=True)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    padded = np.pad(arr.astype(np.float64), (pad_l, pad_r), mode="edge")
    cs = np.concatenate([[0.0], np.cumsum(padded)])
    return cs[w:] - cs[:-w]


def _box_filter_2d_cols(mat: np.ndarray, window: int) -> np.ndarray:
    """Fast 1D sliding sum along axis=1 (columns) with nearest-edge padding."""
    w = int(window)
    if w <= 1 or mat.shape[1] == 0:
        return mat.astype(np.float64, copy=True)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    padded = np.pad(mat.astype(np.float64), ((0, 0), (pad_l, pad_r)), mode="edge")
    cs = np.concatenate([np.zeros((mat.shape[0], 1), dtype=np.float64), np.cumsum(padded, axis=1)], axis=1)
    return cs[:, w:] - cs[:, :-w]


def _max_filter_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """1D sliding maximum filter with edge padding using pure NumPy."""
    w = int(window)
    n = len(arr)
    if w <= 1 or n == 0:
        return arr.astype(np.float64, copy=True)
    pad_l = w // 2
    pad_r = w - 1 - pad_l
    padded = np.pad(arr.astype(np.float64), (pad_l, pad_r), mode="edge")
    shape = (n, w)
    strides = (padded.strides[0], padded.strides[0])
    view = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return view.max(axis=1)


def _find_peaks_1d(
    y: np.ndarray,
    min_height: float,
    min_distance: int = 5,
    min_prominence: float = 0.0,
    prom_window: int = 12,
) -> np.ndarray:
    """Vectorized 1D local peak finder with height, distance, and local prominence filtering."""
    n = len(y)
    if n < 3:
        return np.empty(0, dtype=np.int64)
    mid = y[1:-1]
    mask = (mid >= min_height) & (mid > y[:-2]) & (mid >= y[2:])
    cand_arr = np.flatnonzero(mask).astype(np.int64) + 1
    if len(cand_arr) == 0:
        return np.empty(0, dtype=np.int64)

    if min_prominence > 0.0:
        w = max(1, int(prom_window))
        pad = np.pad(y.astype(np.float64), (w, w), mode="constant", constant_values=np.inf)
        strides = (pad.strides[0], pad.strides[0])
        left_view = np.lib.stride_tricks.as_strided(pad[: n + w], shape=(n, w), strides=strides)
        right_view = np.lib.stride_tricks.as_strided(pad[w + 1 :], shape=(n, w), strides=strides)
        l_min = left_view[cand_arr].min(axis=1)
        r_min = right_view[cand_arr].min(axis=1)
        prom = y[cand_arr] - np.maximum(l_min, r_min)
        cand_arr = cand_arr[prom >= min_prominence]
        if len(cand_arr) == 0:
            return np.empty(0, dtype=np.int64)

    if min_distance <= 1 or len(cand_arr) <= 1:
        return cand_arr

    order = np.argsort(-y[cand_arr], kind="stable")
    kept = []
    suppressed = np.zeros(len(cand_arr), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        p = cand_arr[idx]
        kept.append(p)
        suppressed |= np.abs(cand_arr - p) < min_distance
    return np.sort(np.array(kept, dtype=np.int64))


def _refine_pillar_positions(peaks: np.ndarray, cuts: np.ndarray, half_pw: int) -> np.ndarray:
    """Move each detected pillar to the centroid of its unsmoothed cut stack.

    ``pillar_score`` is box-filtered over ``pillar_window`` bp, so a cut stack one boundary
    wide becomes a flat plateau and the peak sits on whichever plateau position background
    noise favours, up to ``half_pw`` bp off. The stack lies within ``half_pw`` of the detected
    peak, so the centroid of raw cuts over ``half_pw + 1`` either side recovers it. Unlike
    the argmax it stays centred on stacks jittered over several boundaries.
    """
    if len(peaks) == 0:
        return peaks
    n = len(cuts)
    cuts = np.asarray(cuts, dtype=np.float64)
    out = peaks.copy()
    for i, p in enumerate(peaks):
        lo, hi = max(0, int(p) - half_pw - 1), min(n, int(p) + half_pw + 2)
        seg = cuts[lo:hi]
        tot = float(seg.sum())
        if tot > 0.0:
            out[i] = int(np.floor(np.dot(np.arange(lo, hi), seg) / tot + 0.5))
    return np.unique(out)


def _relative_expected(expected_cuts: Any, W: int) -> np.ndarray:
    """Validate ``expected_cuts`` and rescale it to mean 1 (clamped at 0.1), so any scale works."""
    exp_arr = np.asarray(expected_cuts, dtype=np.float64)
    if exp_arr.shape != (W,):
        raise ValueError(f"`expected_cuts` must have one value per bp of the window ({W}), got shape {exp_arr.shape}.")
    if not np.all(np.isfinite(exp_arr)) or np.any(exp_arr < 0.0) or not np.any(exp_arr > 0.0):
        raise ValueError("`expected_cuts` must be finite, non-negative and not all zero.")
    return np.maximum(0.1, exp_arr / float(np.mean(exp_arr)))


def _compute_cov_weight(cov_tf: np.ndarray) -> np.ndarray:
    """Smooth local coverage weighting in [0, 1] using a 301-bp local peak envelope."""
    if len(cov_tf) == 0:
        return np.empty(0, dtype=np.float64)
    cov_f64 = cov_tf.astype(np.float64)
    local_cov_max = _max_filter_1d(cov_f64, 301)
    return cov_f64 / (cov_f64 + 0.10 * local_cov_max + 10.0)


_MOTIF_COLUMNS = [
    "chrom",
    "start",
    "end",
    "center",
    "width",
    "name",
    "score",
    "strand",
    "protection_score",
    "raw_log2_protection",
    "core_pillar_score",
    "left_pillar_score",
    "right_pillar_score",
    "tf_coverage",
    "composite_score",
]

_CALLERS = ("profile", "pairs")


def _check_caller(caller: str) -> str:
    if caller not in _CALLERS:
        raise ValueError(f"caller must be 'profile' or 'pairs', got {caller!r}.")
    return caller


def _default_min_protection(caller: str, value: float | None) -> float:
    """``min_protection_score`` default per caller: profile peaks 1.0, pillar pairs 0.75."""
    if value is not None:
        return float(value)
    return 1.0 if caller == "profile" else 0.75


def _protection_profile(
    pillar_score: np.ndarray,
    cov_weight: np.ndarray,
    widths: np.ndarray,
    motif_width_range: tuple[int, int],
    flank_pad: int,
    flank_mode: str,
    pseudocount: float,
    expected_cuts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window protection at every boundary and every footprint width.

    For width ``w`` centred on boundary ``x`` the footprint is ``[x - w // 2, x + w - w // 2)``;
    its core (``w`` minus 2 bp at each edge) is compared with the strongest ``pillar_score``
    within ``flank_pad`` of each edge. With ``expected_cuts`` the pillar score is divided by the
    rescaled sequence-expected cut rate first. Returns ``(protection_2d, raw_protection_2d,
    protection_1d, raw_protection_1d, best_widths)``; the 1-D profile takes the best width
    within ``motif_width_range`` where any width is positive, the mean over widths elsewhere.
    """
    W = len(pillar_score)
    ps = np.asarray(pillar_score, dtype=np.float64)
    if expected_cuts is not None:
        ps = ps / _relative_expected(expected_cuts, W)
    prot_2d_scale = np.zeros((len(widths), W), dtype=np.float64)
    prot_2d_raw = np.zeros((len(widths), W), dtype=np.float64)
    ps_pad = np.concatenate([[0.0], np.cumsum(ps)])
    xs = np.arange(W, dtype=np.int64)
    p_max_flank = _max_filter_1d(ps, 2 * int(flank_pad) + 1)
    pc = float(pseudocount)

    for idx_w, w in enumerate(widths):
        half_l = int(w) // 2
        half_r = int(w) - half_l
        in_l = max(1, half_l - 2)
        in_r = max(1, half_r - 2)
        c_lo = np.clip(xs - in_l, 0, W)
        c_hi = np.clip(xs + in_r + 1, 0, W)
        core_mean = (ps_pad[c_hi] - ps_pad[c_lo]) / np.maximum(1, c_hi - c_lo)
        core_max = _max_filter_1d(ps, in_l + in_r + 1)
        core_eff = 0.5 * core_mean + 0.5 * core_max

        l_idx = np.clip(xs - half_l, 0, max(0, W - 1))
        r_idx = np.clip(xs + half_r, 0, max(0, W - 1))
        flank_l = p_max_flank[l_idx] if W > 0 else np.zeros(0)
        flank_r = p_max_flank[r_idx] if W > 0 else np.zeros(0)

        f_min = np.minimum(flank_l, flank_r)
        f_max = np.maximum(flank_l, flank_r)
        if flank_mode == "both":
            flank_paired = (np.maximum(0.0, f_min) ** 0.65) * (np.maximum(0.0, f_max) ** 0.35)
        else:
            flank_paired = 0.65 * f_max + 0.35 * f_min

        raw_log2 = np.log2((flank_paired + pc) / (core_eff + pc))
        prot_2d_raw[idx_w] = raw_log2
        prot_2d_scale[idx_w] = raw_log2 * cov_weight

    w_mask = (widths >= motif_width_range[0]) & (widths <= motif_width_range[1])
    if not np.any(w_mask):
        w_mask = np.ones(len(widths), dtype=bool)
    sub_widths = widths[w_mask]
    sub_scale = prot_2d_scale[w_mask]
    sub_raw = prot_2d_raw[w_mask]

    if W > 0:
        best_w_idx = np.argmax(sub_scale, axis=0)
        max_scale = sub_scale[best_w_idx, np.arange(W)]
        mean_scale = sub_scale.mean(axis=0)
        prot_1d = np.where(max_scale > 0.0, max_scale, mean_scale)
        prot_1d_raw = np.where(max_scale > 0.0, sub_raw[best_w_idx, np.arange(W)], sub_raw.mean(axis=0))
        best_widths = sub_widths[best_w_idx]
    else:
        prot_1d = np.empty(0, dtype=np.float64)
        prot_1d_raw = np.empty(0, dtype=np.float64)
        best_widths = np.empty(0, dtype=np.int32)
    return prot_2d_scale, prot_2d_raw, prot_1d, prot_1d_raw, best_widths


def _call_profile_peaks(
    *,
    chrom: str,
    L: int,
    protection: np.ndarray,
    raw_protection: np.ndarray,
    best_widths: np.ndarray,
    pillar_score: np.ndarray,
    cov_tf: np.ndarray,
    flank_pad: int,
    min_protection_score: float,
    min_raw_protection: float,
    min_motif_distance: int,
    expected_cuts: np.ndarray | None = None,
) -> pd.DataFrame:
    """Call footprints as peaks of the 1-bp protection profile.

    A peak at boundary ``x`` with best width ``w`` spans ``[x - w // 2, x + w - w // 2)``; each
    edge is then moved to the tallest ``pillar_score`` within ``flank_pad`` bp, the flank the
    profile scored. Peaks must reach ``min_protection_score`` (coverage-weighted) and
    ``min_raw_protection`` (unweighted) and lie at least ``min_motif_distance`` bp apart (the
    higher peak wins). ``center`` is the peak itself.
    """
    W = len(protection)
    if W < 3:
        return pd.DataFrame(columns=_MOTIF_COLUMNS)
    peaks = _find_peaks_1d(
        np.asarray(protection, dtype=np.float64),
        min_height=float(min_protection_score),
        min_distance=max(1, int(min_motif_distance)),
    )
    peaks = peaks[raw_protection[peaks] >= float(min_raw_protection)]
    if len(peaks) == 0:
        return pd.DataFrame(columns=_MOTIF_COLUMNS)
    ps = np.asarray(pillar_score, dtype=np.float64)
    eff = ps if expected_cuts is None else ps / _relative_expected(expected_cuts, W)
    fp = int(flank_pad)
    w0 = best_widths[peaks].astype(np.int64)

    def snap(b: np.ndarray, lo_lim: np.ndarray, hi_lim: np.ndarray) -> np.ndarray:
        out = b.copy()
        for i, x in enumerate(b):
            a, z = max(int(lo_lim[i]), int(x) - fp, 0), min(int(hi_lim[i]), int(x) + fp, W - 1)
            if z >= a:
                out[i] = a + int(np.argmax(eff[a : z + 1]))
        return out

    lo = snap(peaks - w0 // 2, np.zeros_like(peaks), peaks - 2)
    hi = snap(peaks + (w0 - w0 // 2), peaks + 2, np.full_like(peaks, W - 1))
    w = hi - lo
    fl_l = eff[np.clip(lo, 0, W - 1)]
    fl_r = eff[np.clip(hi, 0, W - 1)]
    prot = protection[peaks].astype(np.float64)
    cv = cov_tf[peaks].astype(np.float64)
    comp = prot * np.log2(1.0 + 0.5 * (fl_l + fl_r)) * np.log2(1.0 + cv)
    centers = L + peaks
    return pd.DataFrame(
        {
            "chrom": chrom,
            "start": (L + lo).astype(np.int64),
            "end": (L + hi).astype(np.int64),
            "center": centers.astype(np.int64),
            "width": w,
            "name": [f"motif_{chrom}_{c}" for c in centers],
            "score": prot,
            "strand": ".",
            "protection_score": prot,
            "raw_log2_protection": raw_protection[peaks].astype(np.float64),
            "core_pillar_score": eff[peaks],
            "left_pillar_score": fl_l,
            "right_pillar_score": fl_r,
            "tf_coverage": cv,
            "composite_score": comp,
        }
    )


def _call_intervals_from_arrays(
    *,
    chrom: str,
    L: int,
    W: int,
    pillar_score: np.ndarray,
    n_len_bins: np.ndarray,
    left_cuts: np.ndarray,
    right_cuts: np.ndarray,
    cov_tf: np.ndarray,
    cov_weight: np.ndarray | None = None,
    motif_width_range: tuple[int, int] = (10, 24),
    flank_mode: str = "both",
    pseudocount: float = 2.0,
    pillar_window: int = 3,
    min_pillar_score: float = 8.0,
    min_len_bins: int = 4,
    min_pillar_prominence: float = 2.0,
    min_pillar_distance: int = 5,
    min_protection_score: float = 0.75,
    min_raw_protection: float = 0.20,
    min_motif_distance: int = 12,
    expected_cuts: np.ndarray | None = None,
    call_pairs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Call discrete cut pillars first, then call candidate motifs as pairs ``(l, r)``
    of called pillars with ``r - l`` in ``motif_width_range`` scored by core depletion
    between them. Interval is ``[l, r)`` by construction.
    """
    p_peaks = _find_peaks_1d(
        pillar_score,
        min_height=min_pillar_score,
        min_distance=max(1, int(min_pillar_distance)),
        min_prominence=min_pillar_prominence,
    )
    if len(p_peaks) > 0:
        p_peaks = p_peaks[n_len_bins[p_peaks] >= min_len_bins]

    half_pw = max(1, int(pillar_window) // 2)
    p_peaks = _refine_pillar_positions(p_peaks, left_cuts + right_cuts, half_pw)
    p_rows = []
    for p in p_peaks:
        w_lo = max(0, int(p) - half_pw)
        w_hi = min(W, int(p) + half_pw + 1)
        l_sum = float(left_cuts[w_lo:w_hi].sum())
        r_sum = float(right_cuts[w_lo:w_hi].sum())
        tot = l_sum + r_sum
        l_frac = l_sum / tot if tot > 0 else 0.5
        p_rows.append(
            {
                "chrom": chrom,
                "start": int(L + p - half_pw),
                "end": int(L + p + half_pw + 1),
                "pos": int(L + p),
                "name": f"pillar_{chrom}_{L + p}",
                "score": float(pillar_score[p]),
                "strand": "+" if l_frac >= 0.5 else "-",
                "pillar_score": float(pillar_score[p]),
                "n_len_bins": int(n_len_bins[p]),
                "cuts": tot,
                "left_cuts": l_sum,
                "right_cuts": r_sum,
                "left_frac": l_frac,
                "tf_coverage": float(cov_tf[p]),
            }
        )
    if p_rows:
        pillars_df = pd.DataFrame(p_rows)
    else:
        pillars_df = pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "pos",
                "name",
                "score",
                "strand",
                "pillar_score",
                "n_len_bins",
                "cuts",
                "left_cuts",
                "right_cuts",
                "left_frac",
                "tf_coverage",
            ]
        )

    # Pair-based motif calling: score every pair (l_p, r_p) of called pillars with
    # width = r_p - l_p in motif_width_range (inclusive).
    if cov_weight is None:
        cov_weight = _compute_cov_weight(cov_tf)

    eff_ps = pillar_score
    if expected_cuts is not None:
        eff_ps = pillar_score / _relative_expected(expected_cuts, W)

    ps_pad = np.concatenate([[0.0], np.cumsum(eff_ps)])
    min_w = max(4, int(motif_width_range[0]))
    max_w = int(motif_width_range[1])
    pc = float(pseudocount)
    edge_trim = max(3, half_pw + 1)

    n_p = len(p_peaks) if call_pairs else 0
    candidates = []
    for i in range(n_p):
        lp = int(p_peaks[i])
        fl_l = float(eff_ps[lp])
        for j in range(i + 1, n_p):
            rp = int(p_peaks[j])
            w = rp - lp
            if w < min_w:
                continue
            if w > max_w:
                break
            fl_r = float(eff_ps[rp])
            f_min = min(fl_l, fl_r)
            f_max = max(fl_l, fl_r)

            # Reject composite pairs that span over an internal called pillar at least 65% as tall as f_min
            if j > i + 1:
                inner_max_pil = max(float(eff_ps[int(p_peaks[k])]) for k in range(i + 1, j))
                if inner_max_pil >= 0.65 * f_min:
                    continue

            # Core interior strictly inside the two pillars (excluding 3-bp pillar_window spillover)
            c_lo = min(rp - 1, lp + edge_trim)
            c_hi = max(c_lo + 1, rp - edge_trim + 1)
            c_mean = float(ps_pad[c_hi] - ps_pad[c_lo]) / float(c_hi - c_lo)
            c_max = float(np.max(eff_ps[c_lo:c_hi]))
            c_eff = 0.5 * c_mean + 0.5 * c_max

            # Both pillar boundaries must exceed the core interior
            if fl_l <= c_mean + 1.0 or fl_r <= c_mean + 1.0:
                continue
            if flank_mode == "both":
                f_paired = (max(0.0, f_min) ** 0.65) * (max(0.0, f_max) ** 0.35)
            else:
                # One called pillar + minimum signal on the other called pillar
                f_paired = (max(0.0, f_min) ** 0.45) * (max(0.0, f_max) ** 0.55)

            raw_prot = float(np.log2((f_paired + pc) / (c_eff + pc)))
            if raw_prot < min_raw_protection:
                continue

            mid = (lp + rp) // 2
            prot = raw_prot * float(cov_weight[mid])
            if prot >= min_protection_score:
                cv = float(cov_tf[mid])
                comp = prot * float(np.log2(1.0 + 0.5 * (fl_l + fl_r))) * float(np.log2(1.0 + cv))
                candidates.append(
                    (
                        lp,
                        rp,
                        mid,
                        w,
                        prot,
                        raw_prot,
                        float(pillar_score[mid]),
                        float(pillar_score[lp]),
                        float(pillar_score[rp]),
                        cv,
                        comp,
                    )
                )

    # Non-maximum suppression: keep highest-scoring pair among overlapping intervals [lp, rp)
    # or centers closer than min_motif_distance
    candidates.sort(key=lambda x: (-x[4], x[0]))
    kept = []
    min_dist = max(1, int(min_motif_distance))
    for cand in candidates:
        lp, rp, mid = cand[0], cand[1], cand[2]
        if any((lp < kr and rp > kl) or abs(mid - km) < min_dist for kl, kr, km, *_ in kept):
            continue
        kept.append(cand)
    kept.sort(key=lambda x: x[0])

    m_rows = []
    for lp, rp, mid, w, prot, raw_prot, core_ps, fl_l, fl_r, cv, comp in kept:
        m_start = int(L + lp)
        m_end = int(L + rp)
        m_center = int(L + mid)
        m_rows.append(
            {
                "chrom": chrom,
                "start": m_start,
                "end": m_end,
                "center": m_center,
                "width": int(w),
                "name": f"motif_{chrom}_{m_center}",
                "score": prot,
                "strand": ".",
                "protection_score": prot,
                "raw_log2_protection": raw_prot,
                "core_pillar_score": core_ps,
                "left_pillar_score": fl_l,
                "right_pillar_score": fl_r,
                "tf_coverage": cv,
                "composite_score": comp,
            }
        )

    if m_rows:
        motifs_df = pd.DataFrame(m_rows)
    else:
        motifs_df = pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "center",
                "width",
                "name",
                "score",
                "strand",
                "protection_score",
                "raw_log2_protection",
                "core_pillar_score",
                "left_pillar_score",
                "right_pillar_score",
                "tf_coverage",
                "composite_score",
            ]
        )
    return pillars_df, motifs_df


def _draw_protection(
    ax: Any, xs: np.ndarray, profile: np.ndarray, motifs: pd.DataFrame, min_prot: float, caller: str = "pairs"
) -> None:
    """Grey 1-bp profile, called motifs as bars at their score, and the threshold."""
    lab = "1-bp profile (calls are its peaks)" if caller == "profile" else "1-bp profile (exploratory, not the caller)"
    ax.fill_between(xs, 0.0, profile, color="0.78", lw=0, step="mid", label=lab)
    ax.axhline(0.0, color="0.3", lw=0.6)
    ax.axhline(min_prot, color="#d62828", ls="--", lw=0.9, label=f"min_protection_score ({min_prot:.2f})")
    for i, (_, m) in enumerate(motifs.iterrows()):
        s, e, sc = float(m["start"]), float(m["end"]), float(m["protection_score"])
        ax.axvspan(s, e, color="#ffd166", alpha=0.35, lw=0, zorder=0)
        ax.bar((s + e) / 2.0, sc, width=e - s, color="#1b4332", alpha=0.9, zorder=3,
               label=f"Called motif (n={len(motifs)})" if i == 0 else None)
    top = max([min_prot, *(float(v) for v in motifs["protection_score"])] + ([float(np.max(profile))] if len(profile) else []))
    bottom = float(np.min(profile)) if len(profile) else 0.0
    # headroom so the legend in the upper corner clears the tallest bar
    ax.set_ylim(min(bottom * 1.1, -0.1), top * 1.9)


# ---------------------------------------------------------------------------
# LocusPillarScan dataclass
# ---------------------------------------------------------------------------

@dataclass(repr=False)
class LocusPillarScan:
    """Continuous 1D/2D cleavage-pillar & footprint-protection profiles and pair-called intervals for one locus.

    All interval and track outputs use ``chrom, start, end`` coordinates (0-based, half-open)
    for direct compatibility with ``pygenomic_tracks``, bedGraph/BED export, and ``moodspy``.
    Called motifs correspond to intervals ``[l, r)`` between pairs of called cut pillars at
    genomic boundaries ``l`` and ``r``.

    ``protection_score``, ``raw_log2_protection``, ``protection_2d`` and ``best_widths`` are a
    sliding-window profile (core depletion against the strongest cuts ``width / 2 +- flank_pad``
    bp away, at every position and every ``footprint_widths``). With ``caller="profile"``
    (default) ``.motifs`` are the peaks of ``protection_score``, each spanning its best width.
    With ``caller="pairs"`` motifs are scored only between pairs of called pillars, so a
    profile peak need not be called and a call need not sit on a peak.

    The profile measures depletion of cuts, which also follows the enzyme's sequence
    preference: MNase cuts GC-rich DNA less, so part of a GC box's depletion is there whether
    or not it is bound. Pass ``expected_cuts`` and compare against a sequence-only baseline
    before reading a footprint as occupancy.
    """

    chrom: str
    start: int
    end: int
    pos: np.ndarray
    pillar_score: np.ndarray
    n_len_bins: np.ndarray
    protection_score: np.ndarray
    raw_log2_protection: np.ndarray
    tf_coverage: np.ndarray
    left_cuts: np.ndarray
    right_cuts: np.ndarray
    best_widths: np.ndarray
    footprint_widths: np.ndarray
    protection_2d: np.ndarray
    raw_protection_2d: np.ndarray
    len_edges: np.ndarray
    cut_matrix: np.ndarray
    log_cut_matrix: np.ndarray
    pillars: pd.DataFrame
    motifs: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_cuts: np.ndarray | None = None

    def __repr__(self) -> str:
        return (
            f"LocusPillarScan({self.chrom}:{self.start}-{self.end}, "
            f"n_pillars={len(self.pillars)}, n_motifs={len(self.motifs)})"
        )

    def call(
        self,
        *,
        min_protection_score: float | None = None,
        min_raw_protection: float | None = None,
        min_motif_distance: int | None = None,
        min_pillar_score: float | None = None,
        min_len_bins: int | None = None,
        min_pillar_prominence: float | None = None,
        min_pillar_distance: int = 5,
        expected_cuts: np.ndarray | None = None,
        caller: str | None = None,
    ) -> LocusPillarScan:
        """Re-call discrete ``.pillars`` and ``.motifs`` intervals with new thresholds
        without re-scanning the underlying fragments.

        Parameters
        ----------
        min_protection_score:
            Minimum coverage-weighted protection to call a footprint ``[l, r)``
            (defaults to the value used when creating this scan; 1.0 for ``caller="profile"``,
            0.75 for ``"pairs"``).
        min_raw_protection:
            Minimum unweighted ``log2((flank + pc) / (core + pc))`` depletion required
            between a pair of flanking pillars (default ``0.20``).
        min_motif_distance:
            Minimum center-to-center distance (bp) between called motifs (default ``12``).
        min_pillar_score:
            Minimum ``pillar_score`` to call a cut pillar (default ``8.0``).
        min_len_bins:
            Minimum number of distinct fragment-length bins with cleavage to call a cut pillar (default ``4``).
        min_pillar_prominence:
            Minimum local prominence of a cut-pillar peak (default ``2.0``).
        min_pillar_distance:
            Minimum distance (bp) between called cut pillars (default ``5``).
        expected_cuts:
            Per-bp sequence-expected cleavage rate of length ``end - start`` (any scale; it is
            rescaled to mean 1), e.g. from [`KmerBiasTable`][pyvplot.KmerBiasTable], dividing
            ``pillar_score`` before the profile and the calls are computed. Defaults to the
            array this scan was built with; a new array also recomputes the 1-bp and 2-D
            protection profiles.
        caller:
            ``"profile"`` (peaks of the 1-bp protection profile) or ``"pairs"`` (pillar pairs).
            Defaults to the caller this scan was built with.

        Returns
        -------
        LocusPillarScan
            A new ``LocusPillarScan`` sharing the same continuous 1D/2D arrays with
            updated ``.motifs``, ``.pillars``, and threshold metadata.
        """
        caller = _check_caller(str(self.metadata.get("caller", "pairs")) if caller is None else caller)
        if min_protection_score is not None:
            mp_score = float(min_protection_score)
        elif caller == str(self.metadata.get("caller", "pairs")) and "min_protection_score" in self.metadata:
            mp_score = float(self.metadata["min_protection_score"])
        else:
            mp_score = _default_min_protection(caller, None)
        mp_prom = float(
            self.metadata.get("min_raw_protection", 0.20)
            if min_raw_protection is None
            else min_raw_protection
        )
        mp_dist = int(
            self.metadata.get("min_motif_distance", 12)
            if min_motif_distance is None
            else min_motif_distance
        )
        pp_score = float(
            self.metadata.get("min_pillar_score", 8.0)
            if min_pillar_score is None
            else min_pillar_score
        )
        pp_bins = int(
            self.metadata.get("min_len_bins", 4)
            if min_len_bins is None
            else min_len_bins
        )
        pp_prom = float(
            self.metadata.get("min_pillar_prominence", 2.0)
            if min_pillar_prominence is None
            else min_pillar_prominence
        )
        pillar_window = int(self.metadata.get("pillar_window", 3))
        motif_width_range = tuple(self.metadata.get("motif_width_range", (10, 24)))
        flank_mode = str(self.metadata.get("flank_mode", "both"))
        pseudocount = float(self.metadata.get("pseudocount", 2.0))
        exp = self.expected_cuts if expected_cuts is None else np.asarray(expected_cuts, dtype=np.float64)

        pillars_df, motifs_df = _call_intervals_from_arrays(
            chrom=self.chrom,
            L=self.start,
            W=self.end - self.start,
            pillar_score=self.pillar_score,
            n_len_bins=self.n_len_bins,
            left_cuts=self.left_cuts,
            right_cuts=self.right_cuts,
            cov_tf=self.tf_coverage,
            motif_width_range=motif_width_range,
            flank_mode=flank_mode,
            pseudocount=pseudocount,
            pillar_window=pillar_window,
            min_pillar_score=pp_score,
            min_len_bins=pp_bins,
            min_pillar_prominence=pp_prom,
            min_pillar_distance=min_pillar_distance,
            min_protection_score=mp_score,
            min_raw_protection=mp_prom,
            min_motif_distance=mp_dist,
            expected_cuts=exp,
            call_pairs=caller == "pairs",
        )
        profile = {}
        if expected_cuts is not None:
            p2, p2r, p1, p1r, bw = _protection_profile(
                self.pillar_score,
                _compute_cov_weight(self.tf_coverage),
                self.footprint_widths,
                motif_width_range,
                int(self.metadata.get("flank_pad", 2)),
                flank_mode,
                pseudocount,
                exp,
            )
            profile = {
                "protection_2d": p2,
                "raw_protection_2d": p2r,
                "protection_score": p1,
                "raw_log2_protection": p1r,
                "best_widths": bw,
            }
        if caller == "profile":
            motifs_df = _call_profile_peaks(
                chrom=self.chrom,
                L=self.start,
                protection=profile.get("protection_score", self.protection_score),
                raw_protection=profile.get("raw_log2_protection", self.raw_log2_protection),
                best_widths=profile.get("best_widths", self.best_widths),
                pillar_score=self.pillar_score,
                cov_tf=self.tf_coverage,
                flank_pad=int(self.metadata.get("flank_pad", 2)),
                min_protection_score=mp_score,
                min_raw_protection=mp_prom,
                min_motif_distance=mp_dist,
                expected_cuts=exp,
            )
        new_meta = dict(self.metadata)
        new_meta.update(
            {
                "caller": caller,
                "min_protection_score": mp_score,
                "min_raw_protection": mp_prom,
                "min_motif_distance": mp_dist,
                "min_pillar_score": pp_score,
                "min_len_bins": pp_bins,
                "min_pillar_prominence": pp_prom,
            }
        )
        return replace(self, pillars=pillars_df, motifs=motifs_df, metadata=new_meta, expected_cuts=exp, **profile)

    def to_track_df(
        self,
        what: str = "protection_score",
        bin_size: int = 1,
        drop_zeros: bool = False,
    ) -> pd.DataFrame:
        """Export any continuous 1-bp profile or called intervals as a ``chrom, start, end`` DataFrame for ``pygenomic_tracks``."""
        if what == "pillars":
            return self.pillars.copy()
        if what == "motifs":
            return self.motifs.copy()

        track_map: dict[str, np.ndarray] = {
            "protection_score": self.protection_score,
            "raw_log2_protection": self.raw_log2_protection,
            "pillar_score": self.pillar_score,
            "tf_coverage": self.tf_coverage,
            "cuts": self.left_cuts + self.right_cuts,
            "left_cuts": self.left_cuts,
            "right_cuts": self.right_cuts,
            "n_len_bins": self.n_len_bins.astype(np.float64),
        }
        if what not in track_map:
            valid = [*track_map.keys(), "pillars", "motifs"]
            raise ValueError(f"Unknown track {what!r}; expected one of {valid}.")

        if bin_size <= 0:
            raise ValueError("`bin_size` must be a positive integer.")

        vals = np.asarray(track_map[what], dtype=np.float64)
        W = len(vals)
        if W == 0:
            return pd.DataFrame(columns=["chrom", "start", "end", "value", "score"])

        bs = int(bin_size)
        if bs == 1:
            starts = self.pos.astype(np.int64)
            ends = starts + 1
            out_vals = vals
        else:
            n_bins = int(np.ceil(W / bs))
            starts = self.start + bs * np.arange(n_bins, dtype=np.int64)
            ends = np.minimum(starts + bs, self.end)
            out_vals = np.array(
                [vals[i * bs : min((i + 1) * bs, W)].mean() for i in range(n_bins)],
                dtype=np.float64,
            )

        if drop_zeros:
            nz = out_vals != 0.0
            starts = starts[nz]
            ends = ends[nz]
            out_vals = out_vals[nz]

        return pd.DataFrame(
            {
                "chrom": np.repeat(self.chrom, len(starts)),
                "start": starts,
                "end": ends,
                "value": out_vals,
                "score": out_vals,
            }
        )

    def to_pygenomic_tracks(self, bin_size: int = 1, drop_zeros: bool = False) -> dict[str, pd.DataFrame]:
        """Return a dictionary of ``chrom, start, end`` DataFrames ready for ``pygenomic_tracks``."""
        return {
            "pillar_score": self.to_track_df("pillar_score", bin_size=bin_size, drop_zeros=drop_zeros),
            "protection_score": self.to_track_df("protection_score", bin_size=bin_size, drop_zeros=drop_zeros),
            "raw_log2_protection": self.to_track_df("raw_log2_protection", bin_size=bin_size, drop_zeros=drop_zeros),
            "tf_coverage": self.to_track_df("tf_coverage", bin_size=bin_size, drop_zeros=drop_zeros),
            "cuts": self.to_track_df("cuts", bin_size=bin_size, drop_zeros=drop_zeros),
            "pillars": self.pillars.copy(),
            "motifs": self.motifs.copy(),
        }

    def plot(
        self,
        ax: Any = None,
        what: str = "protection",
        show_motifs: bool = True,
        show_pillars: bool = True,
        store: FragmentStore | None = None,
        hits: pd.DataFrame | None = None,
        mode: str = "tracks",
        show_raw_protection: bool = True,
        figsize: tuple[float, float] = (12.5, 9.5),
    ) -> tuple[Any, Any]:
        """Plot into a single existing ``ax`` (for ``pygenomic_tracks`` / multi-track figures)
        or create a standalone multi-panel QC figure when ``ax is None``.

        Called motifs are drawn as bars spanning ``[start, end)`` at their
        ``protection_score``, with ``min_protection_score`` as the threshold line. The 1-bp
        ``protection_score`` profile is drawn in grey: with ``caller="profile"`` calls are its
        peaks, with ``caller="pairs"`` it is context only.

        ``mode="tracks"`` (default) returns four axes: fragment ends (or the length-resolved
        cut matrix without ``store``), protection, pillar score, and TF-size fragment coverage.
        ``mode="multiscale"`` adds the width-resolved ``protection_2d`` panel.
        """
        import matplotlib.pyplot as plt

        L, R = self.start, self.end
        xs = self.pos

        if ax is not None:
            if what not in {"protection", "pillars", "both"}:
                raise ValueError(f"what must be 'protection', 'pillars' or 'both', got {what!r}.")
            fig = ax.figure
            if what in {"protection", "both"}:
                _draw_protection(
                    ax,
                    xs,
                    self.protection_score,
                    self.motifs if show_motifs else self.motifs.iloc[:0],
                    float(self.metadata.get("min_protection_score", 0.75)), str(self.metadata.get("caller", "pairs")),
                )
                ax.set_ylabel("Protection")
            if what == "pillars":
                ax.fill_between(xs, 0, self.pillar_score, color="#3a86ff", alpha=0.30)
                ax.plot(xs, self.pillar_score, color="#1d3557", lw=1.2, label="Pillar score")
                ax.set_ylabel("Pillar score")
                if show_pillars and len(self.pillars) > 0:
                    ax.scatter(
                        self.pillars["pos"],
                        self.pillars["pillar_score"],
                        color="#e63946",
                        s=18,
                        zorder=5,
                        label="Cut pillar",
                    )
                if show_motifs and len(self.motifs) > 0:
                    for i, (_, m) in enumerate(self.motifs.iterrows()):
                        ax.axvspan(
                            float(m["start"]),
                            float(m["end"]),
                            color="#ffd166",
                            alpha=0.35,
                            zorder=0,
                            label="Called motif" if i == 0 else None,
                        )
            elif what == "both":
                ax_r = ax.twinx()
                ax_r.plot(xs, self.pillar_score, color="#3a86ff", lw=1.0, alpha=0.75, label="Pillar score")
                ax_r.set_ylabel("Pillar score", color="#1d3557", fontsize=8.5)
                if show_pillars and len(self.pillars) > 0:
                    ax_r.scatter(
                        self.pillars["pos"],
                        self.pillars["pillar_score"],
                        color="#e63946",
                        s=15,
                        zorder=5,
                    )
            ax.set_xlim(L, R)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            return fig, ax

        sub_hits = None
        if hits is not None and len(hits) > 0:
            sub_hits = hits[
                (hits["chrom"].astype(str) == self.chrom)
                & (hits["end"] > L)
                & (hits["start"] < R)
            ]

        if mode == "tracks":
            fig, axes = plt.subplots(
                4,
                1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [1.6, 1.1, 1.1, 0.5], "hspace": 0.08},
            )
            ax_top, ax_prot, ax_pillar, ax_cov = axes
            min_l, max_l = self.metadata.get("len_range", (20, 250))

            if store is not None and not store.is_stranded:
                lv = store.locus_vplot((self.chrom, L, R), feature="cuts", len_range=(min_l, max_l))
                pts = lv.points
                for end, color, label in (("left", "#e63946", "Fragment start"), ("right", "#1d3557", "Fragment end")):
                    sel = pts[pts["end"] == end]
                    ax_top.scatter(sel["pos"], sel["frag_len"], s=4, color=color, alpha=0.55, lw=0, label=label)
                ax_top.set_ylim(min_l, max_l)
                ax_top.set_ylabel("Fragment length (bp)")
            else:
                ax_top.imshow(
                    self.log_cut_matrix,
                    aspect="auto",
                    origin="lower",
                    extent=[L, R, float(self.len_edges[0]), float(self.len_edges[-1])],
                    cmap="magma",
                    interpolation="nearest",
                )
                ax_top.set_ylabel("Fragment length (bp)\ncuts, log2(1 + n)")
            for i, (_, m) in enumerate(self.motifs.iterrows()):
                ax_top.axvspan(
                    float(m["start"]),
                    float(m["end"]),
                    color="#ffd166",
                    alpha=0.35,
                    lw=0,
                    label=f"Called motif (n={len(self.motifs)})" if i == 0 else None,
                )
            if sub_hits is not None:
                for i, (_, h) in enumerate(sub_hits.iterrows()):
                    ax_top.axvspan(
                        float(h["start"]),
                        float(h["end"]),
                        ymin=0.93,
                        ymax=1.0,
                        color="#ef476f",
                        label=f"Reference hit (n={len(sub_hits)})" if i == 0 else None,
                    )
            if ax_top.get_legend_handles_labels()[0]:
                ax_top.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
            ax_top.set_title(f"{self.chrom}:{L:,}-{R:,}", fontsize=11, loc="left")

            _draw_protection(
                ax_prot, xs, self.protection_score, self.motifs, float(self.metadata.get("min_protection_score", 0.75)), str(self.metadata.get("caller", "pairs"))
            )
            ax_prot.set_ylabel("Protection")
            ax_prot.legend(loc="upper right", fontsize=8, ncol=1, framealpha=0.9)

            ax_pillar.fill_between(xs, 0, self.pillar_score, color="#3a86ff", alpha=0.30, lw=0)
            ax_pillar.plot(xs, self.pillar_score, color="#1d3557", lw=1.0, label="Pillar score")
            if show_pillars and len(self.pillars) > 0:
                ax_pillar.scatter(
                    self.pillars["pos"],
                    self.pillars["pillar_score"],
                    color="#e63946",
                    s=14,
                    zorder=5,
                    label=f"Called pillar (n={len(self.pillars)})",
                )
            ax_pillar.set_ylabel("Pillar score")
            ax_pillar.legend(loc="upper left", fontsize=8, framealpha=0.9)

            ax_cov.fill_between(xs, 0, self.tf_coverage, color="0.7", lw=0, step="mid")
            ax_cov.set_ylabel("TF-size\ncoverage", fontsize=8.5)
            ax_cov.set_xlabel(f"{self.chrom} (cut boundary, 0-based)")
            ax_cov.set_xlim(L, R)
            return fig, axes

        # mode == "multiscale"
        fig, axes = plt.subplots(
            4,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1.3, 1.3, 1.35, 1.1], "hspace": 0.14},
        )
        ax0 = axes[0]
        im0 = ax0.imshow(
            self.log_cut_matrix,
            aspect="auto",
            origin="lower",
            extent=[L, R, float(self.len_edges[0]), float(self.len_edges[-1])],
            cmap="magma",
            interpolation="nearest",
        )
        if sub_hits is not None:
            for _, h in sub_hits.iterrows():
                ax0.axvspan(float(h["start"]), float(h["end"]), color="#00e5ff", alpha=0.18, lw=0)
        ax0.set_ylabel("Frag len (bp)\n[Cut pillars]")
        ax0.set_title(
            f"{self.chrom}:{L:,}-{R:,} — 2D Cleavage Pillars & Multi-Scale Protection Score",
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        cb0 = fig.colorbar(im0, ax=ax0, pad=0.01, aspect=15)
        cb0.set_label("log2(1+cuts)", fontsize=8)

        ax1 = axes[1]
        if len(self.footprint_widths) > 1:
            step = float(self.footprint_widths[1] - self.footprint_widths[0])
        else:
            step = 1.0
        w_lo = float(self.footprint_widths[0]) - 0.5 * step
        w_hi = float(self.footprint_widths[-1]) + 0.5 * step
        im1 = ax1.imshow(
            self.protection_2d,
            aspect="auto",
            origin="lower",
            extent=[L, R, w_lo, w_hi],
            cmap="viridis",
            interpolation="bilinear",
        )
        if len(self.motifs) > 0:
            ax1.scatter(
                self.motifs["center"],
                np.clip(self.motifs["width"], w_lo + 0.5, w_hi - 0.5),
                color="white",
                edgecolor="black",
                s=45,
                zorder=5,
                label=f"Called motif (n={len(self.motifs)})",
            )
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.85)
        if sub_hits is not None:
            for _, h in sub_hits.iterrows():
                hw = float(h["end"] - h["start"])
                ax1.plot([float(h["start"]), float(h["end"])], [hw, hw], color="#ff4d4d", lw=2.2, zorder=6)
        ax1.set_ylabel("Width W (bp)\n[2D Protection]")
        cb1 = fig.colorbar(im1, ax=ax1, pad=0.01, aspect=15)
        cb1.set_label("Protection", fontsize=8)

        ax2 = axes[2]
        if show_raw_protection:
            ax2.plot(xs, self.raw_log2_protection, color="0.6", lw=0.8, ls=":", label="1-bp profile, unweighted")
        _draw_protection(ax2, xs, self.protection_score, self.motifs, float(self.metadata.get("min_protection_score", 0.75)), str(self.metadata.get("caller", "pairs")))
        if sub_hits is not None:
            for i, (_, h) in enumerate(sub_hits.iterrows()):
                ax2.axvspan(
                    float(h["start"]),
                    float(h["end"]),
                    ymin=0.88,
                    ymax=1.0,
                    color="#ef476f",
                    alpha=0.88,
                    label=f"Reference hit (n={len(sub_hits)})" if i == 0 else None,
                )
        ax2.set_ylabel("Protection\n(1-bp track)")
        ax2.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
        cb_d1 = fig.colorbar(im1, ax=ax2, pad=0.01, aspect=15)
        cb_d1.ax.set_visible(False)

        ax3 = axes[3]
        ax3.fill_between(xs, 0, self.pillar_score, color="#3a86ff", alpha=0.30, label="Multi-length pillar score")
        ax3.plot(xs, self.pillar_score, color="#1d3557", lw=1.2)
        if len(self.pillars) > 0:
            ax3.scatter(
                self.pillars["pos"],
                self.pillars["pillar_score"],
                color="#e63946",
                s=22,
                zorder=5,
                label=f"Called cut pillars (n={len(self.pillars)})",
            )
        ax3.set_ylabel("Pillar score")
        ax3.set_xlabel(f"{self.chrom} genomic coordinate (bp, 0-based)")
        ax3.legend(loc="upper left", fontsize=8, framealpha=0.9)
        cb_d2 = fig.colorbar(im1, ax=ax3, pad=0.01, aspect=15)
        cb_d2.ax.set_visible(False)
        ax3.set_xlim(L, R)

        return fig, axes


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def scan_locus_pillars(
    store: FragmentStore,
    region: Any,
    *,
    len_range: tuple[int, int] = (20, 250),
    len_bin: int = 10,
    tf_len_range: tuple[int, int] = (20, 120),
    pillar_window: int = 3,
    footprint_widths: Sequence[int] | np.ndarray | None = None,
    motif_width_range: tuple[int, int] = (10, 24),
    flank_pad: int = 2,
    flank_mode: str = "both",
    pseudocount: float = 2.0,
    min_pillar_score: float = 8.0,
    min_len_bins: int = 4,
    min_pillar_prominence: float = 2.0,
    min_protection_score: float | None = None,
    min_raw_protection: float = 0.20,
    min_motif_distance: int = 12,
    expected_cuts: np.ndarray | Callable[[str, int, int], np.ndarray] | None = None,
    caller: str = "profile",
) -> LocusPillarScan:
    """Compute continuous 1D/2D cleavage-pillar and footprint-protection profiles and call
    cut pillars and protected footprints ``[l, r)`` in a genomic window.

    Pillars are detected on ``pillar_score`` and placed on their unsmoothed cut stacks.

    ``caller="profile"`` (default) calls footprints as peaks of the 1-bp ``protection_score``
    profile, each spanning its best width (from ``footprint_widths`` within
    ``motif_width_range``; snapping each edge to its flank pillar can change the width by up
    to ``2 * flank_pad``), so calls always sit on the profile. ``caller="pairs"`` is the
    original pillar-pair caller: motifs are scored only between pairs of called pillars
    ``r - l`` apart, with ``r - l`` in ``motif_width_range`` (inclusive). For both,
    ``min_raw_protection`` is the minimum unweighted ``log2((flank + pseudocount) / (core +
    pseudocount))`` and ``min_protection_score`` the minimum after TF-coverage weighting
    (default 1.0 for ``"profile"``, 0.75 for ``"pairs"``). On KLF5 ChEC-seq, scored against
    FiNeMo hits of a sequence model, profile peaks gave higher precision and recall than
    pillar pairs at every call density (``benchmarks/footprints`` in the repository).

    For assays with sequence-dependent endonuclease bias (MNase prefers AT-rich DNA, so a
    GC-rich core between AT-rich flanks can look protected), pass per-bp ``expected_cuts``
    of length ``end - start`` (any scale), or a function ``(chrom, start, end) -> array`` such
    as the one [`pillar_expected_cuts`][pyvplot.pillar_expected_cuts] builds from a
    [`KmerBiasTable`][pyvplot.KmerBiasTable]; it divides ``pillar_score`` before both the
    profile and the calls. Fit the table on DNA the factor does not bind: cuts away from peaks
    can still carry weak binding (on KLF5 ChEC-seq, off-peak KLF-like 9-mers were twice as
    depleted as other 9-mers of the same GC), which a table fitted there absorbs. Calibrate
    ``min_protection_score`` against a free-MNase / naked-DNA control store where one exists.
    """
    if store.is_stranded:
        raise ValueError("scan_locus_pillars requires paired-end fragment intervals; store is stranded.")
    if flank_mode not in {"both", "either"}:
        raise ValueError(f"flank_mode must be 'both' or 'either', got {flank_mode!r}.")
    _check_caller(caller)
    min_prot = _default_min_protection(caller, min_protection_score)

    chrom, L, R, _ = _parse_region(region, "region")
    min_len, max_len = int(len_range[0]), int(len_range[1])
    if max_len <= min_len or len_bin <= 0:
        raise ValueError("`len_range` must have max_len > min_len and `len_bin` > 0.")

    if footprint_widths is None:
        widths = np.arange(8, 36, 1, dtype=np.int32)
    else:
        widths = np.asarray(list(footprint_widths), dtype=np.int32)
        if len(widths) == 0 or np.any(widths < 4):
            raise ValueError("`footprint_widths` must contain integers >= 4.")

    W = R - L
    if callable(expected_cuts):
        expected_cuts = expected_cuts(chrom, int(L), int(R))
    exp = None if expected_cuts is None else np.asarray(expected_cuts, dtype=np.float64)
    if exp is not None:
        _relative_expected(exp, W)
    if chrom in store.starts and len(store.starts[chrom]) > 0:
        s_all = store.starts[chrom]
        e_all = store.ends[chrom]
    else:
        s_all = np.empty(0, dtype=np.int32)
        e_all = np.empty(0, dtype=np.int32)

    cut_mat, left_cuts, right_cuts, cov_tf = _accumulate_region(
        s_all,
        e_all,
        L,
        R,
        min_len,
        max_len,
        int(len_bin),
        int(tf_len_range[0]),
        int(tf_len_range[1]),
    )

    win_mat = _box_filter_2d_cols(cut_mat, int(pillar_window))
    log_mat = np.log2(1.0 + win_mat)
    pillar_score = log_mat.sum(axis=0)
    n_len_bins = (win_mat >= 1.0).sum(axis=0).astype(np.int32)

    cov_weight = _compute_cov_weight(cov_tf)
    pc = float(pseudocount)
    prot_2d_scale, prot_2d_raw, prot_1d, prot_1d_raw, best_widths = _protection_profile(
        pillar_score, cov_weight, widths, tuple(motif_width_range), int(flank_pad), flank_mode, pc, exp
    )

    pillars_df, motifs_df = _call_intervals_from_arrays(
        chrom=chrom,
        L=L,
        W=W,
        pillar_score=pillar_score,
        n_len_bins=n_len_bins,
        left_cuts=left_cuts,
        right_cuts=right_cuts,
        cov_tf=cov_tf,
        cov_weight=cov_weight,
        motif_width_range=tuple(motif_width_range),
        flank_mode=flank_mode,
        pseudocount=pc,
        pillar_window=int(pillar_window),
        min_pillar_score=float(min_pillar_score),
        min_len_bins=int(min_len_bins),
        min_pillar_prominence=float(min_pillar_prominence),
        min_pillar_distance=5,
        min_protection_score=min_prot,
        min_raw_protection=float(min_raw_protection),
        min_motif_distance=int(min_motif_distance),
        expected_cuts=exp,
        call_pairs=caller == "pairs",
    )
    if caller == "profile":
        motifs_df = _call_profile_peaks(
            chrom=chrom,
            L=L,
            protection=prot_1d,
            raw_protection=prot_1d_raw,
            best_widths=best_widths,
            pillar_score=pillar_score,
            cov_tf=cov_tf,
            flank_pad=int(flank_pad),
            min_protection_score=min_prot,
            min_raw_protection=float(min_raw_protection),
            min_motif_distance=int(min_motif_distance),
            expected_cuts=exp,
        )

    n_bins = cut_mat.shape[0]
    len_edges = min_len + int(len_bin) * np.arange(n_bins + 1, dtype=np.int64)
    return LocusPillarScan(
        chrom=chrom,
        start=L,
        end=R,
        pos=np.arange(L, R, dtype=np.int64),
        pillar_score=pillar_score,
        n_len_bins=n_len_bins,
        protection_score=prot_1d,
        raw_log2_protection=prot_1d_raw,
        tf_coverage=cov_tf.astype(np.float64),
        left_cuts=left_cuts.astype(np.float64),
        right_cuts=right_cuts.astype(np.float64),
        best_widths=best_widths,
        footprint_widths=widths,
        protection_2d=prot_2d_scale,
        raw_protection_2d=prot_2d_raw,
        len_edges=len_edges,
        cut_matrix=cut_mat,
        log_cut_matrix=log_mat,
        pillars=pillars_df,
        motifs=motifs_df,
        metadata={
            "len_range": (min_len, max_len),
            "len_bin": int(len_bin),
            "tf_len_range": tuple(tf_len_range),
            "pillar_window": int(pillar_window),
            "motif_width_range": tuple(motif_width_range),
            "flank_pad": int(flank_pad),
            "flank_mode": flank_mode,
            "pseudocount": pc,
            "min_pillar_score": float(min_pillar_score),
            "min_len_bins": int(min_len_bins),
            "min_pillar_prominence": float(min_pillar_prominence),
            "min_protection_score": min_prot,
            "min_raw_protection": float(min_raw_protection),
            "min_motif_distance": int(min_motif_distance),
            "caller": caller,
        },
        expected_cuts=exp,
    )


def pillar_expected_cuts(table: KmerBiasTable, sequences: Any) -> Callable[[str, int, int], np.ndarray]:
    """Sequence-expected cut rate at every boundary of a window, for ``expected_cuts=``.

    Returns ``f(chrom, start, end)`` giving, for each boundary ``b`` in ``[start, end)``, the
    k-mer propensity of a fragment starting at ``b`` (forward strand, cut before base ``b``)
    plus that of a fragment ending at ``b`` (reverse strand, cut after base ``b - 1``), the
    same two frames [`KmerBiasTable.from_cuts`][pyvplot.KmerBiasTable.from_cuts] estimates.
    Pass ``f`` to ``scan_genome_pillars(expected_cuts=f)`` or ``scan_locus_pillars(expected_cuts=f)``.

    Parameters
    ----------
    table:
        K-mer cleavage bias table, ideally estimated with ``KmerBiasTable.from_cuts`` on a
        naked-DNA or free-MNase control digested with the same enzyme.
    sequences:
        Genome: FASTA path (plain or ``.gz``), ``pysam.FastaFile``, ``pyfaidx.Fasta`` or
        ``{chrom: str}`` dict. Bases outside the chromosome, ``N`` and missing chromosomes
        score the neutral weight ``1.0``.
    """
    from pyvplot.bias import _fetch_subseq, _open_genome

    genome = _open_genome(sequences)
    cache: dict[str, str | None] = {}
    k, off = int(table.k), int(table.cut_offset)
    # Boundary b: the forward k-mer spans [b - off, b - off + k), the reverse one [b - k + off, b + off).
    pad_l = max(off, k - off)
    pad_r = max(k - 1 - off, off - 1)

    def expected(chrom: str, start: int, end: int) -> np.ndarray:
        w = int(end) - int(start)
        seq = _fetch_subseq(genome, cache, chrom, int(start) - pad_l, int(end) + pad_r)
        fwd = table.score_sequence(seq)[pad_l : pad_l + w]
        rev = table.score_sequence_reverse(seq)[pad_l : pad_l + w]
        return fwd + rev

    return expected


def find_active_windows(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    tf_len_range: tuple[int, int] = (20, 120),
    bin_size: int = 50,
    min_bin_frags: int = 12,
    merge_gap: int = 250,
    pad: int = 150,
) -> list[tuple[int, int]]:
    """Identify active genomic windows on a chromosome in ``O(N_frags)`` time using binned fragment midpoints."""
    if len(starts) == 0:
        return []
    fl = ends - starts
    mask = (fl >= tf_len_range[0]) & (fl <= tf_len_range[1])
    s = starts[mask]
    e = ends[mask]
    if len(s) == 0:
        return []
    mids = (s.astype(np.int64) + e.astype(np.int64)) >> 1
    max_pos = int(e.max()) + pad + bin_size
    n_bins = max_pos // bin_size + 1
    counts = np.bincount(mids // bin_size, minlength=n_bins)
    active_bins = np.flatnonzero(counts >= min_bin_frags)
    if len(active_bins) == 0:
        return []

    win_starts = active_bins * bin_size - pad
    win_ends = (active_bins + 1) * bin_size + pad

    merged: list[tuple[int, int]] = []
    cur_s = max(0, int(win_starts[0]))
    cur_e = int(win_ends[0])
    for ws, we in zip(win_starts[1:], win_ends[1:], strict=True):
        if ws <= cur_e + merge_gap:
            cur_e = max(cur_e, int(we))
        else:
            merged.append((cur_s, cur_e))
            cur_s = max(0, int(ws))
            cur_e = int(we)
    merged.append((cur_s, cur_e))
    return merged


def scan_genome_pillars(
    store: FragmentStore,
    *,
    chroms: Collection[str] | None = None,
    len_range: tuple[int, int] = (20, 250),
    len_bin: int = 10,
    tf_len_range: tuple[int, int] = (20, 120),
    pillar_window: int = 3,
    motif_width_range: tuple[int, int] = (10, 24),
    footprint_widths: Sequence[int] | np.ndarray | None = None,
    flank_pad: int = 2,
    min_bin_frags: int = 12,
    flank_mode: str = "both",
    pseudocount: float = 2.0,
    min_pillar_score: float = 8.0,
    min_len_bins: int = 4,
    min_pillar_prominence: float = 2.0,
    min_protection_score: float | None = None,
    min_raw_protection: float = 0.20,
    min_motif_distance: int = 12,
    expected_cuts: Callable[[str, int, int], np.ndarray] | None = None,
    return_pillars: bool = False,
    caller: str = "profile",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Scan active regions across the whole genome (or selected chromosomes) for de-novo
    protected motif footprints ``[l, r)`` and multi-length cut pillars.

    Uses ``find_active_windows`` followed by the same per-window profile accumulation
    (accelerated by Numba when installed) and shared pair-based calling as
    [`scan_locus_pillars`][pyvplot.scan_locus_pillars], so Numba and pure-Python backends
    produce identical intervals without fixed-size buffer truncation.

    ``expected_cuts`` is called as ``expected_cuts(chrom, start, end)`` for each active window
    and must return ``end - start`` per-bp sequence-expected cleavage rates (any scale), the
    same as ``scan_locus_pillars(expected_cuts=...)`` for that window. A callable, because
    a genome-length array would not fit in memory.

    ``caller``, ``footprint_widths``, ``flank_pad`` and the thresholds mean the same as in
    ``scan_locus_pillars``; with the default ``caller="profile"`` each window's 1-bp
    protection profile is computed and its peaks are called.
    """
    if store.is_stranded:
        raise ValueError("scan_genome_pillars requires paired-end fragment intervals; store is stranded.")
    if flank_mode not in {"both", "either"}:
        raise ValueError(f"flank_mode must be 'both' or 'either', got {flank_mode!r}.")
    _check_caller(caller)
    min_prot = _default_min_protection(caller, min_protection_score)
    if footprint_widths is None:
        widths = np.arange(8, 36, 1, dtype=np.int32)
    else:
        widths = np.asarray(list(footprint_widths), dtype=np.int32)
        if len(widths) == 0 or np.any(widths < 4):
            raise ValueError("`footprint_widths` must contain integers >= 4.")
    min_len, max_len = int(len_range[0]), int(len_range[1])
    if max_len <= min_len or len_bin <= 0:
        raise ValueError("`len_range` must have max_len > min_len and `len_bin` > 0.")

    target_chroms = list(store.chroms) if chroms is None else [c for c in store.chroms if c in set(chroms)]
    motif_frames: list[pd.DataFrame] = []
    pillar_frames: list[pd.DataFrame] = []

    for chrom in target_chroms:
        s_arr = store.starts[chrom]
        e_arr = store.ends[chrom]
        wins = find_active_windows(
            s_arr,
            e_arr,
            tf_len_range=tf_len_range,
            min_bin_frags=min_bin_frags,
        )
        if not wins:
            continue

        for w0, w1 in wins:
            W = int(w1 - w0)
            if W <= 0:
                continue
            cut_mat, left_cuts, right_cuts, cov_tf = _accumulate_region(
                s_arr,
                e_arr,
                w0,
                w1,
                min_len,
                max_len,
                int(len_bin),
                int(tf_len_range[0]),
                int(tf_len_range[1]),
            )
            win_mat = _box_filter_2d_cols(cut_mat, int(pillar_window))
            log_mat = np.log2(1.0 + win_mat)
            pillar_score = log_mat.sum(axis=0)
            n_len_bins = (win_mat >= 1.0).sum(axis=0).astype(np.int32)

            exp_w = None if expected_cuts is None else np.asarray(expected_cuts(chrom, int(w0), int(w1)), dtype=np.float64)
            cov_weight = _compute_cov_weight(cov_tf)
            p_df, m_df = _call_intervals_from_arrays(
                chrom=chrom,
                L=w0,
                W=W,
                pillar_score=pillar_score,
                n_len_bins=n_len_bins,
                left_cuts=left_cuts,
                right_cuts=right_cuts,
                cov_tf=cov_tf,
                cov_weight=cov_weight,
                motif_width_range=tuple(motif_width_range),
                flank_mode=flank_mode,
                pseudocount=float(pseudocount),
                pillar_window=int(pillar_window),
                min_pillar_score=float(min_pillar_score),
                min_len_bins=int(min_len_bins),
                min_pillar_prominence=float(min_pillar_prominence),
                min_pillar_distance=5,
                min_protection_score=min_prot,
                min_raw_protection=float(min_raw_protection),
                min_motif_distance=int(min_motif_distance),
                expected_cuts=exp_w,
                call_pairs=caller == "pairs",
            )
            if caller == "profile":
                _, _, p1, p1r, bw = _protection_profile(
                    pillar_score, cov_weight, widths, tuple(motif_width_range), int(flank_pad),
                    flank_mode, float(pseudocount), exp_w,
                )
                m_df = _call_profile_peaks(
                    chrom=chrom,
                    L=w0,
                    protection=p1,
                    raw_protection=p1r,
                    best_widths=bw,
                    pillar_score=pillar_score,
                    cov_tf=cov_tf,
                    flank_pad=int(flank_pad),
                    min_protection_score=min_prot,
                    min_raw_protection=float(min_raw_protection),
                    min_motif_distance=int(min_motif_distance),
                    expected_cuts=exp_w,
                )
            if len(m_df) > 0:
                motif_frames.append(m_df)
            if return_pillars and len(p_df) > 0:
                pillar_frames.append(p_df)

    if motif_frames:
        motifs_df = (
            pd.concat(motif_frames, ignore_index=True)
            .sort_values("composite_score", ascending=False)
            .reset_index(drop=True)
        )
    else:
        motifs_df = pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "center",
                "width",
                "name",
                "score",
                "strand",
                "protection_score",
                "raw_log2_protection",
                "core_pillar_score",
                "left_pillar_score",
                "right_pillar_score",
                "tf_coverage",
                "composite_score",
            ]
        )

    if not return_pillars:
        return motifs_df

    if pillar_frames:
        pillars_df = (
            pd.concat(pillar_frames, ignore_index=True)
            .sort_values("pillar_score", ascending=False)
            .reset_index(drop=True)
        )
    else:
        pillars_df = pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "pos",
                "name",
                "score",
                "strand",
                "pillar_score",
                "n_len_bins",
                "cuts",
                "left_cuts",
                "right_cuts",
                "left_frac",
                "tf_coverage",
            ]
        )
    return motifs_df, pillars_df
