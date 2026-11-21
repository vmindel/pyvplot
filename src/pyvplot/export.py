"""Export size-stratified cut-site, midpoint, and coverage tracks to bedGraph, BigWig, and pymisha."""

from __future__ import annotations

import gzip
import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyvplot.matrices import _resolve_len_range
from pyvplot.store import FragmentStore


def to_binned_track(
    store: FragmentStore,
    feature: str | None = None,
    len_range: tuple[int, int] | str | None = None,
    bin_size: int = 10,
    normalize: str | None = "cpm",
    chroms: Collection[str] | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Compute a sorted genome-wide binned signal track as a DataFrame.

    Returns a `pandas.DataFrame` with columns ``chrom``, ``start``,
    ``end``, ``value`` containing non-zero genomic bins.

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to bin.
    feature:
        ``"midpoint"``, ``"cuts"``, ``"cuts_plus"``, ``"cuts_minus"``, or
        ``"coverage"``. Defaults to ``store.assay.default_feature``.
    len_range:
        Optional inclusive ``(min_len, max_len)`` filter or named assay size bin.
    bin_size:
        Genomic bin size in bp (default ``10``).
    normalize:
        ``None`` / ``"raw"``, ``"cpm"``, or ``"rpkm"``.
    chroms:
        Optional subset of chromosomes to include.
    chrom_sizes:
        Optional mapping ``chrom -> length`` used to clip bin end coordinates.
    """
    if bin_size <= 0:
        raise ValueError("`bin_size` must be a positive integer.")

    if feature is None:
        feat = "cuts" if store.assay.default_feature == "cuts_stranded" else store.assay.default_feature
    else:
        feat = feature

    if feat == "cuts_stranded":
        feat = "cuts"
    valid_feats = {"midpoint", "cuts", "cuts_plus", "cuts_minus", "coverage"}
    if feat not in valid_feats:
        raise ValueError(f"Unsupported feature {feat!r}; expected one of {sorted(valid_feats)}.")

    resolved_len = _resolve_len_range(store, len_range)
    target_chroms = list(store.chroms) if chroms is None else [c for c in store.chroms if c in set(chroms)]

    scale = 1.0
    if normalize is not None and normalize.lower() != "raw":
        norm_l = normalize.lower()
        if norm_l == "cpm":
            scale = 1e6 / float(store.n_fragments) if store.n_fragments > 0 else 1.0
        elif norm_l == "rpkm":
            denom = (float(store.n_fragments) / 1e6) * (float(bin_size) / 1000.0)
            scale = 1.0 / denom if denom > 0.0 else 1.0
        else:
            raise ValueError(
                f"Unknown normalize mode {normalize!r}; expected None, 'raw', 'cpm', or 'rpkm'."
            )

    frames: list[pd.DataFrame] = []
    bs = int(bin_size)

    for chrom in target_chroms:
        s = store.starts[chrom]
        e = store.ends[chrom]
        st = store.strands[chrom] if store.strands is not None else None
        if len(s) == 0:
            continue

        if resolved_len is not None:
            fl = e - s
            mask = (fl >= resolved_len[0]) & (fl <= resolved_len[1])
            s = s[mask]
            e = e[mask]
            if st is not None:
                st = st[mask]
            if len(s) == 0:
                continue

        chrom_limit = int(chrom_sizes[chrom]) if chrom_sizes is not None and chrom in chrom_sizes else None

        if feat == "coverage":
            # Exact per-bin average base coverage via sweep over bin endpoints
            b_start = (s // bs).astype(np.int64)
            b_end = ((e - 1) // bs).astype(np.int64)
            max_bin = int(np.max(b_end))
            # Difference array for full bins strictly inside (b_start, b_end) plus boundary partial overlaps
            vals_arr = np.zeros(max_bin + 2, dtype=np.float64)
            same_bin = b_start == b_end
            if np.any(same_bin):
                np.add.at(
                    vals_arr,
                    b_start[same_bin],
                    (e[same_bin] - s[same_bin]).astype(np.float64) / float(bs),
                )
            multi = ~same_bin
            if np.any(multi):
                bs_m = b_start[multi]
                be_m = b_end[multi]
                s_m = s[multi]
                e_m = e[multi]
                # First bin partial overlap
                first_ov = ((bs_m + 1) * bs - s_m).astype(np.float64) / float(bs)
                np.add.at(vals_arr, bs_m, first_ov)
                # Last bin partial overlap
                last_ov = (e_m - be_m * bs).astype(np.float64) / float(bs)
                np.add.at(vals_arr, be_m, last_ov)
                # Full interior bins [bs_m + 1, be_m - 1]
                has_interior = be_m > (bs_m + 1)
                if np.any(has_interior):
                    diff = np.zeros(max_bin + 2, dtype=np.float64)
                    np.add.at(diff, bs_m[has_interior] + 1, 1.0)
                    np.add.at(diff, be_m[has_interior], -1.0)
                    vals_arr += np.cumsum(diff)
            nz_bins = np.flatnonzero(vals_arr > 0.0)
            bin_vals = vals_arr[nz_bins] * scale
        else:
            if feat == "midpoint":
                pos = (s.astype(np.int64) + e.astype(np.int64)) // 2
            elif feat == "cuts":
                if st is not None:
                    pos = s.astype(np.int64)
                else:
                    pos = np.concatenate([s.astype(np.int64), e.astype(np.int64)])
            elif feat == "cuts_plus":
                pos = s[st >= 0].astype(np.int64) if st is not None else s.astype(np.int64)
            else:  # cuts_minus
                pos = s[st < 0].astype(np.int64) if st is not None else e.astype(np.int64)

            if len(pos) == 0:
                continue
            if chrom_limit is not None:
                pos = pos[(pos >= 0) & (pos < chrom_limit)]
                if len(pos) == 0:
                    continue
            bins = pos // bs
            nz_bins, raw_counts = np.unique(bins, return_counts=True)
            bin_vals = raw_counts.astype(np.float64) * scale

        starts_out = nz_bins * bs
        ends_out = starts_out + bs
        if chrom_limit is not None:
            valid = starts_out < chrom_limit
            starts_out = starts_out[valid]
            ends_out = np.minimum(ends_out[valid], chrom_limit)
            bin_vals = bin_vals[valid]

        if len(starts_out) > 0:
            frames.append(
                pd.DataFrame(
                    {
                        "chrom": np.repeat(chrom, len(starts_out)),
                        "start": starts_out.astype(np.int64),
                        "end": ends_out.astype(np.int64),
                        "value": bin_vals,
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["chrom", "start", "end", "value"])
    return pd.concat(frames, ignore_index=True)


def to_bedgraph(
    store: FragmentStore,
    output_path: str | Path,
    feature: str | None = None,
    len_range: tuple[int, int] | str | None = None,
    bin_size: int = 10,
    normalize: str | None = "cpm",
    chroms: Collection[str] | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    track_name: str | None = None,
) -> Path:
    """Export a size-stratified signal track to a sorted bedGraph (or ``.bedGraph.gz``) file."""
    df = to_binned_track(
        store=store,
        feature=feature,
        len_range=len_range,
        bin_size=bin_size,
        normalize=normalize,
        chroms=chroms,
        chrom_sizes=chrom_sizes,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out.suffix == ".gz" else open

    with opener(out, "wt", encoding="utf-8") as fh:
        if track_name is not None:
            fh.write(f'track type=bedGraph name="{track_name}"\n')
        for chrom, start, end, val in zip(
            df["chrom"], df["start"], df["end"], df["value"], strict=True
        ):
            fh.write(f"{chrom}\t{int(start)}\t{int(end)}\t{float(val):.6g}\n")
    return out


def to_bigwig(
    store: FragmentStore,
    output_path: str | Path,
    chrom_sizes: Mapping[str, int],
    feature: str | None = None,
    len_range: tuple[int, int] | str | None = None,
    bin_size: int = 10,
    normalize: str | None = "cpm",
    chroms: Collection[str] | None = None,
) -> Path:
    """Export a size-stratified signal track to BigWig via ``pyBigWig``."""
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError(
            "Exporting BigWig files requires `pyBigWig`. Install it with `pip install pyvplot[bigwig]`."
        ) from exc

    df = to_binned_track(
        store=store,
        feature=feature,
        len_range=len_range,
        bin_size=bin_size,
        normalize=normalize,
        chroms=chroms,
        chrom_sizes=chrom_sizes,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [(str(c), int(size)) for c, size in chrom_sizes.items()]
    unknown = sorted(set(df["chrom"]) - {c for c, _ in header})
    if unknown:
        shown = ", ".join(repr(c) for c in unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        n_rows = int(df["chrom"].isin(unknown).sum())
        warnings.warn(
            f"{n_rows} bins on chromosomes missing from chrom_sizes ({shown}) are not written "
            "to the BigWig. Add them to chrom_sizes or restrict the export with chroms=.",
            UserWarning,
            stacklevel=2,
        )
    bw = pyBigWig.open(str(out), "w")
    try:
        bw.addHeader(header)
        if len(df) > 0:
            # Write in header chromosome order
            for chrom, _ in header:
                sub = df[df["chrom"] == chrom]
                if len(sub) == 0:
                    continue
                bw.addEntries(
                    [chrom] * len(sub),
                    sub["start"].astype(int).tolist(),
                    ends=sub["end"].astype(int).tolist(),
                    values=sub["value"].astype(float).tolist(),
                )
    finally:
        bw.close()
    return out


def to_pymisha(
    store: FragmentStore,
    track_name: str,
    feature: str | None = None,
    len_range: tuple[int, int] | str | None = None,
    bin_size: int = 10,
    mode: str = "sparse",
    normalize: str | None = "cpm",
    description: str = "",
    chroms: Collection[str] | None = None,
    chrom_sizes: Mapping[str, int] | None = None,
    defval: float = 0.0,
) -> Any:
    """Export a size-stratified signal track directly into a ``pymisha`` genomic database.

    Parameters
    ----------
    store:
        [`FragmentStore`][pyvplot.FragmentStore] to export.
    track_name:
        Destination track name in the active ``pymisha`` database.
    mode:
        ``"sparse"`` (calls ``pymisha.gtrack_create_sparse``) or ``"dense"``
        (calls ``pymisha.gtrack_create_dense``).
    """
    if mode not in {"sparse", "dense"}:
        raise ValueError(f"Invalid mode {mode!r}; expected 'sparse' or 'dense'.")

    try:
        import pymisha as pm
    except ImportError as exc:
        raise ImportError(
            "Exporting tracks to pymisha requires `pymisha` to be installed and initialized."
        ) from exc

    df = to_binned_track(
        store=store,
        feature=feature,
        len_range=len_range,
        bin_size=bin_size,
        normalize=normalize,
        chroms=chroms,
        chrom_sizes=chrom_sizes,
    )
    intervals = df[["chrom", "start", "end"]].copy()
    values = df["value"].to_numpy(dtype=np.float64)

    if mode == "sparse":
        return pm.gtrack_create_sparse(track_name, description, intervals, values)
    return pm.gtrack_create_dense(
        track_name, description, intervals, values, binsize=int(bin_size), defval=float(defval)
    )
