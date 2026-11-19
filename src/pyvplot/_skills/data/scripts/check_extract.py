#!/usr/bin/env python3
"""Independent brute-force verification of pyvplot strand-aware extraction.

Compares ``pyvplot.vplot`` and ``pyvplot.to_matrix`` (midpoint, cuts, cuts_plus,
cuts_minus, coverage) against an independent O(N_anchors * N_fragments) Python
loop on a sample of anchors.

Examples
--------
Run on a synthetic dataset with both + and - strand anchors:
    python scripts/check_extract.py --synthetic

Verify a real cached FragmentStore against a BED anchor file:
    python scripts/check_extract.py --store sample.frags.npz --anchors motifs.bed --n 25
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

import pyvplot as vp


def _naive_vplot(
    store: vp.FragmentStore,
    anchors: pd.DataFrame,
    flank: int,
    min_len: int,
    max_len: int,
    pos_bin: int,
    len_bin: int,
    feature: str,
) -> np.ndarray:
    n_pos = (2 * flank) // pos_bin
    n_len = (max_len - min_len) // len_bin
    grid = np.zeros((n_len, n_pos), dtype=np.float64)
    for _, row in anchors.iterrows():
        chrom = str(row["chrom"])
        if chrom not in store.starts:
            continue
        c = float(row["center"])
        a_st = -1 if str(row.get("strand", "+")) == "-" else 1
        starts = store.starts[chrom]
        ends = store.ends[chrom]
        for j in range(len(starts)):
            s, e = int(starts[j]), int(ends[j])
            fl = e - s
            if not (min_len <= fl < max_len):
                continue
            li = (fl - min_len) // len_bin
            if feature == "midpoint":
                mid = 0.5 * (s + e)
                rels = [(mid - c) if a_st >= 0 else (c - mid)]
            else:
                rels = [s - c, e - c] if a_st >= 0 else [c - e, c - s]
            for rel in rels:
                pi = int(math.floor((rel + flank) / pos_bin))
                if 0 <= pi < n_pos:
                    grid[li, pi] += 1.0
    return grid


def _naive_matrix(
    store: vp.FragmentStore,
    anchors: pd.DataFrame,
    flank: int,
    bin_size: int,
    feature: str,
    len_range: tuple[int, int] | None,
) -> np.ndarray:
    n_bins = (2 * flank) // bin_size
    out = np.zeros((len(anchors), n_bins), dtype=np.float64)
    for i, (_, row) in enumerate(anchors.iterrows()):
        chrom = str(row["chrom"])
        if chrom not in store.starts:
            continue
        c = float(row["center"])
        a_st = -1 if str(row.get("strand", "+")) == "-" else 1
        starts = store.starts[chrom]
        ends = store.ends[chrom]
        for j in range(len(starts)):
            s, e = int(starts[j]), int(ends[j])
            fl = e - s
            if len_range is not None and not (len_range[0] <= fl <= len_range[1]):
                continue
            if feature == "coverage":
                r_s = (s - c) if a_st >= 0 else (c - e)
                r_e = (e - c) if a_st >= 0 else (c - s)
                for b in range(n_bins):
                    lo = -flank + b * bin_size
                    hi = lo + bin_size
                    ov = min(r_e, hi) - max(r_s, lo)
                    if ov > 0:
                        out[i, b] += ov / bin_size
                continue
            if feature == "midpoint":
                mid = 0.5 * (s + e)
                rels = [(mid - c) if a_st >= 0 else (c - mid)]
            elif feature == "cuts":
                rels = [s - c, e - c] if a_st >= 0 else [c - e, c - s]
            elif feature == "cuts_plus":
                rels = [(s - c) if a_st >= 0 else (c - e)]
            else:
                rels = [(e - c) if a_st >= 0 else (c - s)]
            for rel in rels:
                bi = int(math.floor((rel + flank) / bin_size))
                if 0 <= bi < n_bins:
                    out[i, bi] += 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Brute-force verification of pyvplot extraction.")
    parser.add_argument("--synthetic", action="store_true", help="verify on a generated synthetic dataset")
    parser.add_argument("--store", help="path to cached FragmentStore (.npz or .parquet)")
    parser.add_argument("--anchors", help="path to anchor BED file")
    parser.add_argument("--n", type=int, default=20, help="number of anchors to sample (default 20)")
    parser.add_argument("--flank", type=int, default=120, help="half-window in bp (default 120)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        s = rng.integers(500, 5000, size=600, dtype=np.int32)
        e = s + rng.integers(20, 220, size=600, dtype=np.int32)
        store = vp.FragmentStore({"chr1": s}, {"chr1": e}, assay="chec")
        anchors = pd.DataFrame(
            {
                "chrom": ["chr1"] * args.n,
                "center": rng.integers(700, 4800, size=args.n),
                "strand": rng.choice(["+", "-"], size=args.n),
            }
        )
    else:
        if not args.store or not args.anchors:
            parser.error("Provide --synthetic OR both --store and --anchors.")
        store = vp.FragmentStore.load(args.store)
        norm = vp.normalize_anchors(args.anchors)
        idx = rng.choice(norm.n_anchors, size=min(args.n, norm.n_anchors), replace=False)
        anchors = pd.DataFrame(
            {
                "chrom": norm.chroms[idx],
                "center": norm.centers[idx],
                "strand": np.where(norm.strands[idx] < 0, "-", "+"),
            }
        )

    mismatches = 0
    for feat in ("midpoint", "cuts"):
        got = vp.vplot(store, anchors, flank=args.flank, min_len=20, max_len=220, pos_bin=4, len_bin=5, feature=feat).counts
        ref = _naive_vplot(store, anchors, flank=args.flank, min_len=20, max_len=220, pos_bin=4, len_bin=5, feature=feat)
        if not np.allclose(got, ref, rtol=0, atol=1e-12):
            mismatches += 1
            print(f"FAIL vplot feature={feat}: max diff {np.max(np.abs(got - ref))}")

    for feat in ("midpoint", "cuts", "cuts_plus", "cuts_minus", "coverage"):
        got = vp.to_matrix(store, anchors, flank=args.flank, bin_size=6, feature=feat, len_range=(30, 180)).values
        ref = _naive_matrix(store, anchors, flank=args.flank, bin_size=6, feature=feat, len_range=(30, 180))
        if not np.allclose(got, ref, rtol=0, atol=1e-12):
            mismatches += 1
            print(f"FAIL matrix feature={feat}: max diff {np.max(np.abs(got - ref))}")

    print(f"Checked {len(anchors)} anchors across 7 extraction modes: {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
