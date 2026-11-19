#!/usr/bin/env python
"""BAM + motif BED -> cached fragment store, QC, V-plot, cut-site footprint, per-motif scores.

    python bam_to_vplot.py --bam sample.bam --motifs motifs.bed --out results/sample

Outputs (all prefixed by --out):
    .frags.npz             fragment store; reused on the next run if newer than the BAM
    .qc.json               fragment counts, length stats, size-bin fractions
    .vplot.png             V-plot, CPM and flank-normalised side by side
    .vplot.tsv.gz          tidy V-plot (pos, length, value) for ggplot2
    .footprint.png         strand-separated 1-bp cut profile
    .footprint.tsv.gz      tidy cut profile (pos, strand, signal)
    .scores.tsv            per-motif protection scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on cluster nodes
import matplotlib.pyplot as plt

import pyvplot as vp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bam", required=True, type=Path)
    p.add_argument("--motifs", required=True, type=Path, help="BED with strand in column 6")
    p.add_argument("--out", required=True, type=Path, help="output prefix")
    p.add_argument("--assay", default="chec", help="preset: chec, atac, mnase, cutrun, cuttag")
    p.add_argument("--flank", type=int, default=300, help="V-plot half-window (bp)")
    p.add_argument("--min-mapq", type=int, default=20)
    p.add_argument("--exclude-chroms", nargs="*", default=["chrM", "MT"])
    p.add_argument("--dedup", action="store_true", help="drop SAM-flagged duplicates")
    return p.parse_args(argv)


def load_store(args: argparse.Namespace) -> vp.FragmentStore:
    cache = args.out.with_name(args.out.name + ".frags.npz")
    if cache.exists() and cache.stat().st_mtime > args.bam.stat().st_mtime:
        print(f"Loading cached store {cache}")
        return vp.FragmentStore.load(cache)
    print(f"Reading {args.bam}")
    store = vp.FragmentStore.from_bam(
        args.bam,
        assay=args.assay,
        min_mapq=args.min_mapq,
        exclude_chroms=args.exclude_chroms,
        dedup="flag" if args.dedup else False,
    )
    store.save(cache)
    return store


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(args.out)

    store = load_store(args)
    print(store)

    qc = store.qc_summary()
    Path(prefix + ".qc.json").write_text(json.dumps(qc, indent=2))
    print("Size-bin fractions:", {k: round(v, 3) for k, v in qc["size_band_fractions"].items()})

    v = store.vplot(args.motifs, flank=args.flank, min_len=20, max_len=250, pos_bin=5, len_bin=5)
    v.to_dataframe().to_csv(prefix + ".vplot.tsv.gz", sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    v.normalize("cpm").plot(ax=axes[0], title="CPM")
    v.normalize("expected", expected_from="flank").plot(ax=axes[1], title="Observed / flank")
    fig.tight_layout()
    fig.savefig(prefix + ".vplot.png", dpi=150)
    plt.close(fig)

    # Short fragments carry the TF footprint; fall back to all lengths if the
    # preset has no "tf"-like bin.
    bins = store.assay.size_bins
    short = next((b for b in ("tf", "nfr", "short", "subnucleosome") if b in bins), None)
    fp = store.cut_site_footprint(args.motifs, flank=100, len_range=short)
    fp.to_dataframe().to_csv(prefix + ".footprint.tsv.gz", sep="\t", index=False)
    fig, ax = fp.plot(title=f"Cut sites, {short or 'all'} fragments")
    fig.savefig(prefix + ".footprint.png", dpi=150)
    plt.close(fig)

    scores = store.footprint_score(args.motifs, len_range=short)
    scores.to_csv(prefix + ".scores.tsv", sep="\t", index_label="motif")
    print(f"Wrote {prefix}.*")


if __name__ == "__main__":
    main()
