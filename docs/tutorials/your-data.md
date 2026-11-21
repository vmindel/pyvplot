# Your own data

This page goes from a BAM and a motif BED to a cached store, QC, figures and tidy tables, first step by step and then as one script for a cluster job. [How pyvplot works](../concepts.md) explains each argument.

## 1. Build the store once per sample

```py
import pyvplot as vp

store = vp.FragmentStore.from_bam(
    "chec.bam",
    assay="chec",              # sets endpoint shift, default feature and size bins
    min_mapq=20,
    exclude_chroms={"chrM", "MT"},
    dedup=False,               # duplicates are kept by default; "flag" drops SAM-flagged ones
)
store.save("chec.frags.npz")   # reload with vp.FragmentStore.load(...)
print(store.qc_summary())
```

The BAM is read start to end, so any sort order works and no index is needed. Only properly paired reads are used.

Other inputs:

```py
# 10x / ArchR fragments: already Tn5-shifted, so no shift is applied by default.
atac = vp.FragmentStore.from_fragments_bed("fragments.tsv.gz", assay="atac")

# Any table with chrom/start/end columns (0-based, half-open).
store = vp.FragmentStore.from_dataframe(df, assay="mnase")
```

Check the length distribution before relying on the preset's named size bins:

```py
store.insert_size_distribution().plot(x="length", y="count")
from pyvplot.plotting import plot_insert_size_matplotlib
plot_insert_size_matplotlib(store, show_bands=True, max_len=500)   # shades the preset's bins
```

## 2. Anchors

A BED file is enough: chromosome names must match the BAM's, the centre of each interval is used, and strand comes from column 6. Motif-scanner output with a strand column works directly. Name (column 4) becomes the row label in matrices and score tables.

```py
v = store.vplot("motifs.bed", flank=300, min_len=20, max_len=250, pos_bin=5, len_bin=5)
```

If you see `UserWarning: ... anchors are on chromosomes with no fragments`, check which chromosomes it lists. If they are ones you excluded at ingestion (the script below drops `chrM`/`MT`), the warning is expected. If it says no anchor matched, the names differ (`chr1` vs `1`): fix the BED, not the store.

## 3. Analysis and export

```py
v.normalize("expected").plot()
v.to_dataframe().to_csv("vplot.tsv.gz", sep="\t", index=False)       # pos, length, value

fp = store.cut_site_footprint("motifs.bed", flank=100, len_range="tf")
fp.to_dataframe().to_csv("footprint.tsv.gz", sep="\t", index=False) # pos, strand, signal

scores = store.footprint_score("motifs.bed", len_range="tf")         # one row per motif
mat = store.to_matrix("motifs.bed", flank=1000, bin_size=10, feature="midpoint", len_range="tf")
mat.normalize("cpm").plot()                                          # heatmap, one row per motif

sizes = dict(line.split()[:2] for line in open("genome.chrom.sizes"))
sizes = {c: int(n) for c, n in sizes.items()}
vp.to_bigwig(store, "tf_midpoints.bw", chrom_sizes=sizes, feature="midpoint", len_range="tf")
```

The tidy tables read straight into R:

```r
library(ggplot2)
v <- readr::read_tsv("vplot.tsv.gz")
ggplot(v, aes(pos, length, fill = value)) + geom_raster() + scale_fill_viridis_c()
```

## 4. The same as a cluster script

The `pyvplot-report` command (source below, also runnable as [`examples/bam_to_vplot.py`](https://github.com/vmindel/pyvplot/blob/main/examples/bam_to_vplot.py)) runs steps 1 to 3 for one sample. It reuses the `.frags.npz` cache when that is newer than the BAM, so re-running with different plotting choices skips ingestion. It is tested end to end on a simulated BAM.

```bash
pyvplot-report --bam chec.bam --motifs motifs.bed --out results/chec
```

On LSF:

```bash
bsub -J vplot -n 4 -R "rusage[mem=8000]" -o logs/%J.out \
  pyvplot-report --bam chec.bam --motifs motifs.bed --out results/chec
```

Ingestion is one Python pass over the BAM. On a synthetic BAM it read 1 M read pairs in 1.5 s (1.1 s with `--threads 4`, which only parallelises BGZF decompression), with peak memory about 35 bytes per fragment above the interpreter's own ~175 MB. Real BAMs, with tags and unpaired reads, will be slower. As a rule of thumb, 30 M pairs need about 1 GB while reading. The analysis steps are seconds. Only Numba uses the extra cores (`-n 4`).

```py
"""BAM + motif BED -> cached fragment store, QC, V-plot, cut-site footprint, per-motif scores.

    pyvplot-report --bam sample.bam --motifs motifs.bed --out results/sample

Outputs (all prefixed by --out):
    .frags.npz             fragment store; reused if newer than the BAM and built with the same settings
    .qc.json               fragment counts, length stats, size-bin fractions
    .vplot.png             V-plot, CPM and flank-normalised side by side
    .vplot.tsv.gz          tidy V-plot (pos, length, value) for ggplot2
    .footprint.png         strand-separated 1-bp cut profile
    .footprint.tsv.gz      tidy cut profile (pos, strand, signal)
    .heatmap.png           one row per motif, short-fragment midpoints, strongest first
    .scores.tsv            per-motif protection scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    p.add_argument("--threads", type=int, default=1, help="BAM decompression threads")
    return p.parse_args(argv)


def load_store(args: argparse.Namespace) -> vp.FragmentStore:
    cache = args.out.with_name(args.out.name + ".frags.npz")
    settings = {
        "bam": str(args.bam.resolve()),
        "assay": args.assay,
        "min_mapq": args.min_mapq,
        "exclude_chroms": sorted(args.exclude_chroms),
        "dedup": args.dedup,
    }
    if cache.exists() and cache.stat().st_mtime > args.bam.stat().st_mtime:
        store = vp.FragmentStore.load(cache)
        if store.metadata.get("ingest_settings") == settings:
            print(f"Loading cached store {cache}")
            return store
        print("Ingestion settings changed; rebuilding the store")
    print(f"Reading {args.bam}")
    store = vp.FragmentStore.from_bam(
        args.bam,
        assay=args.assay,
        min_mapq=args.min_mapq,
        exclude_chroms=args.exclude_chroms,
        dedup="flag" if args.dedup else False,
        threads=args.threads,
    )
    store.metadata["ingest_settings"] = settings
    store.save(cache)
    return store


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display on cluster nodes
    import matplotlib.pyplot as plt

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

    mat = store.to_matrix(args.motifs, flank=500, bin_size=10, feature="midpoint", len_range=short)
    fig, _ = mat.normalize("cpm").plot(title=f"Midpoints, {short or 'all'} fragments")
    fig.savefig(prefix + ".heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {prefix}.*")


if __name__ == "__main__":
    main()
```
