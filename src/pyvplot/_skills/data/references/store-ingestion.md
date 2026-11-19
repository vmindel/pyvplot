# FragmentStore Ingestion, Caching & Filtering

`vp.FragmentStore` stores 0-based, half-open `[start, end)` intervals as sorted `int32` per-chromosome arrays (`starts`, `ends`, optional `strands`, plus `max_len` per chromosome to tightly bound binary search).

## From BAM / CRAM

```python
import pyvplot as vp

frags = vp.FragmentStore.from_bam(
    "sample.bam",
    assay="chec",
    min_mapq=20,
    dedup=False,                # False | True (SAM 0x400 + coords) | "flag" | "coords"
    min_len=10,
    max_len=1000,
    exclude_chroms={"chrM", "MT"},
    pairing="read1",            # "read1" (fast O(1) memory) or "mate" (checks both mates' MAPQ)
    apply_shift=True,           # applies assay.left_shift / assay.right_shift ONCE
)
```

## From BED, 10x/ArchR `.frag.tsv.gz`, or BEDPE

```python
# 10x / ArchR fragment files are usually already Tn5-shifted -> apply_shift=False (default)
atac = vp.FragmentStore.from_fragments_bed(
    "fragments.tsv.gz",
    assay="atac",
    apply_shift=False,
    use_count=False,            # set True to expand the 5th column fragment count
)

# 6+-column BEDPE is auto-detected (inter-chromosomal discordant pairs are skipped)
chec = vp.FragmentStore.from_fragments_bed("pairs.bedpe", assay="chec")
```

## From DataFrame or NumPy Arrays

```python
store = vp.FragmentStore.from_dataframe(df, assay="mnase", chrom_col="chrom", start_col="start", end_col="end")
store = vp.FragmentStore.from_arrays(starts_dict, ends_dict, assay="chec", apply_shift=False)
```

## Caching (`.npz` and `.parquet`)

Parse each BAM once and cache to disk:

```python
frags.save("sample.frags.npz")              # uncompressed .npz -> fastest load, zero extra deps
frags.save("sample.frags.npz", compressed=True)
frags.save("sample.frags.parquet")          # requires pyvplot[io] (pyarrow)

frags = vp.FragmentStore.load("sample.frags.npz")
```

Both `.npz` and `.parquet` preserve `assay`, `strands` (for `chip_exo`), `metadata`, and sort order.

## Filtering & Merging

```python
tf_frags = frags.filter(min_len=20, max_len=120, exclude_chroms={"chrM"}, dedup=True)
merged = vp.FragmentStore.merge([rep1, rep2], dedup=False)
```

## Footguns

- **Do not double-shift ATAC fragments:** `from_bam(..., assay="atac")` shifts by `(+4, -5)` by default (`apply_shift=True`). `from_fragments_bed(...)` defaults to `apply_shift=False`. If your BED came from raw unshifted `bedtools bamtobed`, pass `apply_shift=True`.
- **Both R1 orientations are handled:** `pairing="read1"` uses signed `template_length` (`tlen > 0` for forward R1, `tlen < 0` for reverse R1) so neither orientation is dropped. Use `pairing="mate"` if you strictly need both mates to pass `min_mapq`.
- **Merging stranded and unstranded stores:** `vp.FragmentStore.merge` raises `ValueError` if you mix `chip_exo` (stranded) and paired-end (unstranded) stores.
