# Assay Presets & Coordinate Geometry

## Built-in Presets (`vp.ASSAY_PRESETS`)

| Name | Aliases | `(left_shift, right_shift)` | `default_feature` | `paired_end` | `size_bins` |
|---|---|---|---|---|---|
| `chec` | `chec-seq` | `(0, 0)` | `"cuts"` | `True` | `tf`: `(20, 120)`, `mononucleosome`: `(130, 200)` |
| `atac` | `atac-seq` | `(+4, -5)` | `"cuts"` | `True` | `nfr`: `(10, 100)`, `mononucleosome`: `(180, 247)`, `dinucleosome`: `(315, 473)` |
| `mnase` | `mnase-seq` | `(0, 0)` | `"midpoint"` | `True` | `subnucleosome`: `(20, 120)`, `nucleosome`: `(135, 165)` |
| `cutrun` | `cut&run`, `cut-run` | `(0, 0)` | `"midpoint"` | `True` | `short`: `(20, 120)`, `nucleosome`: `(150, 200)` |
| `cuttag` | `cut&tag`, `cut-tag` | `(0, 0)` | `"cuts"` | `True` | `short`: `(20, 120)`, `nucleosome`: `(150, 200)` |
| `chip_exo` | `chip-exo`, `chip-nexus` | `(0, 0)` | `"cuts_stranded"` | `False` | `{}` |
| `generic` | — | `(0, 0)` | `"midpoint"` | `True` | `{}` |

Customize or register presets with `vp.get_assay("atac").with_overrides(...)` and `vp.register_assay(cfg)`.

## Anchor Input Formats (`vp.normalize_anchors`)

Every extraction function accepts a `pandas.DataFrame`, `dict`, list of records, or path to a `.bed` / `.bed.gz` file:
- **Chromosome column:** `chrom`, `chr`, `seqnames`, or `chromosome`.
- **Center column:** `center`, `pos`, `midpoint`, or `summit`; alternatively `start` and `end` (uses exact midpoint `0.5 * (start + end)`).
- **Strand column (optional):** `strand`, `orientation`, or `dir`. `"-"` or `-1` triggers 5'->3' strand flipping; `"+"`, `"."`, `1`, or missing defaults to `+1`.
- **Anchor IDs:** `DataFrame.index` is preserved as `region_index`; for BED input, column 4 (name) becomes the index.
- **Chromosome names must match the store.** Anchors on chromosomes the store lacks produce a `UserWarning` and all-zero rows.
- Interpretation: fragments that *contain* a protected site put `cuts_plus` upstream and `cuts_minus` downstream of it; fragments that *abut* it do the reverse.

## Strand-Flipping Reflection Rule

For an anchor at `center` with `strand == "-"`, coordinates are reflected by `x -> 2 * center - x`:
- Fragment `[start, end)` maps to relative interval `[center - end, center - start)`.
- `rel_midpoint = center - 0.5 * (start + end)`
- `cuts_plus` (the fragment end that is upstream in anchor orientation; for unstranded paired-end data this is not a read strand):
  - Paired-end: `start - center` on `+` anchors, `center - end` on `-` anchors.
  - Single-end (`chip_exo`): reads with `frag_strand * anchor_strand > 0`.
- `cuts_minus` (the fragment end that is downstream in anchor orientation):
  - Paired-end: `end - center` on `+` anchors, `center - start` on `-` anchors.
  - Single-end (`chip_exo`): reads with `frag_strand * anchor_strand < 0`.

## Asymmetric Windows & Raw Event Tables

```python
# Promoter window: 1000 bp upstream to 500 bp downstream (flipped properly for '-' genes!)
mat = vp.to_matrix(frags, tss_anchors, upstream=1000, downstream=500, bin_size=10)

# Compact per-fragment event table for custom analysis
events = vp.extract_events(frags, anchors, flank=200, len_range=(20, 120), overlap_mode="midpoint")  # or len_range="tf"
# Columns: anchor_idx, rel_mid, rel_left, rel_right, frag_len (plus frag_strand for chip_exo)
```
