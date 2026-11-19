# V-Plots & Differential V-Plots

`vp.vplot` streams fragments directly into a 2D `(n_len, n_pos)` grid using binary search + parallel chunked Numba accumulation (zero intermediate fragment table allocation).

## Computing & Normalizing a V-Plot

```python
import pyvplot as vp

vp_wt = vp.vplot(
    frags_wt,
    anchors,
    flank=300,              # or asymmetric upstream=500, downstream=200
    min_len=20,
    max_len=250,            # half-open [min_len, max_len)
    pos_bin=1,
    len_bin=1,
    feature="midpoint",     # "midpoint" (default), "cuts", "cuts_plus", "cuts_minus"
)

vp_cpm = vp_wt.normalize("cpm")
vp_oe = vp_wt.normalize("expected", expected_from="flank", flank_frac=0.2)
vp_oe_genome = vp_wt.normalize("expected", expected_from="genome")
vp_log2oe = vp_wt.normalize("log2_oe", expected_from="flank", pseudocount=0.1)
vp_sm = vp_oe.smooth(sigma=(1.5, 1.5))
```

| `normalize(method=...)` | Formula / Meaning |
|---|---|
| `"cpm"` | `counts / (n_fragments / 1e6) / n_anchors` (per-anchor CPM) |
| `"cpm_total"` | `counts / (n_fragments / 1e6)` |
| `"per_anchor"` | `counts / n_anchors` |
| `"density"` | `counts / counts.sum()` |
| `"expected"` | Divide each length row by expected background (`expected_from="flank"`, `"genome"`, or `"row_mean"`) |
| `"log2_oe"` | `log2((obs + pseudocount) / (expected + pseudocount))` |
| `"row_zscore"` | Z-score each length row independently across positions |

## Differential V-Plots

```python
vp_diff = vp_mut.diff(
    vp_wt,
    mode="log2fc",          # "log2fc", "subtract", "ratio", "zscore"
    pseudocount=0.5,
    smooth_sigma=(1.5, 1.5),
    normalize_first="cpm",  # auto-normalizes raw VPlots to CPM before differencing
    smooth_before=True,
)
```

`vp_diff.is_diff` is `True`, so `.plot()` and `.plot_plotnine()` automatically switch to a diverging red-white-blue scale centered at `0`.

## Marginals, Tidy Export & Faceted `VPlotCollection`

```python
tf_profile = vp_wt.marginal_position(len_range=(30, 120))
center_lens = vp_wt.marginal_length(pos_range=(-30, 30))

coll = (
    vp.VPlotCollection()
    .add(vp_wt.normalize("cpm"), tf="CTCF", condition="WT")
    .add(vp_mut.normalize("cpm"), tf="CTCF", condition="Mut")
)
df = coll.to_dataframe()
p = coll.plot_plotnine(facet="tf ~ condition", cmap="magma")
```

## Footguns

- **Nucleosomes drowning out TF V-signatures:** Raw or CPM V-plots are dominated by abundant ~150 bp nucleosome rows. Always inspect `.normalize("expected", expected_from="flank")` or `"log2_oe"` when looking for 40–100 bp TF footprints alongside nucleosomes.
- **Half-open length bins:** `min_len=20, max_len=250, len_bin=1` creates 230 uniform bins `[20, 21), ..., [249, 250)`. A fragment of exact length `250` is excluded; pass `max_len=251` if you want length `250` included.
- **Divisibility:** `upstream + downstream` must be divisible by `pos_bin`, and `max_len - min_len` must be divisible by `len_bin`.
- **Diffing pre-normalized V-plots:** `normalize_first="cpm"` only runs when **both** V-plots have `normalization == "raw"`. If you already normalized one V-plot, normalize the other with the same method before calling `.diff()`.
