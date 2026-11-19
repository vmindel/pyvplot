# Installation

pyvplot needs Python 3.10 or later. It is not on PyPI yet, so install from GitHub:

```bash
pip install "pyvplot[all] @ git+https://github.com/vmindel/pyvplot.git"
```

The core depends only on NumPy and pandas. The extras add:

| Extra | Packages | Needed for |
|---|---|---|
| `numba` | `numba` | Parallel compiled kernels. Results are identical without it; V-plots were about 6× slower without it on 4 cores. |
| `bam` | `pysam` | `FragmentStore.from_bam`; also opens FASTA files for `KmerBiasTable.from_cuts` |
| `io` | `pyarrow` | `.parquet` stores (`.npz` needs nothing extra) |
| `plot` | `matplotlib`, `plotnine`, `scipy` | `.plot()` / `.plot_plotnine()`, faster smoothing and peak finding |
| `bigwig` | `pyBigWig` | `to_bigwig` (not included in `all`) |
| `all` | everything above except `pyBigWig` | |

`to_pymisha` needs [`pymisha`](https://pypi.org/project/pymisha/) installed and a misha database initialised; it is not an extra.

On a cluster, install into a project environment once and point jobs at it:

```bash
uv venv ~/envs/pyvplot
uv pip install --python ~/envs/pyvplot/bin/python "pyvplot[all] @ git+https://github.com/vmindel/pyvplot.git"
```

The first call to each Numba kernel compiles it, which took about 4 s in total in our test. The compiled code is cached on disk and reused by later sessions. `NUMBA_CACHE_DIR` sets where the cache goes, which is useful if the install directory is shared or read-only.

## Development install

```bash
git clone https://github.com/vmindel/pyvplot.git
cd pyvplot
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,docs]"
.venv/bin/pytest
```
