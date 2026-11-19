# Installation

`pyvplot` requires **Python 3.10+** and has only two required dependencies (`numpy` and `pandas`). Performance and ecosystem integrations are available as optional extras.

## Basic Install

```bash
pip install pyvplot
```

## Optional Extras

| Extra | Packages | Enables |
|---|---|---|
| `numba` | `numba` | Multi-threaded JIT-compiled binary-search extraction & phasogram kernels |
| `bam` | `pysam` | Direct BAM / CRAM ingestion via `FragmentStore.from_bam` |
| `io` | `pyarrow` | Fast `.parquet` fragment store caching |
| `plot` | `matplotlib`, `plotnine`, `scipy` | Plotting helpers, 2D Gaussian smoothing, and SciPy peak finding |
| `bigwig` | `pyBigWig` | BigWig track export via `to_bigwig` |
| `all` | `numba`, `pysam`, `pyarrow`, `matplotlib`, `plotnine`, `scipy` | Recommended full analysis environment |

```bash
# Recommended full install
pip install "pyvplot[all]"

# Or pick only what you need
pip install "pyvplot[numba,bam,plot]"
```

## Development Install

```bash
git clone https://github.com/vmindel/pyvplot.git
cd pyvplot
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev,docs]"
.venv/bin/pytest
```
