#!/bin/bash
#BSUB -J pyvplot_docs
#BSUB -q short
#BSUB -R "rusage[mem=16000]"
#BSUB -M 16000
#BSUB -R "affinity[thread*4]"
#BSUB -oo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/docs.%J.log
#BSUB -eo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/docs.%J.err
set -uo pipefail
PY=/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTHONPATH=/home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/pytest_shim
cd /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot
$PY docs/run_docs.py 2>&1 | tail -20
$PY -m pytest -q tests 2>&1 | tail -30
