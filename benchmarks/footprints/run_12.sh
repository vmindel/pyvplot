#!/bin/bash
#BSUB -J fp_occstrat
#BSUB -q short
#BSUB -R "rusage[mem=16000]"
#BSUB -M 16000
#BSUB -R "affinity[thread*8]"
#BSUB -oo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_occstrat.%J.log
#BSUB -eo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_occstrat.%J.err
set -euo pipefail
PY=/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python
export PYTHONNOUSERSITE=1 NPROC=8 OMP_NUM_THREADS=1 PYTHONPATH=/home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/.claude/worktrees/occupancy-score/src
cd /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints
$PY 12_occ_stratified.py
