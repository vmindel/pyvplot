#!/bin/bash
#BSUB -J fp_kmerbias
#BSUB -q short
#BSUB -R "rusage[mem=64000]"
#BSUB -M 64000
#BSUB -R "affinity[thread*4]"
#BSUB -oo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/.claude/worktrees/occupancy-score/benchmarks/footprints/out/logs/fp_kmerbias.%J.log
#BSUB -eo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/.claude/worktrees/occupancy-score/benchmarks/footprints/out/logs/fp_kmerbias.%J.err
set -euo pipefail
W=/home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/.claude/worktrees/occupancy-score
PY=/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 PYTHONPATH=$W/src
cd $W/benchmarks/footprints
$PY -c "import pyvplot; print('pyvplot from', pyvplot.__file__); assert '$W' in pyvplot.__file__"
$PY 09_kmer_binding_vs_bias.py
