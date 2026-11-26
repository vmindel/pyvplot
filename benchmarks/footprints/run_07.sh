#!/bin/bash
#BSUB -J fp_modisco
#BSUB -q short
#BSUB -R "rusage[mem=48000]"
#BSUB -M 48000
#BSUB -R "affinity[thread*8]"
#BSUB -oo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_modisco.%J.log
#BSUB -eo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_modisco.%J.err
set -euo pipefail
PY=/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python
export PYTHONNOUSERSITE=1 NPROC=8 OMP_NUM_THREADS=1
cd /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints
$PY 07_fp_modisco.py
