#!/bin/bash
#BSUB -J fp_denovo_occ
#BSUB -q short-gpu
#BSUB -gpu "num=1:j_exclusive=no:gmem=16G"
#BSUB -R "rusage[mem=48000]"
#BSUB -M 48000
#BSUB -R "affinity[thread*8]"
#BSUB -oo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_denovo_occ.%J.log
#BSUB -eo /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints/out/logs/fp_denovo_occ.%J.err
set -euo pipefail
PY=/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python
export PYTHONNOUSERSITE=1 TORCHDYNAMO_DISABLE=1 MPLBACKEND=Agg
cd /home/labs/barkailab/vovam/KLFNov2026_paper/packages/pyvplot/benchmarks/footprints
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY 11_denovo_occupancy.py
