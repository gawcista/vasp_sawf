#!/bin/bash -l
#SBATCH -D ./
#SBATCH -J SAWF
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --partition=p.short
#SBATCH --mem=0
#SBATCH --time=1:00:00

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$PWD/.cache/numba" MPLCONFIGDIR="$PWD/.cache/matplotlib"

srun sawf-extract --energy-tol=1e-4 --accept-approximate \
    --matrix-tol=1e-3 --hamiltonian-tol=1e-4 --output symmetry_standard
srun sawf-run --symmetry symmetry_standard --output sawf_standard
