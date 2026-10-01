#!/bin/bash
#SBATCH --job-name=sawf-extract
#SBATCH --partition=p.large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/ptmp/tSnS/sawf661
#SBATCH --output=extract.out
#SBATCH --error=extract.err
#SBATCH --open-mode=append
#SBATCH --export=ALL

set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$PWD/.cache/numba" MPLCONFIGDIR="$PWD/.cache/matplotlib"

srun python -m vasp_sawf.extract_symmetry \
    --seed /ptmp/tSnS/scdm661/wannier90 \
    --wavecar /ptmp/tSnS/scdm661/WAVECAR \
    --outcar /ptmp/tSnS/scdm661/OUTCAR \
    --output "$PWD/symmetry"
