#!/bin/bash
#SBATCH --job-name=sawf-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=extract-%j.out
#SBATCH --error=extract-%j.err

set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch [resource options] extract_ada.sbatch INPUT_DIR OUTPUT_DIR" >&2
    exit 2
fi
input_dir=$(realpath -- "$1")
output_dir="$2"
if [[ "$output_dir" != /* ]]; then
    output_dir="$PWD/$output_dir"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun --chdir="$input_dir" sawf-extract --output "$output_dir"
