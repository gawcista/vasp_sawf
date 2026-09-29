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
: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run extraction on a login node}"

INPUT_ROOT=/ptmp/tSnS/scdm661
PYTHON="$(command -v python)"
SEED="$INPUT_ROOT/wannier90"
WAVECAR="$INPUT_ROOT/WAVECAR"
OUTCAR="$INPUT_ROOT/OUTCAR"
OUTPUT="$PWD/symmetry"

# Inherit the environment in which the package was installed.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

test -x "$PYTHON"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    printf 'Output already exists: %s\n' "$OUTPUT" >&2
    exit 1
fi
for input in "$SEED.win" "$SEED.mmn" "$SEED.amn" "$SEED.eig" "$WAVECAR" "$OUTCAR"; do
    test -r "$input" || { printf 'Input is not readable: %s\n' "$input" >&2; exit 1; }
done

export NUMBA_CACHE_DIR="$PWD/.cache/numba"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
printf 'Job %s on %s\n' "$SLURM_JOB_ID" "${SLURMD_NODENAME:-unknown}"
"$PYTHON" -c 'import sys, vasp_sawf; from importlib.metadata import version; print(sys.version); print(sys.executable); print("vasp-sawf", version("vasp-sawf"), vasp_sawf.__file__); assert sys.version_info[:2] == (3, 13)'

# Reject unsupported symmetry before reading any wavefunction coefficients.
srun --ntasks=1 "$PYTHON" - "$SEED" "$WAVECAR" <<'PYTHON'
import sys
from pathlib import Path
from irrep.spacegroup import SpaceGroup
from vasp_sawf.inputs import read_inputs
from vasp_sawf.symmetry import _cell_from_win, _validate_grey_group, build_symmetry_maps
from vasp_sawf.wavecar import inspect_wavecar

seed, wavecar = map(Path, sys.argv[1:])
header = inspect_wavecar(wavecar)
print(f"Header: NK={header.num_kpoints}, NB={header.num_bands}, RTAG={header.rtag}", flush=True)
print("WAVECAR preflight: 128 logical bytes; no coefficients", flush=True)
bundle = read_inputs(seed, source_nb=header.num_bands)
positions, types, _ = _cell_from_win(Path(f"{seed}.win").read_text(), bundle.lattice)
group = SpaceGroup.from_cell(cell=(bundle.lattice, positions, types),
                            spinor=True, magmom=True, include_TR=True, verbosity=0)
print(f"Candidate structural grey group: {group.name}", flush=True)
translations = sorted({tuple(op.translation.tolist()) for op in group.symmetries})
print(f"Translations: {translations}", flush=True)
_validate_grey_group(group)
build_symmetry_maps(bundle, group)
print("Structural preflight passed; magnetic and source checks follow in extraction", flush=True)
PYTHON

srun --ntasks=1 /usr/bin/time -v "$PYTHON" -m vasp_sawf.extract_symmetry \
    --seed "$SEED" \
    --wavecar "$WAVECAR" \
    --outcar "$OUTCAR" \
    --output "$OUTPUT"
