#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH -p standard
#SBATCH -J agv2_mesh
#SBATCH -t 01:00:00

set -euo pipefail

BASE=/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation
CODE=${BASE}/anisotropic-gruneisen-v2/0.scripts/gruneisen_anisotropy_calcu
PYTHON=/home/chenguangming/miniconda3/envs/mattersim125/bin/python
SOURCE_RESULT=${SOURCE_RESULT:?Set SOURCE_RESULT}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
MESHES=(12 16 20 24 28 32)
MESH=${MESHES[${TASK_ID}]}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export PYTHONPATH=${CODE}:${PYTHONPATH:-}

mkdir -p "${OUTPUT_ROOT}/mesh_${MESH}"
"${PYTHON}" "${CODE}/benchmark_mesh_convergence_v2.py" \
  --source-result "${SOURCE_RESULT}" \
  --mesh "${MESH}" "${MESH}" "${MESH}" \
  --output-dir "${OUTPUT_ROOT}/mesh_${MESH}"
