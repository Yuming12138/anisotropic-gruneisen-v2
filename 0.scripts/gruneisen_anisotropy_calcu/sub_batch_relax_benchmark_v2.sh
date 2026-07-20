#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J agv2_br
#SBATCH -t 02:00:00

set -euo pipefail

BASE=/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation
CODE=${BASE}/anisotropic-gruneisen-v2/0.scripts/gruneisen_anisotropy_calcu
RESULTS=${BASE}/run_20260717_batch1024_all10/results
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
PYTHON=/home/chenguangming/miniconda3/envs/mattersim125/bin/python
MODEL=/home/chenguangming/2.model/mattersim-v1.0.0-1M.pth
MAX_NATOMS_PER_BATCH=${MAX_NATOMS_PER_BATCH:-2048}

IDS=(0106 0003 0129 0167)
MATERIAL_ID=${IDS[${TASK_ID}]}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/results"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

"${PYTHON}" "${CODE}/benchmark_batch_relax_v2.py" \
  --elastic-poscar "${RESULTS}/${MATERIAL_ID}/elastic/POSCAR" \
  --model "${MODEL}" \
  --output "${OUTPUT_ROOT}/results/${MATERIAL_ID}.json" \
  --device cuda \
  --dtype float64 \
  --strain 0.005 \
  --fmax 0.001 \
  --max-steps 1000 \
  --maxstep 0.1 \
  --max-natoms-per-batch "${MAX_NATOMS_PER_BATCH}"
