#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J agv2_reg
#SBATCH -t 02:00:00

set -euo pipefail

BASE=/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation
CODE=${BASE}/anisotropic-gruneisen-v2/0.scripts/gruneisen_anisotropy_calcu
ELASTIC_RESULTS=${BASE}/run_20260717_batch1024_all10/results
RELAX_RESULTS=${BASE}/bench_agv2_br_20260717/results
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
PYTHON=/home/chenguangming/miniconda3/envs/mattersim125/bin/python
MODEL=/home/chenguangming/2.model/mattersim-v1.0.0-1M.pth

IDS=(0106 0003)
SUPERCELLS=("2 2 2" "3 2 2")
MATERIAL_ID=${IDS[${TASK_ID}]}
read -r SC1 SC2 SC3 <<< "${SUPERCELLS[${TASK_ID}]}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/${MATERIAL_ID}"

ARGS=(
  "${PYTHON}" "${CODE}/benchmark_full_regression_v2.py"
  --material-id "${MATERIAL_ID}"
  --root-poscar "${BASE}/materials/${MATERIAL_ID}/POSCAR"
  --elastic-poscar "${ELASTIC_RESULTS}/${MATERIAL_ID}/elastic/POSCAR"
  --elastic-tensor "${ELASTIC_RESULTS}/${MATERIAL_ID}/elastic/ELASTIC_TENSOR"
  --relax-benchmark-json "${RELAX_RESULTS}/${MATERIAL_ID}.json"
  --model "${MODEL}"
  --output-dir "${OUTPUT_ROOT}/${MATERIAL_ID}"
  --device cuda
  --dtype float64
  --supercell "${SC1}" "${SC2}" "${SC3}"
  --mesh 6 6 6
  --strain 0.005
  --displacement 0.01
  --batch-atom-cap 1024
)
if [[ ${ISOLATE:-0} == 1 ]]; then
  ARGS+=(--isolate)
fi
"${ARGS[@]}"
