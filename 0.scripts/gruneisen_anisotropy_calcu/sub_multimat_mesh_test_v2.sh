#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J agv2_mtest
#SBATCH -t 02:00:00

set -euo pipefail

BASE=/home/chenguangming/3.projects/NTE_PTE_342_elastic_recalculation
CODE=${BASE}/anisotropic-gruneisen-v2/0.scripts/gruneisen_anisotropy_calcu
PYTHON=/home/chenguangming/miniconda3/envs/mattersim125/bin/python
MODEL=/home/chenguangming/2.model/mattersim-v1.0.0-1M.pth
TASK_ID=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
RESULT_SUBDIR=${RESULT_SUBDIR:-agv2_br_m6_test}
STRAIN=${STRAIN:-0.005}
IDS=(0236 0171 0003 0091 0157 0292 0223)
SUPERCELLS=("3 3 3" "2 2 3" "3 2 2" "2 2 2" "4 4 3" "2 2 2" "3 3 2")
MATERIAL_ID=${IDS[${TASK_ID}]}
read -r SC1 SC2 SC3 <<< "${SUPERCELLS[${TASK_ID}]}"
MATERIAL=${BASE}/run_20260717_batch1024_all10/results/${MATERIAL_ID}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-20}
export PYTHONPATH=${CODE}:${PYTHONPATH:-}

rm -rf "${MATERIAL}/${RESULT_SUBDIR}"
"${PYTHON}" "${CODE}/run_gruneisen_thermal_expansion_v2.py" \
  --material-dir "${MATERIAL}" \
  --result-subdir "${RESULT_SUBDIR}" \
  --model "${MODEL}" \
  --device cuda \
  --dtype float64 \
  --supercell "${SC1}" "${SC2}" "${SC3}" \
  --mesh 6 6 6 \
  --strain "${STRAIN}" \
  --displacement 0.01 \
  --fmax 0.001 \
  --max-steps 1000 \
  --batch-relax \
  --batch-relax-atom-cap 1024
