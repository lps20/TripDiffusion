#!/bin/bash
# Tuning jobs as a SLURM array.  From the repo root:
#   python scripts/tuning/make_jobs.py          # writes exp/tuning/jobs.tsv (48 jobs)
#   mkdir -p exp/tuning/slurm && sbatch --array=1-48%8 scripts/tuning/tuning_array.slurm
# Afterwards: python scripts/tuning/select_configs.py
#SBATCH -J r2_tune
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o exp/tuning/slurm/%A_%a.out
source ~/.bashrc
conda activate tripdiffusion
PY=python bash scripts/tuning/run_job.sh "${SLURM_ARRAY_TASK_ID}"
