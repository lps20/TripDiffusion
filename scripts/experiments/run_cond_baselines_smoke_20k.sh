#!/usr/bin/env bash
# Conditional baselines on the 20k subset (seed 42), paper budget (4000 steps ~ 100 epochs), evaluated 1:1 on val 50k.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
OUT=exp/cond_baselines_smoke_20k
TR=../D3PM_revision/revision_exp/robustness_size/20k/seed_42/train_subset_20000.csv
COMMON="--traindata $TR --testdata data/val_subset_50k.csv --max_steps 4000 --batch_size 500 --seed 42 --num_seeds 1 --no_save_models"
run() { name=$1; shift; [ -f $OUT/$name/baseline_metrics.csv ] && return; mkdir -p $OUT/$name
  echo "=== $name $(date +%T)"; $PY scripts/baselines/run_tabular_baselines.py $COMMON --output_dir $OUT/$name "$@" > $OUT/$name.log 2>&1 || echo "FAILED $name"; }
run ctgan_cond   --models ctgan_cond
run tvae_cond    --models tvae_cond
run tabddpm_cond --models tabddpm --tabddpm_conditional --lr 1e-3
run datgan_cond  --models datgan --datgan_conditional
echo "=== done $(date +%T)"
