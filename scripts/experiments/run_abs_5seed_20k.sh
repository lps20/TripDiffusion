#!/usr/bin/env bash
# Absorbing D3PM-SC3T, config selected on validation (T=100, random unmasking, tau=0.8; tau=0.9 as sensitivity),
# trained on each of the five 20k subsets and evaluated on the fixed 100k test subset.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
OUT=exp/d3pm_abs_5seed_20k
COMMON="--testdata exp/d3pm_tricks_20k/test_subset_100k.csv --epochs 100 --batch_size 500 --lr 0.001 \
  --lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 100 --min_delta 0.0 \
  --batch_sampling shuffle --st_cascade --num_seeds 1 --absorbing --T 100 --sample_unmask random"
for s in 42 43 44 45 46; do
  TR=../D3PM_revision/revision_exp/robustness_size/20k/seed_$s/train_subset_20000.csv
  d=$OUT/ABS_tau0.8/seed_$s
  [ -f $d/generated_samples_metrics.json ] || { echo "=== train+tau0.8 seed $s $(date +%T)";
    $PY scripts/train/run_d3pm_sc3t.py $COMMON --traindata $TR --seed $s --exp_dir $d --sample_temperature 0.8 > $d.stdout 2>&1 || echo FAILED; }
  e=$OUT/ABS_tau0.9/seed_$s
  [ -f $e/generated_samples_metrics.json ] || { echo "=== tau0.9 seed $s $(date +%T)"; mkdir -p $(dirname $e);
    $PY scripts/train/run_d3pm_sc3t.py $COMMON --traindata $TR --seed $s --exp_dir $e --sample_temperature 0.9 \
       --checkpoint $d/model.pth --eval_only > $e.stdout 2>&1 || echo FAILED; }
done
echo "=== done $(date +%T)"
