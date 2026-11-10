#!/usr/bin/env bash
# Validation-set tuning of absorbing D3PM-SC3T on the 20k subset (seed 42).
# Grid: T in {20,50,100} x temperature in {0.8,0.9,1.0} x unmask order in {random,cascade}.
# Evaluated on data/val_subset_50k.csv (individual-disjoint from test and from the rest of train).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
OUT=exp/d3pm_abs_tune_20k
TRAIN=../D3PM_revision/revision_exp/robustness_size/20k/seed_42/train_subset_20000.csv
mkdir -p $OUT
COMMON="--traindata $TRAIN --testdata data/val_subset_50k.csv --batch_size 500 --lr 0.001 \
  --lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 100 --min_delta 0.0 \
  --batch_sampling shuffle --st_cascade --seed 42 --num_seeds 1 --absorbing"
run() { name=$1; shift
  [ -f "$OUT/$name/generated_samples_metrics.json" ] && { echo "skip $name"; return; }
  echo "=== $name $(date +%T)"; $PY scripts/train/run_d3pm_sc3t.py $COMMON --exp_dir "$OUT/$name" "$@" > "$OUT/$name.stdout" 2>&1 || echo "FAILED $name"; }
# Train (T=50 checkpoint reused from the screen: identical training settings).
mkdir -p $OUT/ckpt_T50; cp -n exp/d3pm_tricks_20k/A2_abs_T50/model.pth $OUT/ckpt_T50/model.pth
for T in 20 100; do run ckpt_T$T --T $T --epochs 100; done
for T in 20 50 100; do for tau in 0.8 0.9 1.0; do for u in random cascade; do
  run "T${T}_tau${tau}_${u}" --T $T --epochs 100 --checkpoint $OUT/ckpt_T$T/model.pth --eval_only \
      --sample_temperature $tau --sample_unmask $u
done; done; done
echo "=== done $(date +%T)"
