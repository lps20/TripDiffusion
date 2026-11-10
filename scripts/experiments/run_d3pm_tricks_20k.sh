#!/usr/bin/env bash
# Screen Embedding-DDPM-inspired tricks on D3PM-SC3T (20k train subset, seed 42, fixed 100k test subset).
# Base settings = paper robustness_size/20k run.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
OUT=exp/d3pm_tricks_20k
TRAIN=../D3PM_revision/revision_exp/robustness_size/20k/seed_42/train_subset_20000.csv
BASE="--traindata $TRAIN --testdata $OUT/test_subset_100k.csv --epochs 100 --batch_size 500 --lr 0.001 \
  --lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 100 --min_delta 0.0 \
  --batch_sampling shuffle --st_cascade --seed 42 --num_seeds 1"
run() { name=$1; shift
  [ -f "$OUT/$name/generated_samples_metrics.json" ] && { echo "skip $name"; return; }
  echo "=== $name $(date +%T)"; $PY scripts/train/run_d3pm_sc3t.py $BASE --exp_dir "$OUT/$name" "$@" > "$OUT/$name.stdout" 2>&1 || echo "FAILED $name"; }
run D0_base_T10      --T 10
run D1_ctok_T10      --T 10 --cond_tokens
run D2_ctok_cd_T10   --T 10 --cond_tokens --cond_dropout 0.1
run D3_T50_refine5   --T 50 --schedule_refine 5
run D4_T50_ctok_cd   --T 50 --schedule_refine 5 --cond_tokens --cond_dropout 0.1
echo "=== done $(date +%T)"

# Sampling-only variants (no retraining) on trained checkpoints.
ev() { name=$1; src=$2; shift 2; run "$name" --checkpoint "$OUT/$src/model.pth" --eval_only "$@"; }
ev S0_D0_argmax   D0_base_T10    --T 10 --sample_final argmax
ev S1_D0_tau09    D0_base_T10    --T 10 --sample_temperature 0.9
ev S2_D2_argmax   D2_ctok_cd_T10 --T 10 --cond_tokens --cond_dropout 0.1 --sample_final argmax
ev S3_D2_tau09    D2_ctok_cd_T10 --T 10 --cond_tokens --cond_dropout 0.1 --sample_temperature 0.9
ev S4_D2_tau08    D2_ctok_cd_T10 --T 10 --cond_tokens --cond_dropout 0.1 --sample_temperature 0.8
ev S5_D2_tau09_am D2_ctok_cd_T10 --T 10 --cond_tokens --cond_dropout 0.1 --sample_temperature 0.9 --sample_final argmax
ev S6_D4_argmax   D4_T50_ctok_cd --T 50 --schedule_refine 5 --cond_tokens --cond_dropout 0.1 --sample_final argmax
echo "=== sampling variants done $(date +%T)"

# Absorbing-state (masking) D3PM-SC3T.
run A0_abs_T10         --T 10 --absorbing
run A1_abs_T10_ctok_cd --T 10 --absorbing --cond_tokens --cond_dropout 0.1
run A2_abs_T50         --T 50 --absorbing
run A3_abs_T50_ctok_cd --T 50 --absorbing --cond_tokens --cond_dropout 0.1
for src in A2_abs_T50 A3_abs_T50_ctok_cd; do
  extra="--T 50 --absorbing"; [ "$src" = A3_abs_T50_ctok_cd ] && extra="$extra --cond_tokens --cond_dropout 0.1"
  ev "${src}__cascade"    $src $extra --sample_unmask cascade
  ev "${src}__confidence" $src $extra --sample_unmask confidence
  ev "${src}__tau09"      $src $extra --sample_temperature 0.9
done
echo "=== absorbing done $(date +%T)"
