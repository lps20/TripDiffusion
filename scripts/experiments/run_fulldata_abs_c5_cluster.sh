#!/usr/bin/env bash
# Cluster version of run_fulldata_abs_c5.sh (seed 42, full train / full test).
#   1) absorbing D3PM-SC3T (T=100, random unmasking): resume from RESUME_CKPT at RESUME_EPOCH if given,
#      then sample with tau 0.9
#   2) same checkpoint, sample with tau 0.8
#   3) fixed Embedding-DDPM C5 on full data
# Usage (from repo root):
#   PY=python RESUME_CKPT=exp/fulldata_abs_c5/resume_epoch65.pth RESUME_EPOCH=65 RESUME_BEST=1.9000 \
#     bash scripts/experiments/run_fulldata_abs_c5_cluster.sh
# Steps whose metrics JSON already exists are skipped, so the script can be re-submitted after a preemption.
# Note: model.pth stores weights only, so Adam moments restart at the resume epoch.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=exp/fulldata_abs_c5
mkdir -p $OUT
COMMON="--traindata data/train_data.csv --testdata data/test_data.csv --epochs 100 --batch_size 500 --lr 0.001 \
  --lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 100 --min_delta 0.0 \
  --batch_sampling shuffle --st_cascade --seed 42 --num_seeds 1 --absorbing --T 100 --sample_unmask random"

if [ ! -f $OUT/ABS_tau0.9/generated_samples_metrics.json ]; then
  RESUME=""
  if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME="--checkpoint $RESUME_CKPT --resume_epoch ${RESUME_EPOCH:?} --resume_best_loss ${RESUME_BEST:?}"
  fi
  echo "=== ABS tau0.9 train+sample $(date +%T) $RESUME"
  $PY scripts/train/run_d3pm_sc3t.py $COMMON --exp_dir $OUT/ABS_tau0.9 --sample_temperature 0.9 $RESUME \
    > $OUT/ABS_tau0.9.stdout 2>&1 || { echo FAILED ABS_tau0.9; exit 1; }
fi
if [ ! -f $OUT/ABS_tau0.8/generated_samples_metrics.json ]; then
  echo "=== ABS tau0.8 sample $(date +%T)"
  $PY scripts/train/run_d3pm_sc3t.py $COMMON --exp_dir $OUT/ABS_tau0.8 --sample_temperature 0.8 \
    --checkpoint $OUT/ABS_tau0.9/model.pth --eval_only > $OUT/ABS_tau0.8.stdout 2>&1 || echo FAILED ABS_tau0.8
fi
echo "=== C5 $(date +%T)"
# The REF row needs ../D3PM_revision/...; it is skipped automatically if that file is absent.
$PY scripts/experiments/screen_embedding_ddpm.py --configs C5_x0_d16_T50_shift2_ctok --seed 42 --train data/train_data.csv \
   --eval_n 10000000 --reference_csv ../D3PM_revision/revision_exp/baselines/hcd/seed_42/generated_samples.csv \
   --out $OUT/C5 > $OUT/C5.stdout 2>&1 || echo FAILED C5
echo "=== done $(date +%T)"
