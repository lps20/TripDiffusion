#!/usr/bin/env bash
# Full-data check (seed 42, full test set): absorbing D3PM-SC3T (T=100, random unmasking, tau 0.9 / 0.8)
# and fixed Embedding-DDPM C5, same training budget as the paper's full-data runs (100 epochs, batch 500, lr 1e-3).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
OUT=exp/fulldata_abs_c5
mkdir -p $OUT
COMMON="--traindata data/train_data.csv --testdata data/test_data.csv --epochs 100 --batch_size 500 --lr 0.001 \
  --lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 100 --min_delta 0.0 \
  --batch_sampling shuffle --st_cascade --seed 42 --num_seeds 1 --absorbing --T 100 --sample_unmask random"
echo "=== ABS tau0.9 train+sample $(date +%T)"
[ -f $OUT/ABS_tau0.9/generated_samples_metrics.json ] || $PY scripts/train/run_d3pm_sc3t.py $COMMON --exp_dir $OUT/ABS_tau0.9 --sample_temperature 0.9 > $OUT/ABS_tau0.9.stdout 2>&1 || echo FAILED
echo "=== ABS tau0.8 sample $(date +%T)"
[ -f $OUT/ABS_tau0.8/generated_samples_metrics.json ] || $PY scripts/train/run_d3pm_sc3t.py $COMMON --exp_dir $OUT/ABS_tau0.8 --sample_temperature 0.8 --checkpoint $OUT/ABS_tau0.9/model.pth --eval_only > $OUT/ABS_tau0.8.stdout 2>&1 || echo FAILED
echo "=== C5 $(date +%T)"
$PY scripts/experiments/screen_embedding_ddpm.py --configs C5_x0_d16_T50_shift2_ctok --seed 42 --train data/train_data.csv \
   --eval_n 10000000 --reference_csv ../D3PM_revision/revision_exp/baselines/hcd/seed_42/generated_samples.csv --out $OUT/C5 > $OUT/C5.stdout 2>&1 || echo FAILED
echo "=== done $(date +%T)"
