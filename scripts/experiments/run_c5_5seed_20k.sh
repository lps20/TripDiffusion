#!/usr/bin/env bash
# Fixed Embedding-DDPM (C5) on the five 20k subsets, evaluated on the fixed 100k test subset.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/d/conda/envs/tripdiffusion/python.exe}
for s in 42 43 44 45 46; do
  R=../D3PM_revision/revision_exp/robustness_size/20k/seed_$s
  $PY scripts/experiments/screen_embedding_ddpm.py --configs C5_x0_d16_T50_shift2_ctok --seed $s \
     --train $R/train_subset_20000.csv --reference_csv $R/hcd/generated_samples.csv \
     --out exp/d3pm_abs_5seed_20k/C5/seed_$s || echo "FAILED C5 seed $s"
done
