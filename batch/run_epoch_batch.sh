#!/bin/bash
cd "$(dirname "$0")/.."

TRAIN="data/train_data.csv"
TEST="data/test_data.csv"

for EPOCHS in 50 100 200 500; do
  for BATCH in 64 128 256; do
    EXP_DIR="exp/epochs_${EPOCHS}_batch_${BATCH}"
    echo "Running: Epochs=${EPOCHS}, Batch Size=${BATCH}"

    mkdir -p "$EXP_DIR"

    python scripts/train/run.py \
      --traindata "$TRAIN" \
      --testdata "$TEST" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" \
      --exp_dir "$EXP_DIR"
  done
done

echo "All experiments completed!"
