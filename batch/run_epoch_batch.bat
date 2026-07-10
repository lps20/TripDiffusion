@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set TRAIN=data\train_data.csv
set TEST=data\test_data.csv

for %%E in (100) do (
    for %%B in (64 128 256) do (
        set EXP=exp\epochs_%%E_batch_%%B
        echo Running: Epochs=%%E, Batch=%%B
        mkdir "!EXP!"
        python scripts\train\run.py --traindata !TRAIN! --testdata !TEST! --epochs %%E --batch_size %%B --exp_dir "!EXP!"
    )
)
