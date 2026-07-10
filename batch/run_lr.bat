@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set TRAIN=data\train_data.csv
set TEST=data\test_data.csv

for %%L in (1e-4 1e-3 1e-2 1e-1) do (
    set EXP=exp\lr_%%L
    echo Running: learning_rate=%%L
    mkdir "!EXP!"
    python scripts\train\run.py --traindata !TRAIN! --testdata !TEST! --lr %%L --exp_dir "!EXP!"
)
