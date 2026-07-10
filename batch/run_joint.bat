@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set TRAIN=data\train_data.csv
set TEST=data\test_data.csv

for %%L in (0.5 1.0 2.0) do (
    set EXP=exp\joint_%%L
    echo Running: lambda_weight=%%L
    mkdir "!EXP!"
    python scripts\train\run.py --traindata !TRAIN! --testdata !TEST! --lambda_joint %%L --exp_dir "!EXP!"
)
