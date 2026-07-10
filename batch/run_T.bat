@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0\.."

set TRAIN=data\train_data.csv
set TEST=data\test_data.csv

for %%T in (10 50 100 500 1000) do (
    set EXP=exp\T_%%T
    echo Running: T=%%T
    mkdir "!EXP!"
    python scripts\train\run.py --traindata !TRAIN! --testdata !TEST! --T %%T --exp_dir "!EXP!"
)
