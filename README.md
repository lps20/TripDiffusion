# TripDiffusionModel

A diffusion-based discrete generative model for synthetic trip generation.

---

## Environment Setup

### Install Required Packages

```bash
pip install -r requirements.txt
```

Or, if you use conda, see Conda Setup.

## Running the Model
Single Run (default settings)

```bash
python run.py --traindata data/train_data.csv --testdata data/test_data.csv
```

## Baselines: CTGAN / DATGAN

Install optional baseline dependencies:

```bash
pip install -r requirements-baselines.txt
```

Run baseline generation and evaluation:

```bash
python run_tabular_baselines.py --models ctgan datgan --traindata data/train_data.csv --testdata data/test_data.csv --num_samples 10000
```

Outputs are saved to `exp/baseline/`:
- `CTGAN_gene.csv`
- `DATGAN_gene.csv`
- `CTGAN_metrics.json`
- `DATGAN_metrics.json`
- `baseline_metrics.csv`

### Parameters you can modify:

- `--epochs` (default: 100)
- `--batch_size` (default: 64)
- `--lr` (default: 1e-3)
- `--lambda_weight` (default: 1.0)
- `--T` (diffusion steps, default: 100)
- `--num_samples` (per-cluster generation, default: 100)
- `--exp_dir` (optional: custom folder to save model/logs)

---

##  Batch Experiments (Windows)

You can test different parameter combinations using `.bat` scripts:

```bash
run_epochs_batch.bat     # (Peisen) Test various epochs & batch sizes
run_T.bat                # (Churong) Test different diffusion step T values
run_lambda.bat           # (Ivan) Test different lambda weights
run_lr.bat               # (Yingnan) Test different learning rates
```

Each run will output:
- `model.pth`: Trained model
- `training.log`: Log file
- `generated_samples.csv`: Generated trip samples
Under:
```bash
exp/<experiment_name>/
```

---

## 💡 Conda Setup

If you prefer using Conda:

```bash
conda env create -f environment.yml
conda activate tripdiffusion
```

---

##  Project Structure

```
project/
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
├── run.py
├── train_utils.py
├── test_utils.py
├── Net.py
├── requirements.txt
├── environment.yml
├── run_epochs_batch.bat
├── run_T.bat
├── run_lambda.bat
├── run_lr.bat
└── exp/
```

