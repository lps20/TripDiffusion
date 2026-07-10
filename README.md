# TripDiffusionModel

A diffusion-based discrete generative model for synthetic trip generation.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate tripdiffusion
```

---

## Running the Model

Single run (default settings):

```bash
python scripts/train/run.py --traindata data/train_data.csv --testdata data/test_data.csv
```

HCD v2 model with multi-seed support:

```bash
python scripts/train/run_hcd_v2.py --traindata data/train_data.csv --testdata data/test_data.csv --num_seeds 3
```

Other diffusion variants:

```bash
python scripts/train/run_transformer.py
python scripts/train/run_absorbing.py
```

---

## Baselines: CTGAN / DATGAN / VAE

Install optional baseline dependencies:

```bash
pip install -r requirements-baselines.txt
```

Run baseline generation and evaluation:

```bash
python scripts/baselines/run_tabular_baselines.py --models ctgan datgan --traindata data/train_data.csv --testdata data/test_data.csv --num_samples 10000
```

Outputs are saved to `exp/baseline/`.

---

## Evaluation & Plotting

Evaluate a generated CSV:

```bash
python scripts/eval/evaluate_generated_csv.py --generated_csv exp/baseline/CTGAN_gene.csv --train_data data/train_data.csv --test_data data/test_data.csv
```

Plot marginal / joint distributions:

```bash
python scripts/plot/plot_marginal_distributions.py
python scripts/plot/plot_age_gender_joint_comparison.py
```

---

## Data Preparation

Regenerate train/test split (80:20 by ID):

```bash
python scripts/data/prepare_train_test_split.py
```

---

## Batch Experiments

Parameter sweeps live under `batch/`:

```bash
batch\run_epoch_batch.bat
batch\run_T.bat
batch\run_lambda.bat
batch\run_lr.bat
batch\run_joint.bat
```

Each run saves `model.pth`, `training.log`, and `generated_samples.csv` under `exp/<experiment_name>/`.

---

## Common Parameters

- `--epochs` (default: 100)
- `--batch_size` (default: 64)
- `--lr` (default: 1e-3)
- `--lambda_weight` (default: 1.0)
- `--T` (diffusion steps, default: 10)
- `--num_samples` (per-cluster generation, default: 100)
- `--exp_dir` (optional custom output folder)

---

## Project Structure

```
project/
├── batch/                  # Windows/Linux batch sweep scripts
├── data/                   # train/test CSVs (gitignored; regenerate locally)
├── model/                  # network definitions
├── scripts/
│   ├── baselines/          # CTGAN, DATGAN, VAE runners
│   ├── data/               # data preparation
│   ├── eval/               # standalone evaluation
│   ├── experiments/        # sensitivity / robustness sweeps
│   ├── plot/               # figure generation
│   └── train/              # diffusion training entry points
├── utils/                  # training, evaluation, metrics
├── exp/                    # experiment outputs (gitignored)
├── requirements.txt
└── environment.yml
```
