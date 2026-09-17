# TripDiffusion

A discrete diffusion model for conditional generation of synthetic activity-travel
records, with a hierarchical causal-cascade denoiser (HCD) over three behavioural
streams (activity, space-time, mode).

Given four socio-demographic conditioning variables, the model generates the eight
categorical / ordinal variables that describe one activity-travel entry:

| role | variables |
| --- | --- |
| conditioning (given) | `relation`, `sex`, `age_code`, `job_type` |
| generated | `start_type`, `start_zcode_num`, `start_time_num_6`, `act_num`, `mode_num`, `trip_time_num_6`, `end_type`, `end_zcode_num` |

This repository contains the model code, the baselines it is compared against, and
the scripts that reproduce the experiments reported in the paper.

---

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate tripdiffusion
```

Baselines (CTGAN / TVAE / DATGAN / sequential econometric) need extra packages:

```bash
pip install -r requirements-baselines.txt
```

DATGAN pins TensorFlow 2.8 and usually needs `numpy<2` and `protobuf<=3.20.3`;
see the comments in `requirements-baselines.txt`.

---

## Data

The experiments use the publicly available component of the **2010 Household Travel
Survey for the Seoul Metropolitan Area (SMA), South Korea**, distributed by the
Korea Transport Database (KTDB):

<https://www.ktdb.go.kr/www/newAddPbldataReqstData.do?clTy=2&key=202>

Each row is one activity-travel entry for one individual. The public extract used
here contains 2,678,474 entries from 457,131 individuals in 186,226 households.
`data/description.docx` documents the coding of every column.

Data files are **not** shipped with this repository. Place the raw extract at
`data/full_data_without_redundant_col.csv`, then build the 80:20 train/test split
(partitioned by individual ID, so no individual appears on both sides):

```bash
python scripts/data/prepare_train_test_split.py
```

This writes `data/train_data.csv` and `data/test_data.csv`, which every script below
expects by default.

---

## Quick start

Train the main model (HCD v2) over three seeds and evaluate:

```bash
python scripts/train/run_hcd_v2.py \
    --traindata data/train_data.csv \
    --testdata data/test_data.csv \
    --num_seeds 3
```

Each run writes `model.pth`, `training.log`, `generated_samples.csv` and
`generated_samples_metrics.json` under `exp/<experiment_name>/`.

Common flags: `--epochs` (100), `--batch_size` (64), `--lr` (1e-3), `--T`
(diffusion steps, 10), `--lambda_weight` (1.0), `--num_samples`, `--exp_dir`.

### Cascade structure

The causal structure of the denoiser is controlled by these flags:

| flag | effect |
| --- | --- |
| *(default)* | soft-gated parallel streams — all three streams updated together with learned gates |
| `--hard_stream_cascade` | true sequential cascade; each stream conditions on the already-updated upstream streams |
| `--stream_order` | stream permutation for the hard cascade (`act_st_mode` default, all six permutations supported) |
| `--st_cascade` | two-phase cascade *within* the space-time stream |
| `--st_cascade_chain` | ordering preset for that sub-chain (`loc_then_time` default; see `ST_CASCADE_PRESETS` in [model/HCD_Net_v2.py](model/HCD_Net_v2.py)) |
| `--no_joint_heads` | drop the joint output heads (ablation) |

### Other diffusion variants

```bash
python scripts/train/run.py             # original HCD
python scripts/train/run_transformer.py # plain transformer denoiser
python scripts/train/run_absorbing.py   # absorbing-state discrete diffusion
python scripts/train/run_mlp.py         # MLP denoiser
```

---

## Baselines

```bash
python scripts/baselines/run_tabular_baselines.py \
    --models ctgan tvae datgan \
    --traindata data/train_data.csv \
    --testdata data/test_data.csv \
    --num_samples 10000
```

Available `--models`: `ctgan`, `tvae`, `vae`, `datgan`, `tabddpm`, `tabddpm_tf`,
`ddpm_tf`, `ddpm_mlp`. Outputs go to `exp/baseline/`.

- **TabDDPM** is vendored under [scripts/baselines/third_party/tab_ddpm/](scripts/baselines/third_party/tab_ddpm/)
  (from <https://github.com/yandex-research/tab-ddpm>).
- **Embedding-DDPM** (`ddpm_tf`, `ddpm_mlp`) is a continuous-embedding DDPM over the
  same categorical schema — see [model/EmbeddingDDPM_Net.py](model/EmbeddingDDPM_Net.py).
  It uses a fixed spherical codebook with cosine decoding, which keeps the
  epsilon objective from collapsing the embedding norms.
- **Sequential econometric baseline** — a classical stage-wise generator built from
  multinomial logit and proportional-odds ordered logit models, in
  [scripts/baselines/sequential_econometric_baseline.py](scripts/baselines/sequential_econometric_baseline.py).

---

## Evaluation

Score any generated CSV against the held-out test set:

```bash
python scripts/eval/evaluate_generated_csv.py \
    --generated_csv exp/baseline/CTGAN_gene.csv \
    --train_data data/train_data.csv \
    --test_data data/test_data.csv
```

Reported metrics ([utils/test_utils.py](utils/test_utils.py)):

- **Marginal fidelity** — per-variable KL, normalised JSD, total variation, and
  earth-mover distance for ordinal variables.
- **Joint fidelity** — joint KL and joint Jensen-Shannon divergence.
- **Logical Validity Rate (LVR)** — share of records satisfying activity /
  location / time consistency rules.
- **Behavioural TSTR** — an MNL mode-choice model trained on synthetic data and
  tested on real data, reported as a TSTR/TRTR F1 ratio
  ([utils/mnl_mode_choice.py](utils/mnl_mode_choice.py)).

Figures:

```bash
python scripts/plot/plot_marginal_distributions.py
python scripts/plot/plot_age_gender_joint_comparison.py
```

---

## Reproducing the paper experiments

All experiment runners live under [scripts/experiments/](scripts/experiments/) and
write JSON/CSV summaries next to their artefacts.

| study | command |
| --- | --- |
| Stream ablation (shared-only / soft / hard), paired 10k subsets | `python scripts/experiments/run_stream_ablation_10k.py` |
| Interaction analysis for the above | `python scripts/experiments/analyze_stream_ablation_interaction.py` |
| Minimum stream & ST ordering study | `python scripts/experiments/run_minimum_ordering.py` |
| Cascade-chain sensitivity | `python scripts/experiments/run_cascade_chain_20k.py` |
| Diffusion-step (T) sensitivity | `python scripts/experiments/run_hcd_v2_t_sensitivity.py` |
| Joint-head ablation | `python scripts/experiments/run_hcd_no_joint_heads.py` |
| Sample-size / seed robustness | `python scripts/experiments/run_size_seed_robustness.py` |
| Sequential econometric baseline (20k / full / by size) | `python scripts/experiments/run_sequential_econometric_20k.py` |
| Embedding-DDPM (full / by size) | `python scripts/experiments/run_embedding_ddpm_full_fixed.py` |
| Privacy risk — exact matches, DCR/NNDR, NN membership inference | `python scripts/experiments/run_privacy_assessment.py` |
| Augmentation value — synthetic data as training augmentation for mode choice | `python scripts/experiments/run_augmentation_assessment.py` |

Most runners accept `--seeds`, `--epochs` and an output-root flag; run any of them
with `--help` for the full list.

Re-evaluation helpers for already-trained checkpoints are under
[scripts/eval/](scripts/eval/) (`reeval_*.py`, `generate_and_eval_*_checkpoint.py`).

---

## Repository layout

```
.
├── model/                    # denoiser architectures
│   ├── HCD_Net_v2.py         #   main model: soft/hard stream cascade + ST sub-chain
│   ├── HCD_Net.py            #   original HCD
│   ├── HCD_Net_absorbing.py  #   absorbing-state variant
│   ├── EmbeddingDDPM_Net.py  #   continuous-embedding DDPM
│   ├── Transformer_Net.py
│   └── Net.py
├── utils/                    # training loop, metrics, encodings, multi-seed driver
├── scripts/
│   ├── data/                 # train/test split
│   ├── train/                # training entry points
│   ├── baselines/            # CTGAN / TVAE / DATGAN / TabDDPM / econometric
│   ├── eval/                 # standalone evaluation & re-evaluation
│   ├── experiments/          # ablations, robustness, privacy, augmentation
│   ├── plot/                 # figures
│   └── revision/             # dataset descriptive statistics
├── batch/                    # Windows/Linux parameter-sweep scripts
├── data/                     # not tracked — see "Data" above
└── exp/                      # not tracked — run outputs
```

`data/`, `exp/`, `outputs/` and `figs*/` are gitignored; everything under them is
regenerated by the scripts above.

---

## License

MIT — see [LICENSE](LICENSE).
