# Soft Hierarchical Diffusion for Conditional Tabular Data Generation

### Application to Travel Survey Entries

**D3PM-SC3T** is a discrete denoising diffusion model for conditional generation of
synthetic activity-travel records. Its denoiser couples a shared transformer with
*soft-causal* adapter blocks over three behavioural streams — activity, space-time,
and mode — so the model can express a behavioural ordering without hard-wiring one.

Given four socio-demographic conditioning variables, the model generates the eight
categorical / ordinal variables that describe one activity-travel entry:

| role | variables |
| --- | --- |
| conditioning (given) | `relation`, `sex`, `age_code`, `job_type` |
| generated | `start_type`, `start_zcode_num`, `start_time_num_6`, `act_num`, `mode_num`, `trip_time_num_6`, `end_type`, `end_zcode_num` |

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

## Training

Train D3PM-SC3T over three seeds and evaluate:

```bash
python scripts/train/run_d3pm_sc3t.py \
    --traindata data/train_data.csv \
    --testdata data/test_data.csv \
    --num_seeds 3
```

Each run writes `model.pth`, `training.log`, `generated_samples.csv` and
`generated_samples_metrics.json` under `exp/<experiment_name>/`.

Common flags: `--epochs` (100), `--batch_size` (64), `--lr` (1e-3), `--T`
(diffusion steps, 10), `--lambda_weight` (1.0), `--num_samples`, `--exp_dir`.
Run with `--help` for the full list.

### Cascade structure

The causal structure of the denoiser is set by these flags:

| flag | effect |
| --- | --- |
| *(default)* | soft-gated parallel streams — all three streams updated together with learned gates |
| `--hard_stream_cascade` | true sequential cascade; each stream conditions on the already-updated upstream streams |
| `--stream_order` | stream permutation for the hard cascade (`act_st_mode` default; all six permutations supported) |
| `--st_cascade` | two-phase cascade *within* the space-time stream |
| `--st_cascade_chain` | ordering preset for that sub-chain (`loc_then_time` default; see `ST_CASCADE_PRESETS` in [model/D3PM_SC3T_Net.py](model/D3PM_SC3T_Net.py)) |
| `--no_joint_heads` | drop the joint output heads |
| `--freeze_gates` / `--gate_init_*` | fix the soft gates at their initial values, to isolate the effect of soft gating |

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
  It uses a fixed spherical codebook with cosine decoding, which keeps the epsilon
  objective from collapsing the embedding norms.
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

To regenerate samples and metrics from checkpoints you have already trained, without
retraining, use the re-evaluation helpers in [scripts/eval/](scripts/eval/):

```bash
python scripts/eval/reeval_revision_d3pm_sc3t.py          # D3PM-SC3T checkpoints
python scripts/eval/reeval_revision_baselines.py          # CTGAN / DDPM-TF / TabDDPM
python scripts/eval/reeval_revision_tvae_datgan.py        # TVAE / DATGAN
python scripts/eval/reeval_joint_sampling_d3pm_sc3t.py    # joint-pair Gibbs sampling
```

---

## Repository layout

```
.
├── model/
│   ├── D3PM_SC3T_Net.py      # the model: shared transformer + soft-causal streams
│   └── EmbeddingDDPM_Net.py  # continuous-embedding DDPM baseline
├── utils/
│   ├── train_utils.py        # diffusion training loop
│   ├── test_utils.py         # fidelity, validity, and TSTR metrics
│   ├── mnl_mode_choice.py    # MNL mode-choice model used for behavioural TSTR
│   ├── data_encoding.py      # category normalisation and encoding
│   └── multi_seed.py         # multi-seed driver and aggregation
├── scripts/
│   ├── data/                 # train/test split
│   ├── train/                # training entry point
│   ├── baselines/            # CTGAN / TVAE / DATGAN / TabDDPM / econometric
│   └── eval/                 # evaluation and checkpoint re-evaluation
├── data/                     # not tracked — see "Data" above
└── exp/                      # not tracked — run outputs
```

`data/`, `exp/`, `outputs/` and `figs*/` are gitignored; everything under them is
produced by the scripts above.

---

## License

MIT — see [LICENSE](LICENSE).
