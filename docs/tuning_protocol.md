# Hyper-parameter tuning protocol (R2 revision)

This protocol is fixed before any tuning run is evaluated. All models (ours and baselines) follow it.

## Data
- Train / test: the individual-disjoint 80/20 split of the paper (`data/train_data.csv`, `data/test_data.csv`).
- Validation: 5% of the *training individuals* (18,285 persons, 106,796 trips) held out by
  `scripts/data/make_validation_split.py`; `data/val_subset_50k.csv` (fixed 50k rows) is the tuning target.
- Tuning training set: `data/tune_train_200k.csv`, a fixed 200k-row sample of the training data with the
  validation individuals removed (`scripts/data/make_tuning_subset.py`).
- The test set is used only once, for the final runs.

## Conditional setting
Every model receives the four demographics (relation, sex, age, job) as an external condition and generates one
trip per validation (later: test) record, 1:1. Baselines that were unconditional in R1 are replaced by
conditional versions: C-CTGAN and C-TVAE (`scripts/baselines/conditional_ctgan_tvae.py`; condition one-hot
concatenated to generator/discriminator and encoder/decoder inputs), C-TabDDPM (condition embedding added to the
time embedding), C-DATGAN (demographics as `conditional_inputs`), the fixed Embedding-DDPM (per-field condition
tokens) and the sequential econometric generator (conditional by construction).

## Budget
- Identical training budget for every configuration: **40,000 gradient steps at batch 500**
  (= 100 epochs of the 200k tuning set; C-CTGAN counts generator updates, one critic step each).
- One seed (42) per configuration; at most **12 trained configurations per model**. Sampling-only variants of a
  trained model (D3PM-SC3T temperature) do not count towards the 12.
- No early stopping on the test set; diffusion models keep the lowest-training-loss checkpoint, the others the
  final one.

## Search spaces
| Model | Grid (trained configs) |
|---|---|
| D3PM-SC3T (absorbing) | T {50, 100} x lr {3e-4, 1e-3} x d_model {128, 256} (8); sampling temperature {0.8, 0.9, 1.0} |
| Embedding-DDPM (x0, condition tokens, cond. dropout 0.1) | embed dim {16, 32} x log-SNR shift {1, 2, 4} x T {50, 100} (12) |
| C-TabDDPM | T {100, 1000} x lr {2e-4, 1e-3} x MLP {256-512-512-256, 512x4} (8) |
| C-CTGAN | embedding {128, 256} x G/D dims {(256,256), (512,512)} x lr {2e-4, 1e-3} (8), pac 10 |
| C-TVAE | latent {64, 128} x dims {(128,128), (256,256)} x loss factor {1, 2} (8) |
| C-DATGAN | time variables {continuous, categorical} x batch {500, 1000} (4) |
| Sequential econometric | specification of the paper (no tuning; it has no free capacity hyper-parameters) |

## Selection rule
All candidates are re-evaluated with one evaluator (`scripts/tuning/select_configs.py`) on the validation subset;
the MNL reference model is fitted on the tuning training set with one shared scaler.
1. **Failed** = missing output, any selection metric undefined, or MNL estimation failure. If every
   configuration of a model fails only because the MNL cannot be estimated (e.g. mode classes missing from the
   synthetic data), that model is ranked on the five non-MNL metrics instead.
2. **Diversity guard**: |share of distinct trip 8-tuples - share in real validation data| <= 0.10. If no
   configuration of a model passes, all non-failed configurations are ranked.
3. **Score** = mean within-model rank over seven metrics: joint JSD, mean entropy-normalised marginal JSD,
   mean ordinal EMD, mean cross-stream pairwise JSD, logical validity rate, MNL coefficient cosine,
   MNL AME cosine. Lowest score wins; ties broken by joint JSD.

## Final runs
The selected configuration of each model is retrained on the full training data with seeds 42-46 at the
paper's full-data budget (100 epochs at batch 500 for every model) and evaluated once on the test set.
Reported: mean +- SD and paired 95% t-intervals against D3PM-SC3T. The appendix lists every selected
configuration, its parameter count, training time and GPU.
