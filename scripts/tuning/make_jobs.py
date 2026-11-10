"""Generate the hyper-parameter tuning job list (one shell command per line).

Protocol (docs/tuning_protocol.md):
  * train on data/tune_train_200k.csv (validation individuals removed), seed 42
  * identical budget for every model: 40,000 gradient steps at batch 500 (= 100 epochs of the subset)
  * generate 1:1 for the conditions of data/val_subset_50k.csv; the test set is never touched
  * <= 12 trained configurations per model (sampling-only variants are free)
Output: exp/tuning/jobs.tsv  (job_id, model, cfg_id, params, outputs, command)
        each `outputs` entry is variant=path_to_generated_csv (row-aligned with the val subset)
Run one line:  bash scripts/tuning/run_job.sh <job_id>   (SLURM: --array=1-N)
"""
import itertools
import json
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import os

# Environment overrides exist only for the pipeline smoke test (TUNE_OUT=exp/tuning_smoke ...).
OUT = os.environ.get("TUNE_OUT", "exp/tuning")
TRAIN = os.environ.get("TUNE_TRAIN", "data/tune_train_200k.csv")
VAL = os.environ.get("TUNE_VAL", "data/val_subset_50k.csv")
STEPS = int(os.environ.get("TUNE_STEPS", 40000))
BATCH = 500
EPOCHS = int(os.environ.get("TUNE_EPOCHS", 100))  # 200k rows / 500 = 400 steps per epoch -> 100 epochs = 40k steps
PY = "${PY:-python}"

jobs = []


def add(model, cfg_id, params, outputs, cmd):
    jobs.append(dict(model=model, cfg_id=cfg_id, params=json.dumps(params, sort_keys=True),
                     outputs=";".join(f"{k}={v}" for k, v in outputs.items()), command=cmd))


# --- absorbing D3PM-SC3T: train once at tau=1.0, then sampling-only tau 0.9 / 0.8 -----------------
D3PM_COMMON = (f"--traindata {TRAIN} --testdata {VAL} --epochs {EPOCHS} --batch_size {BATCH} "
               "--lambda_weight 1.0 --lambda_joint 0.0 --joint_pairs [] --patience 1000 --min_delta 0.0 "
               "--batch_sampling shuffle --st_cascade --seed 42 --num_seeds 1 --absorbing --sample_unmask random")
for i, (T, lr, d) in enumerate(itertools.product([50, 100], [3e-4, 1e-3], [128, 256]), 1):
    cid = f"D{i:02d}"
    base = f"{OUT}/d3pm_abs/{cid}"
    arch = f"--T {T} --lr {lr} --d_model {d}"
    cmds = [f"{PY} scripts/train/run_d3pm_sc3t.py {D3PM_COMMON} {arch} --exp_dir {base}/tau1.0 --sample_temperature 1.0"]
    outs = {"tau1.0": f"{base}/tau1.0/generated_samples.csv"}
    for tau in ("0.9", "0.8"):
        cmds.append(f"{PY} scripts/train/run_d3pm_sc3t.py {D3PM_COMMON} {arch} --exp_dir {base}/tau{tau} "
                    f"--sample_temperature {tau} --checkpoint {base}/tau1.0/model.pth --eval_only")
        outs[f"tau{tau}"] = f"{base}/tau{tau}/generated_samples.csv"
    add("d3pm_abs", cid, dict(T=T, lr=lr, d_model=d), outs, " && ".join(cmds))

# --- fixed Embedding-DDPM (x0 parameterization, per-field condition tokens) ----------------------
for i, (de, shift, T) in enumerate(itertools.product([16, 32], [1.0, 2.0, 4.0], [50, 100]), 1):
    cid = f"E{i:02d}"
    cfg = dict(T=T, embed_dim=de, snr_shift=shift, parameterization="x0", cond_tokens=True, cond_dropout=0.1)
    cmd = (f"{PY} scripts/experiments/screen_embedding_ddpm.py --train {TRAIN} --full_train {TRAIN} --test {VAL} "
           f"--eval_n 100000000 --reference_csv none --out {OUT}/emb_ddpm --configs {cid} "
           f"--config_json {shlex.quote(json.dumps({cid: cfg}))} --epochs {EPOCHS} --batch_size {BATCH} --lr 1e-3 --seed 42")
    add("emb_ddpm", cid, cfg, {"default": f"{OUT}/emb_ddpm/{cid}/generated.csv"}, cmd)

# --- baselines through run_tabular_baselines.py ---------------------------------------------------
B_COMMON = f"--traindata {TRAIN} --testdata {VAL} --max_steps {STEPS} --batch_size {BATCH} --seed 42 --num_seeds 1 --no_save_models"


def baseline(model_key, tag, cid, params, extra):
    d = f"{OUT}/{model_key}/{cid}"
    cmd = f"{PY} scripts/baselines/run_tabular_baselines.py {B_COMMON} --output_dir {d} {extra}"
    add(model_key, cid, params, {"default": f"{d}/{tag}_gene.csv"}, cmd)


for i, (T, lr, mlp) in enumerate(itertools.product([100, 1000], [2e-4, 1e-3], ["256 512 512 256", "512 512 512 512"]), 1):
    baseline("tabddpm_cond", "TABDDPM", f"T{i:02d}", dict(T=T, lr=lr, mlp=mlp),
             f"--models tabddpm --tabddpm_conditional --tabddpm_t {T} --lr {lr} --tabddpm_hidden_layers {mlp}")
for i, (emb, dims, lr) in enumerate(itertools.product([128, 256], ["256 256", "512 512"], [2e-4, 1e-3]), 1):
    baseline("ctgan_cond", "CTGAN_COND", f"G{i:02d}", dict(embedding_dim=emb, dims=dims, lr=lr),
             f"--models ctgan_cond --ctgan_embedding_dim {emb} --ctgan_gen_dims {dims} --ctgan_dis_dims {dims} --ctgan_lr {lr}")
for i, (lat, dims, lf) in enumerate(itertools.product([64, 128], ["128 128", "256 256"], [1.0, 2.0]), 1):
    baseline("tvae_cond", "TVAE_COND", f"V{i:02d}", dict(latent=lat, dims=dims, loss_factor=lf),
             f"--models tvae_cond --tvae_latent_dim {lat} --tvae_dims {dims} --tvae_loss_factor {lf}")
for i, (cont, bs) in enumerate(itertools.product([True, False], [500, 1000]), 1):
    flag = "--datgan_continuous_time" if cont else "--no-datgan_continuous_time"
    d = f"{OUT}/datgan_cond/A{i:02d}"
    cmd = (f"{PY} scripts/baselines/run_tabular_baselines.py --traindata {TRAIN} --testdata {VAL} --max_steps {STEPS} "
           f"--batch_size {bs} --seed 42 --num_seeds 1 --no_save_models --output_dir {d} "
           f"--models datgan --datgan_conditional {flag}")
    add("datgan_cond", f"A{i:02d}", dict(continuous_time=cont, batch_size=bs), {"default": f"{d}/DATGAN_gene.csv"}, cmd)

out = REPO / OUT
out.mkdir(parents=True, exist_ok=True)
with open(out / "jobs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("job_id\tmodel\tcfg_id\tparams\toutputs\tcommand\n")
    for j, job in enumerate(jobs, 1):
        f.write(f"{j}\t{job['model']}\t{job['cfg_id']}\t{job['params']}\t{job['outputs']}\t{job['command']}\n")
counts = {}
for job in jobs:
    counts[job["model"]] = counts.get(job["model"], 0) + 1
print(f"{len(jobs)} jobs -> {out / 'jobs.tsv'}: {counts}")
