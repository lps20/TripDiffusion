"""Shared single/multi-seed runner for diffusion training scripts."""

from __future__ import annotations

import argparse
import ast
import copy
import datetime
import json
import logging
import os
import time
from typing import Any, Callable, Optional, Type

import pandas as pd
import torch
from torch import nn
import torch.optim as optim

import utils.train_utils
import utils.test_utils
from utils.multi_seed import add_multiseed_arguments, parse_seeds, run_multiseed, set_global_seed


def default_features_info():
    return [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241},
    ]


def default_cond_info():
    return [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9},
    ]


def configure_logging(exp_dir: str) -> str:
    log_file = os.path.join(exp_dir, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return log_file


def run_diffusion_once(
    args: Any,
    seed: int,
    exp_dir: str,
    model_cls: Type[nn.Module],
    experiment_name: str,
    model_note: Optional[str] = None,
    post_train_hook: Optional[Callable[[nn.Module], Any]] = None,
) -> dict:
    os.makedirs(exp_dir, exist_ok=True)
    configure_logging(exp_dir)

    model_file = os.path.join(exp_dir, "model.pth")
    generation_file = os.path.join(exp_dir, "generated_samples.csv")
    metrics_file = getattr(args, "metrics_file", None) or os.path.join(exp_dir, "generated_samples_metrics.json")

    logging.info("Command line arguments: %s", vars(args))
    set_global_seed(seed)
    logging.info("Global random seed set to: %d", seed)

    features_info = default_features_info()
    cond_info = default_cond_info()
    joint_pairs_list = ast.literal_eval(args.joint_pairs)
    T = args.T
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    if model_note:
        logging.info(model_note)

    model = model_cls(features_info, cond_info, T, joint_pairs_list).to(device)

    if args.checkpoint:
        logging.info("Loading model checkpoint from %s", args.checkpoint)
        state_dict = torch.load(args.checkpoint, map_location=device)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        logging.info("Model checkpoint loaded successfully.")

    if args.parallel:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.eval_only:
        logging.info("Loading Dataset: %s", args.traindata)
        logging.info("Training with %d epochs, batch size %d", args.epochs, args.batch_size)
        if args.traindata:
            dataset = utils.train_utils.load_data(args.traindata, features_info, cond_info)
        else:
            dataset = utils.train_utils.generate_synthetic_trips(num_samples=1000)
        utils.train_utils.train_model(
            model,
            optimizer,
            dataset,
            features_info,
            args.lambda_weight,
            args.lambda_joint,
            T,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            loss_type=args.loss_type,
            causal_weight=args.causal_weight,
            model_save_path=model_file,
            patience=args.patience,
            min_delta=args.min_delta,
            batch_sampling=args.batch_sampling,
            sampling_feature=args.sampling_feature,
            sampling_power=args.sampling_power,
            t_sampling=args.t_sampling,
        )
    else:
        logging.info("Evaluation only mode. Skipping training.")

    extra_payload = {}
    if post_train_hook is not None:
        hook_result = post_train_hook(model)
        if hook_result is not None:
            extra_payload["post_train"] = hook_result

    logging.info("Model saved to %s", model_file)

    test_df = pd.read_csv(args.testdata)
    train_eval_df = pd.read_csv(args.traindata) if args.traindata else None
    sample_start = time.perf_counter()
    generated_samples, truth_samples = utils.train_utils.sample_trip(
        model,
        test_df,
        num_samples=args.num_samples,
        device=device,
    )
    sampling_seconds = float(time.perf_counter() - sample_start)
    utils.train_utils.save_generated_samples(generated_samples, output_file=generation_file)

    eva_all = utils.test_utils.evaluate_generated_trips(
        [s["trip"] for s in truth_samples],
        [s["trip"] for s in generated_samples],
        features_info,
        generated_samples=generated_samples,
        cond_info=cond_info,
        train_real_df=train_eval_df,
        test_real_df=test_df,
        random_state=seed,
    )
    logging.info("Evaluation results on all data: %s", eva_all)

    metrics_payload = {
        "seed": int(seed),
        "experiment": experiment_name,
        "T": int(T),
        "num_samples": int(args.num_samples),
        "sampling_seconds": sampling_seconds,
        "sampling_seconds_per_10k": float(sampling_seconds * (10000.0 / max(float(args.num_samples), 1.0))),
        "evaluation": eva_all,
    }
    metrics_payload.update(extra_payload)

    os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", metrics_file)
    return metrics_payload


def launch_diffusion_experiment(
    args: Any,
    model_cls: Type[nn.Module],
    experiment_name: str,
    model_note: Optional[str] = None,
    post_train_hook: Optional[Callable[[nn.Module], Any]] = None,
) -> None:
    seeds = parse_seeds(args.seed, args.seeds, args.num_seeds, args.seed_start)

    def _run_once(run_args, seed, exp_dir):
        return run_diffusion_once(
            run_args,
            seed,
            exp_dir,
            model_cls=model_cls,
            experiment_name=experiment_name,
            model_note=model_note,
            post_train_hook=post_train_hook,
        )

    if len(seeds) == 1:
        if args.exp_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join("exp", datetime.datetime.now().strftime("%Y%m%d"), timestamp)
        else:
            exp_dir = args.exp_dir
        _run_once(args, seeds[0], exp_dir)
        print("Training and evaluation completed. Check logs and saved model in:", exp_dir)
        return

    if args.exp_dir is not None:
        output_root = args.exp_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = os.path.join("exp", datetime.datetime.now().strftime("%Y%m%d"), f"multiseed_{timestamp}")

    os.makedirs(output_root, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Running %s across %d seeds: %s", experiment_name, len(seeds), seeds)

    single_args = copy.copy(args)
    if hasattr(single_args, "metrics_file"):
        single_args.metrics_file = None
    result = run_multiseed(
        args=single_args,
        seeds=seeds,
        run_once=_run_once,
        output_root=output_root,
        experiment_name=experiment_name,
    )
    print("Multi-seed experiment completed. Summary:", result["summary"])
    print("Artifacts saved under:", output_root)


def add_common_diffusion_arguments(parser: argparse.ArgumentParser, description: str) -> None:
    parser.description = description
    parser.add_argument("--traindata", type=str, default="data/train_data.csv")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_weight", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--parallel", type=bool, default=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--joint_pairs", type=str, default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]")
    parser.add_argument("--loss_type", type=str, default="standard", choices=["standard", "causal"])
    parser.add_argument("--causal_weight", type=str, default=None)
    parser.add_argument("--batch_sampling", type=str, default="sequential", choices=["sequential", "shuffle", "balanced"])
    parser.add_argument("--sampling_feature", type=str, default="act_num")
    parser.add_argument("--sampling_power", type=float, default=1.0)
    parser.add_argument("--t_sampling", type=str, default="uniform", choices=["uniform", "sqrt", "late"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    add_multiseed_arguments(parser, default_num_seeds=5)
