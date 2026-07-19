import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import logging
import os
import datetime
import ast
import json
import time
import copy

from model.HCD_Net_v2 import TripDiffusionModel
import utils.train_utils, utils.test_utils
from utils.multi_seed import (
    add_multiseed_arguments,
    parse_seeds,
    run_multiseed,
    set_global_seed,
)


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _report_gate_values(model):
    base_model = _unwrap_model(model)
    if not hasattr(base_model, "get_gate_values"):
        logging.warning("Model does not expose gate values; skipping gate report.")
        return None

    gate_values = base_model.get_gate_values()
    logging.info("Learned gate values (alpha = sigmoid(raw), blend strength per stream):")
    for gate in gate_values:
        logging.info(
            "  Layer %d | act: alpha=%.4f (raw=%.4f) | st: alpha=%.4f (raw=%.4f) | mode: alpha=%.4f (raw=%.4f)",
            gate["layer"],
            gate["alpha_act"],
            gate["gate_act_raw"],
            gate["alpha_st"],
            gate["gate_st_raw"],
            gate["alpha_mode"],
            gate["gate_mode_raw"],
        )
    return gate_values


def _default_features_info():
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


def _default_cond_info():
    return [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9},
    ]


def _configure_logging(exp_dir, append=False):
    log_file = os.path.join(exp_dir, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a" if append else "w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return log_file


def run_once(args, seed, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    resume_epoch = int(getattr(args, "resume_epoch", 0) or 0)
    _configure_logging(exp_dir, append=(resume_epoch > 0 and bool(args.checkpoint)))

    model_file = os.path.join(exp_dir, "model.pth")
    generation_file = os.path.join(exp_dir, "generated_samples.csv")
    metrics_file = args.metrics_file or os.path.join(exp_dir, "generated_samples_metrics.json")

    logging.info("Command line arguments: %s", vars(args))
    set_global_seed(seed)
    logging.info("Global random seed set to: %d", seed)

    features_info = _default_features_info()
    cond_info = _default_cond_info()

    try:
        joint_pairs_list = ast.literal_eval(args.joint_pairs)
        print(f"Joint pairs loaded: {joint_pairs_list}")
    except (ValueError, SyntaxError):
        raise ValueError("joint_pairs invalid format. Please provide a valid list of tuples, e.g., '[(0,4),(1,5)]'")
    T = args.T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    d_model = args.d_model
    shared_layers = args.shared_layers
    causal_layers = args.causal_layers
    if args.no_joint_heads:
        d_model = d_model if d_model is not None else 192
        shared_layers = shared_layers if shared_layers is not None else 3
        causal_layers = causal_layers if causal_layers is not None else 3
        logging.info(
            "No joint heads: using marginal sample-statistics joint loss (%s); "
            "d_model=%d, shared_layers=%d, causal_layers=%d",
            args.joint_loss_mode,
            d_model,
            shared_layers,
            causal_layers,
        )
    else:
        d_model = d_model if d_model is not None else 128
        shared_layers = shared_layers if shared_layers is not None else 2
        causal_layers = causal_layers if causal_layers is not None else 2

    model = TripDiffusionModel(
        features_info,
        cond_info,
        T,
        joint_pairs_list,
        gate_init={
            "act": args.gate_init_act,
            "st": args.gate_init_st,
            "mode": args.gate_init_mode,
        },
        freeze_gates=args.freeze_gates,
        st_cascade=args.st_cascade,
        st_cascade_chain=args.st_cascade_chain,
        hard_stream_cascade=args.hard_stream_cascade,
        use_joint_heads=not args.no_joint_heads,
        d_model=d_model,
        shared_layers=shared_layers,
        causal_layers=causal_layers,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info("Model parameters: %d (%.2f M)", num_params, num_params / 1e6)
    if args.hard_stream_cascade:
        logging.info("Using HARD stream cascade: act -> st -> mode (sequential, full replace, no soft gates).")
    if args.freeze_gates:
        import math as _math

        logging.info(
            "Gates FROZEN at init | act=%.3f (α≈%.3f) | st=%.3f (α≈%.3f) | mode=%.3f (α≈%.3f)",
            args.gate_init_act,
            1.0 / (1.0 + _math.exp(-args.gate_init_act)),
            args.gate_init_st,
            1.0 / (1.0 + _math.exp(-args.gate_init_st)),
            args.gate_init_mode,
            1.0 / (1.0 + _math.exp(-args.gate_init_mode)),
        )
    if args.st_cascade:
        logging.info(
            "Using HCD v2 ST cascade chain=%s | phase1=%s | phase2=%s",
            args.st_cascade_chain,
            getattr(model, "st_cascade_phase1_names", None),
            getattr(model, "st_cascade_phase2_names", None),
        )
    else:
        logging.info("Using HCD v2: shared transformer + soft causal adapters.")

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

    feature_loss_weights = {}
    if args.feature_loss_weights:
        feature_loss_weights = json.loads(args.feature_loss_weights)

    if not args.eval_only:
        logging.info("Loading Dataset: %s", args.traindata)
        logging.info("Training with %d epochs, batch size %d", args.epochs, args.batch_size)
        if args.traindata:
            dataset = utils.train_utils.load_data(args.traindata, features_info, cond_info)
        else:
            logging.info("No dataset file provided. Generating synthetic data.")
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
            feature_loss_weights=feature_loss_weights,
            joint_loss_mode=args.joint_loss_mode,
            start_epoch=resume_epoch,
            initial_best_loss=getattr(args, "resume_best_loss", None),
        )
    else:
        logging.info("Evaluation only mode. Skipping training.")

    gate_values = _report_gate_values(model)
    logging.info("Model saved to %s", model_file)

    test_df = pd.read_csv(args.testdata)
    train_eval_df = pd.read_csv(args.traindata) if args.traindata else None
    match_test = not args.random_condition_sampling
    joint_sample_steps = None
    if args.joint_sampling_at_inference:
        try:
            joint_sample_steps = ast.literal_eval(args.joint_sample_steps)
            if not isinstance(joint_sample_steps, (list, tuple)):
                raise ValueError("joint_sample_steps must be a list of integers")
            joint_sample_steps = [int(t) for t in joint_sample_steps]
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"joint_sample_steps invalid format: {args.joint_sample_steps!r}"
            ) from exc

    sample_start = time.perf_counter()
    generated_samples, truth_samples = utils.train_utils.sample_trip(
        model,
        test_df,
        num_samples=args.num_samples,
        device=device,
        match_test_one_to_one=match_test,
        use_joint_sampling=args.joint_sampling_at_inference,
        joint_sample_steps=joint_sample_steps,
        joint_gibbs_iters=args.joint_gibbs_iters,
    )
    num_generated = len(generated_samples)
    sampling_seconds = float(time.perf_counter() - sample_start)
    logging.info(
        "Sampling completed in %.4f seconds for %d samples (match_test=%s).",
        sampling_seconds,
        num_generated,
        match_test,
    )
    utils.train_utils.save_generated_samples(generated_samples, output_file=generation_file)

    truth_trips_all = [s["trip"] for s in truth_samples]
    generated_trips_all = [s["trip"] for s in generated_samples]
    eva_all = utils.test_utils.evaluate_generated_trips(
        truth_trips_all,
        generated_trips_all,
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
        "T": int(T),
        "num_samples": int(num_generated),
        "eval_sampling": "match_test_one_to_one" if match_test else "random_with_replacement",
        "joint_sampling_at_inference": bool(args.joint_sampling_at_inference),
        "joint_sample_steps": joint_sample_steps if args.joint_sampling_at_inference else None,
        "joint_gibbs_iters": int(args.joint_gibbs_iters) if args.joint_sampling_at_inference else None,
        "st_cascade": bool(args.st_cascade),
        "st_cascade_chain": args.st_cascade_chain if args.st_cascade else None,
        "hard_stream_cascade": bool(args.hard_stream_cascade),
        "freeze_gates": bool(args.freeze_gates),
        "gate_init_act": float(args.gate_init_act),
        "gate_init_st": float(args.gate_init_st),
        "gate_init_mode": float(args.gate_init_mode),
        "use_joint_heads": not bool(args.no_joint_heads),
        "joint_loss_mode": args.joint_loss_mode if args.no_joint_heads else "joint_head",
        "d_model": int(d_model),
        "shared_layers": int(shared_layers),
        "causal_layers": int(causal_layers),
        "num_parameters": int(num_params),
        "sampling_seconds": sampling_seconds,
        "sampling_seconds_per_10k": float(sampling_seconds * (10000.0 / max(float(num_generated), 1.0))),
        "evaluation": eva_all,
    }
    if gate_values is not None:
        metrics_payload["gate_values"] = gate_values

    os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics JSON: %s", metrics_file)
    logging.info("Training and evaluation completed. Generated samples saved to %s", generation_file)

    return metrics_payload


def main(args):
    seeds = parse_seeds(args.seed, args.seeds, args.num_seeds, args.seed_start)

    if len(seeds) == 1:
        if args.exp_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join("exp", datetime.datetime.now().strftime("%Y%m%d"), timestamp)
        else:
            exp_dir = args.exp_dir
        run_once(args, seeds[0], exp_dir)
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
    logging.info("Running HCD v2 across %d seeds: %s", len(seeds), seeds)

    single_args = copy.copy(args)
    single_args.metrics_file = None
    result = run_multiseed(
        args=single_args,
        seeds=seeds,
        run_once=run_once,
        output_root=output_root,
        experiment_name="HCD_v2",
    )
    print("Multi-seed experiment completed. Summary:", result["summary"])
    print("Artifacts saved under:", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HCD v2 TripDiffusionModel and generate samples.")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv", help="Path to training dataset CSV file")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv", help="Path to testing dataset CSV file")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum change in loss to qualify as an improvement.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=2.0, help="Weight for auxiliary loss")
    parser.add_argument("--lambda_joint", type=float, default=0.5, help="Weight for joint loss")
    parser.add_argument("--T", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--parallel", type=bool, default=False, help="Parallel computing")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Generated sample count when --random_condition_sampling is set (>0). "
        "Default 0 uses the full test set with 1:1 matched conditions.",
    )
    parser.add_argument(
        "--random_condition_sampling",
        action="store_true",
        help="Randomly sample test conditions with replacement instead of 1:1 full-test generation.",
    )
    parser.add_argument("--exp_dir", type=str, default=None, help="Directory to save logs and models (default: auto timestamp)")
    parser.add_argument("--joint_pairs", type=str, default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]", help="List of joint feature pairs for joint loss, e.g., [(0,4),(1,5)]")
    parser.add_argument("--loss_type", type=str, default="standard", help="Type of loss function to use: 'standard' or 'causal'")
    parser.add_argument("--causal_weight", type=str, default=None, help="Weights for causal loss groups in JSON format, e.g., '{\"st\": 1.0, \"mode\": 1.0}'")
    parser.add_argument("--batch_sampling", type=str, default="sequential", choices=["sequential", "shuffle", "balanced"], help="Batch index sampling strategy")
    parser.add_argument("--sampling_feature", type=str, default="act_num", help="Feature used for balanced batch sampling")
    parser.add_argument("--sampling_power", type=float, default=1.0, help="Strength of inverse-frequency reweighting for balanced sampling")
    parser.add_argument("--t_sampling", type=str, default="uniform", choices=["uniform", "sqrt", "late"], help="Diffusion timestep sampling strategy")
    parser.add_argument("--gate_init_act", type=float, default=-1.0, help="Initial raw gate for act stream (sigmoid -> alpha).")
    parser.add_argument("--gate_init_st", type=float, default=-1.0, help="Initial raw gate for space-time stream.")
    parser.add_argument("--gate_init_mode", type=float, default=-1.0, help="Initial raw gate for mode stream.")
    parser.add_argument(
        "--freeze_gates",
        action="store_true",
        help="Freeze soft-causal gates at gate_init (ablate learnable soft gating; α=sigmoid(init)).",
    )
    parser.add_argument(
        "--feature_loss_weights",
        type=str,
        default=None,
        help='JSON dict of per-feature CE/VB weights, e.g. \'{"start_zcode_num": 2.0, "end_zcode_num": 2.0}\'',
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained model.pth")
    parser.add_argument("--eval_only", action="store_true", help="Set this flag to skip training and only evaluate")
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=0,
        help="Number of epochs already completed when resuming from --checkpoint (0-based completed count). "
        "Training continues from resume_epoch+1 through --epochs.",
    )
    parser.add_argument(
        "--resume_best_loss",
        type=float,
        default=None,
        help="Best train loss observed before resume (for early-stopping continuity).",
    )
    parser.add_argument(
        "--no_joint_heads",
        action="store_true",
        help="Remove explicit joint heads; supervise pairs via marginal sample statistics.",
    )
    parser.add_argument(
        "--joint_loss_mode",
        type=str,
        default="batch_stats",
        choices=["batch_stats", "product"],
        help="Joint loss without joint heads: batch co-occurrence KL or per-sample product CE.",
    )
    parser.add_argument("--d_model", type=int, default=None, help="Transformer hidden size.")
    parser.add_argument("--shared_layers", type=int, default=None, help="Shared encoder layers.")
    parser.add_argument("--causal_layers", type=int, default=None, help="Causal adapter layers.")
    parser.add_argument(
        "--joint_sampling_at_inference",
        action="store_true",
        help="Refine paired features at inference via joint heads or marginal-product Gibbs.",
    )
    parser.add_argument(
        "--joint_sample_steps",
        type=str,
        default="[1]",
        help="Reverse-diffusion timesteps for joint pair Gibbs sampling, e.g. '[1]' or '[1,2]'.",
    )
    parser.add_argument(
        "--joint_gibbs_iters",
        type=int,
        default=3,
        help="Gibbs sweeps over all joint pairs when joint_sampling_at_inference is enabled.",
    )
    parser.add_argument(
        "--st_cascade",
        action="store_true",
        help="Use ST loc/time mini-cascade inside causal adapters (Step B).",
    )
    parser.add_argument(
        "--hard_stream_cascade",
        action="store_true",
        help="Hard stream cascade: update act->st->mode sequentially with full replace (no soft gates).",
    )
    parser.add_argument(
        "--st_cascade_chain",
        type=str,
        default="loc_then_time",
        choices=[
            "loc_then_time",
            "time_then_loc",
            "end_first_loc",
            "zcode_first",
            "types_then_z",
            "start_then_end",
        ],
        help="ST cascade token order preset (only used with --st_cascade).",
    )
    parser.add_argument("--metrics_file", type=str, default=None, help="Path to write metrics JSON (default: <exp_dir>/generated_samples_metrics.json)")
    add_multiseed_arguments(parser, default_num_seeds=5)

    args = parser.parse_args()
    main(args)
