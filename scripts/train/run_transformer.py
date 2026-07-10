import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse

from model.Transformer_Net import TripDiffusionModel
from utils.diffusion_experiment import add_common_diffusion_arguments, launch_diffusion_experiment


def main(args):
    launch_diffusion_experiment(
        args=args,
        model_cls=TripDiffusionModel,
        experiment_name="D3PM_TF",
        model_note="Using pure Transformer diffusion model.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_diffusion_arguments(parser, "Train pure-Transformer TripDiffusionModel and generate samples.")
    parser.set_defaults(lambda_weight=2.0, lambda_joint=0.5, loss_type="standard")
    main(parser.parse_args())
