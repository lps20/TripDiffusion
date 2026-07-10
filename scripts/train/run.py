import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import setup

setup()

import argparse

from model.HCD_Net import TripDiffusionModel
from utils.diffusion_experiment import add_common_diffusion_arguments, launch_diffusion_experiment


def main(args):
    launch_diffusion_experiment(
        args=args,
        model_cls=TripDiffusionModel,
        experiment_name="D3PM_HCD",
        model_note="Using HCD causal diffusion model.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_diffusion_arguments(parser, "Train HCD TripDiffusionModel and generate samples.")
    parser.set_defaults(loss_type="causal", lambda_weight=1.0, lambda_joint=0.0)
    main(parser.parse_args())
