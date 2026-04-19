import argparse
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import logging
import os
import datetime
import ast

from model.Transformer_Net import TripDiffusionModel
import utils.train_utils, utils.test_utils

def main(args):
    if args.exp_dir is not None:
        exp_dir = args.exp_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join("exp", datetime.datetime.now().strftime("%Y%m%d"), f"{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, f"training.log")
    model_file = os.path.join(exp_dir, f"model.pth")
    generation_file = os.path.join(exp_dir, f"generated_samples.csv")
    
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add FileHandler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Add StreamHandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.info("Command line arguments: %s", vars(args))
    
    # Define feature and condition information
    features_info = [
        {"name": "start_type", "type": "categorical", "num_classes": 5},
        {"name": "start_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "act_num", "type": "categorical", "num_classes": 9},
        {"name": "mode_num", "type": "categorical", "num_classes": 9},
        {"name": "end_type", "type": "categorical", "num_classes": 5},
        {"name": "end_zcode_num", "type": "categorical", "num_classes": 77},
        {"name": "start_time_num_6", "type": "ordinal", "num_classes": 241},
        {"name": "trip_time_num_6", "type": "ordinal", "num_classes": 241}
    ]
    cond_info = [
        {"name": "relation", "num_classes": 5},
        {"name": "sex", "num_classes": 2},
        {"name": "age_code", "num_classes": 13},
        {"name": "job_type", "num_classes": 9}
    ]

    try:
        joint_pairs_list = ast.literal_eval(args.joint_pairs)
        print(f"Joint pairs loaded: {joint_pairs_list}") # 调试打印，确认格式正确
    except (ValueError, SyntaxError):
        raise ValueError("joint_pairs invalid format. Please provide a valid list of tuples, e.g., '[(0,4),(1,5)]'") 
    T = args.T  # diffusion steps
    
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")


    model = TripDiffusionModel(features_info, cond_info, T, joint_pairs_list).to(device)

    if args.checkpoint:
        logging.info(f"Loading model checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict)
        logging.info("Model checkpoint loaded successfully.")

    if args.parallel == True:
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if not args.eval_only:
        lambda_weight = args.lambda_weight
        lambda_joint = args.lambda_joint
        
        logging.info("Loading Dataset: %s", args.traindata)
        logging.info("Training with %d epochs, batch size %d", args.epochs, args.batch_size)

        # Load dataset
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
            lambda_weight, 
            lambda_joint, 
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
            t_sampling=args.t_sampling)
    else:
        logging.info("Evaluation only mode. Skipping training.")

    # Save the trained model
    logging.info("Model saved to %s", model_file)

    # Generate samples from the trained model
    test_df = pd.read_csv(args.testdata)
    train_eval_df = pd.read_csv(args.traindata) if args.traindata else None
    generated_samples, truth_samples = utils.train_utils.sample_trip(
        model, 
        test_df, 
        num_samples=args.num_samples,
        device=device
    )
    utils.train_utils.save_generated_samples(generated_samples, output_file = generation_file)


    truth_trips_all = [s["trip"] for s in truth_samples]
    generated_trips_all = [s["trip"] for s in generated_samples]
    
    eva_all = utils.test_utils.evaluate_generated_trips(
        truth_trips_all,
        generated_trips_all,
        features_info,
        generated_samples=generated_samples,
        cond_info=cond_info,
        train_real_df=train_eval_df,
        test_real_df=test_df
    )
    logging.info("Evaluation results on all data: %s", eva_all)

    logging.info("Training and evaluation completed. Generated samples saved to %s", generation_file)
    print("Training and evaluation completed. Check logs and saved model in:", exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pure-Transformer TripDiffusionModel and generate samples.")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv", help="Path to training dataset CSV file")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv", help="Path to testing dataset CSV file")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum change in loss to qualify as an improvement.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=2.0, help="Weight for auxiliary loss")
    parser.add_argument("--lambda_joint", type=float, default=0.5, help="Weight for joint loss")
    parser.add_argument("--T", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--parallel", type=bool, default=True, help="Parallel computing")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate after training")
    parser.add_argument("--exp_dir", type=str, default=None, help="Directory to save logs and models (default: auto timestamp)")
    parser.add_argument("--joint_pairs", type=str, default="[(0,4),(1,5),(2,6),(3,6),(2,3),(6,7)]", help="List of joint feature pairs for joint loss, e.g., [(0,4),(1,5)]")
    parser.add_argument("--loss_type", type=str, default="standard", help="Type of loss function to use: 'standard' or 'causal'")
    parser.add_argument("--causal_weight", type=str, default=None, help="Weights for causal loss groups in JSON format, e.g., '{\"st\": 1.0, \"mode\": 1.0}'")
    parser.add_argument("--batch_sampling", type=str, default="sequential", choices=["sequential", "shuffle", "balanced"], help="Batch index sampling strategy")
    parser.add_argument("--sampling_feature", type=str, default="act_num", help="Feature used for balanced batch sampling")
    parser.add_argument("--sampling_power", type=float, default=1.0, help="Strength of inverse-frequency reweighting for balanced sampling")
    parser.add_argument("--t_sampling", type=str, default="uniform", choices=["uniform", "sqrt", "late"], help="Diffusion timestep sampling strategy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pre-trained model.pth")
    parser.add_argument("--eval_only", action="store_true", help="Set this flag to skip training and only evaluate")
    
    args = parser.parse_args()
    main(args)
