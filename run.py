import argparse
import pandas as pd
import torch
import torch.optim as optim
import logging
import os
import datetime

from model.Net import TripDiffusionModel
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

    T = args.T  # diffusion steps
    
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")


    model = TripDiffusionModel(features_info, cond_info, T).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lambda_weight = args.lambda_weight
    
    logging.info("Loading Dataset: %s", args.traindata)
    logging.info("Training with %d epochs, batch size %d", args.epochs, args.batch_size)

    # Load dataset
    if args.traindata:
        dataset = utils.train_utils.load_data(args.traindata, features_info, cond_info)
    else:
        logging.info("No dataset file provided. Generating synthetic data.")
        dataset = utils.train_utils.generate_synthetic_trips(num_samples=1000)

    utils.train_utils.train_model(model, optimizer, dataset, features_info, lambda_weight, T,
                              epochs=args.epochs, batch_size=args.batch_size, device=device)

    # Save the trained model
    torch.save(model.state_dict(), model_file)
    logging.info("Model saved to %s", model_file)

    # Generate samples from the trained model
    clustered_df = pd.read_csv(args.testdata)
    generated_samples, truth_samples = utils.train_utils.sample_trip_by_clusters(model, clustered_df, num_samples_each=args.num_samples, device=device)
    utils.train_utils.save_generated_samples(generated_samples, output_file = generation_file)


    clusters = sorted(clustered_df["Cluster"].unique())
    truth_trips_all = []
    generated_trips_all = []
    for cluster in clusters:
        logging.info("Evaluating cluster %s", cluster)
        truth_trips = [sample["trip"] for sample in truth_samples[cluster]]
        generated_trips = [sample["trip"] for sample in generated_samples[cluster]]
        truth_trips_all.extend(truth_trips)
        generated_trips_all.extend(generated_trips)
        eva = utils.test_utils.evaluate_generated_trips(truth_trips, generated_trips, features_info)
        logging.info("Evaluation results on cluster %s: %s", cluster, eva)
    eva_all = utils.test_utils.evaluate_generated_trips(truth_trips_all, generated_trips_all, features_info)
    logging.info("Evaluation results on all clusters: %s", eva_all)

    logging.info("Training and evaluation completed. Generated samples saved to %s", generation_file)
    print("Training and evaluation completed. Check logs and saved model in:", exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TripDiffusionModel and generate samples.")
    parser.add_argument("--traindata", type=str, default="data/train_data.csv", help="Path to training dataset CSV file")
    parser.add_argument("--testdata", type=str, default="data/test_data.csv", help="Path to testing dataset CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=1.0, help="Weight for auxiliary loss")
    parser.add_argument("--T", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples of each cluster to generate after training")
    parser.add_argument("--exp_dir", type=str, default=None, help="Directory to save logs and models (default: auto timestamp)")
    args = parser.parse_args()
    main(args)
