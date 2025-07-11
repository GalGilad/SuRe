# sure/__main__.py
# This script serves as the main command-line interface (CLI) for the SuRe project.
# It uses argparse to define and handle three primary commands: `train`, `infer`, and `evaluate`.

import argparse
import os

def main():
    """
    Parses command-line arguments and runs the appropriate command.
    """
    parser = argparse.ArgumentParser(
        description="SuRe: A Supervised Neural Network for Mutational Signature Refitting."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Training Parser ---
    train_parser = subparsers.add_parser("train", help="Train a new SuRe model.")
    train_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data subdirectory (e.g., 'data/brca' or 'data/pan')."
    )
    train_parser.add_argument(
        "--num_experts",
        type=int,
        default=8,
        help="Number of expert networks in the Mixture-of-Experts model."
    )
    train_parser.add_argument(
        "--hidden_units",
        type=int,
        default=500,
        help="Number of hidden units in each expert's layers."
    )
    train_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate for the optimizer."
    )
    train_parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for regularization."
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs."
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training."
    )

    # --- Inference Parser ---
    infer_parser = subparsers.add_parser("infer", help="Run inference with a trained SuRe model.")
    infer_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth). The script will look for config.json and trait_map.json in the same directory."
    )
    infer_parser.add_argument(
        "--mutation_counts_path",
        type=str,
        required=True,
        help="Path to the CSV file with mutation counts for inference."
    )
    infer_parser.add_argument(
        "--trait_path",
        type=str,
        default=None,
        help="Path to the sample-to-trait pickle file. Not required if the model was trained on a single trait."
    )
    infer_parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the predicted exposures CSV file."
    )

    # --- Evaluation Parser ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model's performance on sparse data.")
    eval_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth)."
    )
    eval_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory containing mutation_counts.csv, exposures.csv, and optionally sample_to_trait.pickle."
    )

    args = parser.parse_args()

    # Create base directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.command == "train":
        from . import train
        train.start_training(args)
    elif args.command == "infer":
        from . import infer
        infer.run_inference(args)
    elif args.command == "evaluate":
        from . import evaluate
        evaluate.run_evaluation(args)

if __name__ == "__main__":
    main()
