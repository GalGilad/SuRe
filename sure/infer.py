# sure/infer.py
# This script runs inference using a trained SuRe model to predict mutational
# signature exposures for new data. It loads the model, its configuration,
# and the trait mapping to ensure consistent and accurate predictions.

import os
import torch
import pandas as pd
import numpy as np
import pickle
import json
from .model import SuReNet


def run_inference(args):
    """
    Loads a trained model and its configuration to run inference on new data.
    """
    print("--- Running SuRe Inference ---")

    # --- Load Model and Configuration ---
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    model_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(model_dir, "config.json")
    trait_map_path = os.path.join(model_dir, "trait_map.json")

    if not os.path.exists(config_path):
        print(f"Error: Configuration file 'config.json' not found in {model_dir}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded model configuration from {config_path}")

    # --- Load Input Data ---
    print(f"Loading mutation counts from {args.mutation_counts_path}...")
    mutation_counts_df = pd.read_csv(args.mutation_counts_path, index_col=0)
    mutation_counts_df = mutation_counts_df.sort_index(axis=0).sort_index(axis=1)
    samples = mutation_counts_df.columns.tolist()
    counts_tensor = torch.FloatTensor(mutation_counts_df.values.T)

    # --- Handle Traits for Inference ---
    traits_tensor = None
    if config["num_traits"] > 0:
        if not os.path.exists(trait_map_path):
            print(f"Error: 'trait_map.json' not found in {model_dir}. Please retrain the model to generate it.")
            return

        with open(trait_map_path, 'r') as f:
            trait_one_hot_map = json.load(f)
        print(f"Loaded trait map from {trait_map_path}")

        if args.trait_path:
            print(f"Loading inference traits from {args.trait_path}...")
            with open(args.trait_path, "rb") as f:
                inference_trait_map = pickle.load(f)
        else:
            print("No inference trait path provided. Assuming a single trait for all samples.")
            single_trait_name = list(trait_one_hot_map.keys())[0]
            inference_trait_map = {sample: single_trait_name for sample in samples}

        # Create one-hot vectors using the consistent map from training
        one_hot_vectors = []
        for sample in samples:
            trait = inference_trait_map.get(sample)
            # If a trait from inference data is not in the training map, use a zero vector.
            one_hot_vector = trait_one_hot_map.get(trait, [0] * config["num_traits"])
            one_hot_vectors.append(one_hot_vector)

        traits_tensor = torch.FloatTensor(one_hot_vectors)
    else:
        traits_tensor = torch.zeros(len(samples), 0)

    # --- Initialize Model ---
    print("Initializing model...")
    model = SuReNet(
        num_signatures=config["num_signatures"],
        num_traits=config["num_traits"],
        num_experts=config["num_experts"],
        hidden_units=config["hidden_units"],
        dropout_rate=0.0  # No dropout for inference
    )

    # --- Load Weights and Run Prediction ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model weights loaded. Predicting exposures...")

    with torch.no_grad():
        counts_tensor = counts_tensor.to(device)
        if traits_tensor is not None:
            traits_tensor = traits_tensor.to(device)

        predicted_exposures = model(counts_tensor, traits_tensor)

    # --- Format and Save Results ---
    signature_names = [f"Signature_{i + 1}" for i in range(config["num_signatures"])]
    results_df = pd.DataFrame(
        predicted_exposures.cpu().numpy(),
        index=samples,
        columns=signature_names
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(args.output_file)
    print(f"--- Inference complete. Results saved to {args.output_file} ---")
