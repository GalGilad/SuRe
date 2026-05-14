# sure/evaluate.py
# This script is used for local evaluation of a trained model's performance
# across various levels of data sparsity.

import os
import torch
import pandas as pd
import numpy as np
import pickle
import json
import math
import argparse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from .model import SuReNet


def l1_reconstruction_error(y_true, y_pred):
    """Calculates the L1 reconstruction error between true and predicted exposures."""
    return np.sum(np.abs(y_true - y_pred))


def plot_evaluation_results(results, output_dir):
    """Generates and saves a bar chart summarizing the evaluation results."""
    labels = [str(k) for k in results.keys()]
    errors = [v['avg_error'] for v in results.values()]
    correlations = [v['avg_correlation'] for v in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Mutations Sampled (m)')
    ax1.set_ylabel('Avg. Reconstruction Error (L1 Norm)', color=color)
    bars1 = ax1.bar(x - width / 2, errors, width, label='Reconstruction Error', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg. Pearson Correlation', color=color)
    bars2 = ax2.bar(x + width / 2, correlations, width, label='Pearson Correlation', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Model Performance vs. Data Sparsity', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    plot_path = os.path.join(output_dir, "evaluation_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n--> Saved evaluation summary plot to: {plot_path}")


def run_evaluation(args):
    """
    Loads a trained model and evaluates its performance across different downsample sizes.
    """
    print("--- Running SuRe Model Evaluation ---")

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

    # --- Load Evaluation Data ---
    print(f"Loading evaluation data from directory: {args.data_dir}...")
    mutation_counts_path = os.path.join(args.data_dir, "mutation_counts.csv")
    exposures_path = os.path.join(args.data_dir, "exposures.csv")

    if not os.path.exists(mutation_counts_path) or not os.path.exists(exposures_path):
        print(f"Error: 'mutation_counts.csv' or 'exposures.csv' not found in {args.data_dir}")
        return

    mutation_counts_df = pd.read_csv(mutation_counts_path, index_col=0).sort_index(axis=0).sort_index(axis=1)
    exposures_df = pd.read_csv(exposures_path, index_col=0).sort_index(axis=0).sort_index(axis=1)

    common_samples = mutation_counts_df.columns.intersection(exposures_df.columns)
    mutation_counts_df = mutation_counts_df[common_samples]
    exposures_df = exposures_df[common_samples]
    samples = common_samples.tolist()

    # --- Trait Handling for Evaluation ---
    trait_one_hot_map = None
    inference_trait_map = None
    if config["num_traits"] > 0:
        if not os.path.exists(trait_map_path):
            print(f"Error: 'trait_map.json' not found in {model_dir}. Please retrain model.")
            return
        with open(trait_map_path, 'r') as f:
            trait_one_hot_map = json.load(f)
        print(f"Loaded trait map from {trait_map_path}")

        eval_trait_path = os.path.join(args.data_dir, "sample_to_trait.pickle")
        if os.path.exists(eval_trait_path):
            with open(eval_trait_path, "rb") as f:
                inference_trait_map = pickle.load(f)
        else:
            single_trait_name = list(trait_one_hot_map.keys())[0]
            inference_trait_map = {sample: single_trait_name for sample in samples}

    # --- Initialize Model ---
    model = SuReNet(
        num_signatures=config["num_signatures"],
        num_traits=config["num_traits"],
        num_experts=config["num_experts"],
        hidden_units=config["hidden_units"],
        dropout_rate=0.0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Applied weights_only=True for secure loading
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded and set to evaluation mode on device: {device}")

    # --- Pre-calculate ground truth relative exposures ---
    true_relative_exposures_list = [
        exposures_df[s].values / np.sum(exposures_df[s].values) if np.sum(exposures_df[s].values) > 0 else np.zeros(
            config["num_signatures"]) for s in samples]

    # --- Evaluation Loop ---
    sparsity_levels = [300, 100, 30, 10, 6, 3]
    print("\n--- Starting Evaluation ---")
    results = {}

    for size in sparsity_levels:
        # Dynamic repetition calculation based on user request
        repetitions = int(min(30, math.ceil(300 / size)))
        print(f"\nGenerating evaluation batch for sample size: {size} (Repetitions: {repetitions})...")

        # --- Batch Generation ---
        batch_counts, batch_traits, batch_true_indices = [], [], []
        for i, sample_name in enumerate(samples):
            full_counts = mutation_counts_df[sample_name].values.astype(np.int64)
            total_mutations = np.sum(full_counts)
            if total_mutations == 0: continue

            for _ in range(repetitions):
                if size > total_mutations:
                    eval_counts = full_counts
                else:
                    flattened = np.repeat(np.arange(len(full_counts)), full_counts)
                    subsampled = np.random.choice(flattened, size=size, replace=False)
                    eval_counts, _ = np.histogram(subsampled, bins=np.arange(len(full_counts) + 1))

                batch_counts.append(eval_counts)

                trait_one_hot = [0.0] * config["num_traits"]
                if inference_trait_map and trait_one_hot_map and sample_name in inference_trait_map:
                    trait_name = inference_trait_map[sample_name]
                    trait_one_hot = trait_one_hot_map.get(trait_name, trait_one_hot)
                batch_traits.append(trait_one_hot)

                batch_true_indices.append(i)

        # --- Batch Inference ---
        print(f"Running inference on {len(batch_counts)} profiles...")
        max_batch_size = 2000
        all_predicted_exposures = []
        with torch.no_grad():
            for i in range(0, len(batch_counts), max_batch_size):
                counts_tensor = torch.FloatTensor(np.array(batch_counts[i:i + max_batch_size])).to(device)
                traits_tensor = torch.FloatTensor(np.array(batch_traits[i:i + max_batch_size])).to(device)
                predictions = model(counts_tensor, traits_tensor).cpu().numpy()
                all_predicted_exposures.append(predictions)
        all_predicted_exposures = np.concatenate(all_predicted_exposures, axis=0)

        # --- Metric Calculation ---
        all_errors, all_correlations = [], []
        for i in range(len(all_predicted_exposures)):
            pred = all_predicted_exposures[i]
            true = true_relative_exposures_list[batch_true_indices[i]]
            all_errors.append(l1_reconstruction_error(true, pred))
            if np.std(true) > 0 and np.std(pred) > 0:
                all_correlations.append(pearsonr(true, pred)[0])

        results[size] = {'avg_error': np.mean(all_errors),
                         'avg_correlation': np.mean(all_correlations) if all_correlations else np.nan}
        print(f"  > Average Exposure Reconstruction Error: {results[size]['avg_error']:.4f}")
        print(f"  > Average Exposure Correlation: {results[size]['avg_correlation']:.4f}")

    print("\n--- Evaluation Complete ---")

    # Save results dictionary as a CSV for easy parsing later
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'm_mutations'

    # Extract the name of that folder (e.g., "brca")
    dataset_name = os.path.basename(model_dir)
    # Create the results directory
    output_dir = os.path.join("results", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
    results_df.to_csv(csv_path)
    print(f"--> Saved metrics to: {csv_path}")

    plot_evaluation_results(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SuRe model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained best_model.pth file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the test mutation_counts.csv and exposures.csv")

    parsed_args = parser.parse_args()
    run_evaluation(parsed_args)