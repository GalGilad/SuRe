# sure/data_loader.py
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def create_one_hot_dict(unique_items):
    """Creates a dictionary mapping unique items to their one-hot encoded vectors."""
    item_to_one_hot = {}
    num_items = len(unique_items)
    for i, item in enumerate(unique_items):
        one_hot = [0] * num_items
        one_hot[i] = 1
        item_to_one_hot[item] = one_hot
    return item_to_one_hot

class MutationDataset(Dataset):
    """
    PyTorch Dataset for loading pre-split mutation data.
    - Implements a power-law subsampling strategy during training and validation
      to simulate data sparsity and improve model robustness.
    """

    def __init__(self, data_dir, is_train=True, trait_one_hot_map=None):
        self.data_dir = data_dir
        self.is_train = is_train

        print(f"--> Loading dataset from {data_dir}...")
        self._load_data_files()
        self._handle_traits(trait_one_hot_map)

    def _load_data_files(self):
        """Loads and aligns the mutation counts and exposures dataframes."""
        self.mutation_counts_df = pd.read_csv(os.path.join(self.data_dir, "mutation_counts.csv"), index_col=0)
        self.exposures_df = pd.read_csv(os.path.join(self.data_dir, "exposures.csv"), index_col=0)

        self.mutation_counts_df = self.mutation_counts_df.sort_index(axis=0).sort_index(axis=1)
        self.exposures_df = self.exposures_df.sort_index(axis=0).sort_index(axis=1)

        common_samples = self.mutation_counts_df.columns.intersection(self.exposures_df.columns)
        self.mutation_counts_df = self.mutation_counts_df[common_samples]
        self.exposures_df = self.exposures_df[common_samples]
        self.samples = common_samples.tolist()

        self.num_samples = len(self.samples)
        self.num_signatures = len(self.exposures_df.index)

    def _handle_traits(self, provided_map):
        """Processes trait information, syncing maps between Train and Val sets."""
        trait_file_path = os.path.join(self.data_dir, "sample_to_trait.pickle")
        if os.path.exists(trait_file_path):
            with open(trait_file_path, "rb") as f:
                self.trait_map = pickle.load(f)
            self.trait_map = {s: t for s, t in self.trait_map.items() if s in self.samples}
        else:
            self.trait_map = {sample: "default_trait" for sample in self.samples}

        if provided_map is not None:
            # For Validation Set: Inherit the exact one-hot map from the Training Set
            self.trait_one_hot_map = provided_map
            self.num_traits = len(next(iter(provided_map.values())))
        else:
            # For Training Set: Create the original one-hot map
            unique_traits = sorted(list(set(self.trait_map.values())))
            self.num_traits = len(unique_traits)
            self.trait_one_hot_map = create_one_hot_dict(unique_traits)
            print(f"--> Initialized dataset with {self.num_traits} trait(s).")

    def __len__(self):
        """Returns the number of samples in the dataset partition."""
        if self.is_train:
            return len(self.samples) * 100
        else:
            return len(self.samples) * 20

    def __getitem__(self, idx):
        """Gets a single data point with power-law subsampling."""
        sample_name = self.samples[idx % len(self.samples)]

        full_counts = self.mutation_counts_df[sample_name].values.astype(np.int64)
        exposures = self.exposures_df[sample_name].values

        total_exposures = np.sum(exposures)
        relative_exposures = exposures / total_exposures if total_exposures > 0 else np.zeros_like(exposures)

        total_mutations = int(np.sum(full_counts))

        # --- Subsampling Logic ---
        if not self.is_train:
            if total_mutations <= 1:
                subsample_size = total_mutations
            else:
                u = random.uniform(0.01, 0.99)
                alpha = 1.15
                subsample_size = round((1 - u) ** (-1 / (alpha - 1)))
        else:
            p = random.random()
            if p < 0.2 or total_mutations <= 10:
                subsample_size = total_mutations
            elif p < 0.4:
                u = random.uniform(0.01, 0.99)
                alpha_sparse = 1.45
                subsample_size = round((1 - u) ** (-1 / (alpha_sparse - 1)))
            else:
                u = random.uniform(0.01, 0.99)
                alpha_original = 1.15
                subsample_size = round((1 - u) ** (-1 / (alpha_original - 1)))

        subsample_size = int(min(max(1, subsample_size), total_mutations))

        if subsample_size == total_mutations:
            counts = full_counts
        else:
            flattened_mutations = np.repeat(np.arange(len(full_counts)), full_counts)
            subsampled_indices = np.random.choice(flattened_mutations, size=subsample_size, replace=False)
            counts, _ = np.histogram(subsampled_indices, bins=np.arange(len(full_counts) + 1))

        trait_name = self.trait_map.get(sample_name, "default_trait")

        trait_one_hot = self.trait_one_hot_map.get(trait_name, [0.0] * self.num_traits)

        return (
            torch.FloatTensor(counts),
            torch.FloatTensor(trait_one_hot),
            torch.FloatTensor(relative_exposures)
        )