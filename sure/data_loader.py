# sure/data_loader.py
# This script handles data loading and preprocessing for the SuRe model.
# It defines a PyTorch Dataset class that implements a power-law subsampling
# strategy to train the model effectively on sparse data. It also performs
# a two-level stratified split to create representative validation sets.

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


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
    PyTorch Dataset for loading mutation data.
    - Automatically detects and handles single-trait vs. multi-trait datasets.
    - Performs a two-level stratified split (by trait, then by exposure pattern)
      to create a representative validation set.
    - Implements a power-law subsampling strategy during training and validation
      to simulate data sparsity and improve model robustness.
    """

    def __init__(self, data_dir, is_train=True, validation_fraction=0.15, master_dataset=None):
        self.data_dir = data_dir
        self.is_train = is_train

        if master_dataset is None:
            # This block runs only for the first (training) dataset instance.
            # It performs the data loading and splitting once.
            print("--> Loading data and performing stratified split...")

            self._load_data_files()
            self._handle_traits()
            self._stratified_split(validation_fraction)

        else:
            # This block runs for the validation dataset instance.
            # It avoids redundant work by copying from the master dataset.
            print("--> Initializing validation dataset from master.")
            self.mutation_counts_df = master_dataset.mutation_counts_df
            self.exposures_df = master_dataset.exposures_df
            self.trait_map = master_dataset.trait_map
            self.trait_one_hot_map = master_dataset.trait_one_hot_map
            self.num_traits = master_dataset.num_traits
            self.num_signatures = master_dataset.num_signatures
            self.train_samples = master_dataset.train_samples
            self.val_samples = master_dataset.val_samples

    def _load_data_files(self):
        """Loads and aligns the mutation counts and exposures dataframes."""
        self.mutation_counts_df = pd.read_csv(os.path.join(self.data_dir, "mutation_counts.csv"), index_col=0)
        self.exposures_df = pd.read_csv(os.path.join(self.data_dir, "exposures.csv"), index_col=0)

        self.mutation_counts_df = self.mutation_counts_df.sort_index(axis=0).sort_index(axis=1)
        self.exposures_df = self.exposures_df.sort_index(axis=0).sort_index(axis=1)

        common_samples = self.mutation_counts_df.columns.intersection(self.exposures_df.columns)
        self.mutation_counts_df = self.mutation_counts_df[common_samples]
        self.exposures_df = self.exposures_df[common_samples]
        self.sample_names = common_samples.tolist()

        self.num_samples = len(self.sample_names)
        self.num_signatures = len(self.exposures_df.index)

    def _handle_traits(self):
        """Automatically detects and processes trait information."""
        trait_file_path = os.path.join(self.data_dir, "sample_to_trait.pickle")
        if os.path.exists(trait_file_path):
            print(f"--> Found and loading trait data from: {trait_file_path}")
            with open(trait_file_path, "rb") as f:
                self.trait_map = pickle.load(f)
            self.trait_map = {s: t for s, t in self.trait_map.items() if s in self.sample_names}
        else:
            print("--> 'sample_to_trait.pickle' not found. Assuming a single trait for all samples.")
            self.trait_map = {sample: "default_trait" for sample in self.sample_names}

        unique_traits = sorted(list(set(self.trait_map.values())))
        self.num_traits = len(unique_traits)
        self.trait_one_hot_map = create_one_hot_dict(unique_traits)
        print(f"--> Initialized dataset with {self.num_traits} trait(s).")

    def _stratified_split(self, validation_fraction, n_clusters=4):
        """
        Performs a two-level stratified split:
        1. Stratifies by trait.
        2. Within each trait, stratifies by exposure pattern using KMeans clustering.
        """
        self.train_samples = []
        self.val_samples = []

        samples_by_trait = {trait: [] for trait in self.trait_one_hot_map.keys()}
        for sample, trait in self.trait_map.items():
            samples_by_trait[trait].append(sample)

        np.random.seed(42)

        for trait, samples in samples_by_trait.items():
            if len(samples) < n_clusters:
                np.random.shuffle(samples)
                val_size = int(np.floor(validation_fraction * len(samples)))
                self.val_samples.extend(samples[:val_size])
                self.train_samples.extend(samples[val_size:])
                continue

            trait_exposures = self.exposures_df[samples].T.values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(trait_exposures)

            for cluster_id in range(n_clusters):
                cluster_samples = [s for i, s in enumerate(samples) if cluster_labels[i] == cluster_id]
                np.random.shuffle(cluster_samples)
                val_size = int(np.floor(validation_fraction * len(cluster_samples)))

                if len(cluster_samples) > 1 and val_size == len(cluster_samples):
                    val_size -= 1

                self.val_samples.extend(cluster_samples[:val_size])
                self.train_samples.extend(cluster_samples[val_size:])

        print(
            f"--> Split data into {len(self.train_samples)} training and {len(self.val_samples)} validation samples using two-level stratification.")

    def __len__(self):
        """Returns the number of samples in the dataset partition."""
        if self.is_train:
            return len(self.train_samples) * 100
        else:  # is_validation
            return len(self.val_samples) * 20

    def __getitem__(self, idx):
        """
        Gets a single data point. For both training and validation, this involves
        power-law subsampling to generate sparse profiles.
        """
        samples_to_use = self.train_samples if self.is_train else self.val_samples
        sample_name = samples_to_use[idx % len(samples_to_use)]

        full_counts = self.mutation_counts_df[sample_name].values.astype(np.int64)
        exposures = self.exposures_df[sample_name].values

        total_exposures = np.sum(exposures)
        relative_exposures = exposures / total_exposures if total_exposures > 0 else np.zeros_like(exposures)

        total_mutations = int(np.sum(full_counts))
        use_full_sample = self.is_train and (random.random() < 0.1 or total_mutations <= 1)

        if use_full_sample:
            counts = full_counts
        else:
            u = random.uniform(0.01, 0.99)
            alpha = 1.15
            subsample_size = round((1 - u) ** (-1 / (alpha - 1)))
            subsample_size = int(min(max(1, subsample_size), total_mutations))

            flattened_mutations = np.repeat(np.arange(len(full_counts)), full_counts)
            subsampled_indices = np.random.choice(flattened_mutations, size=subsample_size, replace=False)
            counts, _ = np.histogram(subsampled_indices, bins=np.arange(len(full_counts) + 1))

        trait_name = self.trait_map.get(sample_name, "default_trait")
        trait_one_hot = self.trait_one_hot_map[trait_name]

        return (
            torch.FloatTensor(counts),
            torch.FloatTensor(trait_one_hot),
            torch.FloatTensor(relative_exposures)
        )
