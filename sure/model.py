# sure/model.py
# This script defines the SuRe neural network architecture using PyTorch.
# It implements the Mixture-of-Experts (MoE) model as described in the manuscript.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertModule(nn.Module):
    """
    Represents an individual expert network in the Mixture-of-Experts model.
    Each expert is a multi-layer perceptron that specializes in a subset of the data.
    """

    def __init__(self, input_size, output_size, hidden_size, dropout_rate):
        super(ExpertModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        return self.output(x)


class GatingNetwork(nn.Module):
    """
    The gating network that learns to assign weights to each expert based on the input.
    The output is a probability distribution over the experts.
    """

    def __init__(self, input_size, num_experts, hidden_size, dropout_rate):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_experts)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        # Use softmax to get a probability distribution over the experts
        return F.softmax(self.output(x), dim=1)


class SuReNet(nn.Module):
    """
    The main SuRe Mixture-of-Experts (MoE) model.
    It combines a gating network and multiple expert modules to produce a weighted
    prediction of mutational signature exposures.
    """

    def __init__(self, num_signatures, num_traits, num_experts, hidden_units, dropout_rate):
        super(SuReNet, self).__init__()
        self.num_signatures = num_signatures
        self.num_traits = num_traits
        self.num_categories = 96  # Standard number of mutation categories
        self.input_size = self.num_categories + self.num_traits

        # A preliminary layer to process the concatenated input
        self.input_transform = nn.Linear(self.input_size, self.input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # A list of expert modules
        self.experts = nn.ModuleList(
            [ExpertModule(self.input_size, self.num_signatures, hidden_units, dropout_rate)
             for _ in range(num_experts)]
        )

        # The gating network
        self.gating_network = GatingNetwork(self.input_size, num_experts, hidden_units, dropout_rate)

    def forward(self, mutation_counts, trait_vectors):
        """
        Defines the forward pass of the model.

        Args:
            mutation_counts (torch.Tensor): Tensor of mutation counts (batch_size, 96).
            trait_vectors (torch.Tensor): One-hot encoded tensor of traits.

        Returns:
            torch.Tensor: The final predicted relative exposures.
        """
        # Concatenate mutation counts and trait vectors to form the input
        x = torch.cat((mutation_counts, trait_vectors), dim=1)

        # Initial transformation of the combined input
        transformed_x = self.dropout(self.relu(self.input_transform(x)))

        # Get the weights for each expert from the gating network
        gate_weights = self.gating_network(transformed_x)  # Shape: (batch_size, num_experts)

        # Get the outputs from each expert module
        expert_outputs = [expert(transformed_x) for expert in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=1)  # Shape: (batch_size, num_experts, num_signatures)

        # Weight the expert outputs using the gate weights
        # Unsqueeze gate_weights to (batch_size, num_experts, 1) for broadcasting
        weighted_expert_outputs = expert_outputs_stacked * gate_weights.unsqueeze(-1)

        # Sum the weighted outputs to get the final combined result
        combined_output = torch.sum(weighted_expert_outputs, dim=1)  # Shape: (batch_size, num_signatures)

        # Apply softmax to the final output to produce a probability distribution
        return F.softmax(combined_output, dim=1)
