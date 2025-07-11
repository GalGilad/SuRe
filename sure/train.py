# sure/train.py
# This script orchestrates the training process for the SuRe model.
# It initializes the model and datasets, runs the main training loop,
# performs validation, and handles model saving and early stopping.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from .model import SuReNet
from .data_loader import MutationDataset


def start_training(args):
    """
    Initializes and runs the entire training process based on command-line arguments.
    """
    print("--- Starting SuRe Model Training ---")

    # --- Setup Directories ---
    data_subfolder = os.path.basename(os.path.normpath(args.data_dir))
    model_dir = os.path.join("models", data_subfolder)
    results_dir = os.path.join("results", data_subfolder)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- Initialize Datasets and Dataloaders ---
    print("Loading and preparing data...")
    # Create the main training dataset instance. This performs data loading and splitting once.
    train_dataset = MutationDataset(data_dir=args.data_dir, is_train=True)
    # Create the validation dataset, referencing the training instance to avoid redundant work.
    val_dataset = MutationDataset(data_dir=args.data_dir, is_train=False, master_dataset=train_dataset)

    # Infer number of signatures and traits from the dataset
    num_signatures = train_dataset.num_signatures
    num_traits = train_dataset.num_traits
    print(f"--> Inferred number of signatures from 'exposures.csv': {num_signatures}")

    # Save the trait mapping once, before training begins, for consistent encoding
    trait_map_path = os.path.join(model_dir, "trait_map.json")
    with open(trait_map_path, 'w') as f:
        json.dump(train_dataset.trait_one_hot_map, f, indent=4)
    print(f"--> Saved trait map for consistent encoding to: {trait_map_path}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize Model, Optimizer, and Loss ---
    print("Initializing model...")
    model = SuReNet(
        num_signatures=num_signatures,
        num_traits=num_traits,
        num_experts=args.num_experts,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # --- Training Loop ---
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience_epochs = 10

    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        total_train_loss = 0
        for i, (counts, traits, exposures) in enumerate(train_loader):
            counts, traits, exposures = counts.to(device), traits.to(device), exposures.to(device)

            optimizer.zero_grad()
            predictions = model(counts, traits)
            loss = criterion(predictions, exposures)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for counts, traits, exposures in val_loader:
                counts, traits, exposures = counts.to(device), traits.to(device), exposures.to(device)
                predictions = model(counts, traits)
                loss = criterion(predictions, exposures)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping and Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            model_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)

            config_path = os.path.join(model_dir, "config.json")
            config_data = {
                "num_signatures": num_signatures,
                "num_traits": num_traits,
                "num_experts": args.num_experts,
                "hidden_units": args.hidden_units,
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            print(f"Validation loss improved. Saved model and config file to {model_dir}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience_epochs}")

        if patience_counter >= patience_epochs:
            print("Early stopping triggered.")
            break

    print("--- Training Finished ---")

    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    print(f"Saved loss curve to {loss_curve_path}")
