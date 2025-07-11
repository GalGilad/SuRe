Structure
/SuRe/
├── data/
│   ├── brca/
│   │   ├── mutation_counts.csv
│   │   ├── exposures.csv
│   │   └── sample_to_trait.pickle
│   └── pan/
│       ├── ...
├── models/
│   └── brca/
│       └── best_model.pth
├── results/
│   └── brca/
│       └── loss_curve.png
│
├── sure/
│   ├── __init__.py
│   ├── __main__.py        # Main CLI entry point
│   ├── model.py           # SuRe neural network architecture
│   ├── data_loader.py     # PyTorch Dataset and DataLoader logic
│   ├── train.py           # Training loop and functions
│   └── infer.py           # Inference logic
│
└── requirements.txt

Usage
The project is run via a command-line interface.

Training a Model
To train a new model, use the train command. You need to provide the path to a data directory containing mutation_counts.csv, exposures.csv, and optionally sample_to_trait.pickle.

Example for Breast Cancer (BRCA) data:

python -m sure train --data_dir data/brca --num_experts 4 --hidden_units 100 --epochs 100 --batch_size 32 --learning_rate 0.1 --dropout_rate 0.2

The best model (best_model.pth) will be saved in models/brca/ and a loss curve plot will be saved in results/brca/.

Running Inference
To predict exposures for new data using a trained model, use the infer command.

Example:

python -m sure infer --model_path models/brca/best_model.pth --mutation_counts_path data/brca/new_unseen_counts.csv --output_file results/brca/predicted_exposures.csv

This will generate a CSV file with the predicted relative exposures for the samples in new_unseen_counts.csv.
