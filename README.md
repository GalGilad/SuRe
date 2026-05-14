## Project Structure

```
SuRe/
│
├── main.py                  # The single CLI entry point for all commands
├── requirements.txt         
├── README.md                
│
├── sure/                    # Core Python package
│   ├── __init__.py          
│   ├── model.py             
│   ├── data_loader.py       
│   ├── train.py             
│   ├── evaluate.py          
│   └── infer.py             
│
├── data/                    
│   ├── brca/                # Breast cancer dataset (contains /train, /val, /test)
│   └── pan/                 # Pan-cancer dataset (contains /train, /val, /test)
│
├── models/                  # Saved models and configurations
│   ├── brca/                # BRCA model (best_model.pth, config.json, trait_map.json)
│   └── pan/                 # Pan-cancer model (best_model.pth, config.json, trait_map.json)
│
└── results/                 # Auto-generated outputs

## Data Directory Setup

For the training script to work correctly, the data must be organized into train and val subdirectories inside your main data folder (e.g., data/brca/train/ and data/brca/val/).

Each of these subdirectories must contain:

mutation_counts.csv: A CSV file where columns are sample IDs and rows are the 96 mutation categories.

exposures.csv: A CSV file with the ground-truth exposures. Columns are sample IDs and rows are signature names.

It can optionally contain:

sample_to_trait.pickle: A Python pickle file containing a dictionary mapping sample IDs to their corresponding trait (e.g., cancer tissue type). If absent, the script assumes all samples belong to a single default trait.


## Usage

The project is run via a single command-line interface: main.py.

1. Evaluating a Pre-Trained Model
The evaluate command assesses a trained model's performance across varying levels of data sparsity (m = 300, 100, 30, 10, 6, 3 mutations per sample).

Example Command (BRCA):

Bash
python main.py evaluate --data_dir data/brca/test --model_path models/brca/best_model.pth
Outputs: Saves an evaluation metrics CSV and a performance summary plot to results/brca/ (or the respective dataset folder).

2. Training a Model
The train command trains a new model. It automatically infers the number of signatures and traits from your data files.

Example Command (Pan-Cancer):

Bash
python main.py train --data_dir data/pan --num_experts 8 --hidden_units 500
Outputs: Trains the model and saves best_model.pth, config.json, and trait_map.json to the models/pan/ directory. It also saves a loss_curve.png to the results/pan/ directory.

3. Running Inference on New Data
The infer command uses a trained model to predict exposures for new, unseen mutation data. It automatically loads the model's architecture from the saved configuration files.

Example Command:

Bash
python main.py infer --model_path models/brca/best_model.pth --mutation_counts_path path/to/mutation_counts.csv --output_file inferred_exposures.csv
Optional Trait Mapping: If your inference data spans multiple cancer types, pass the pickle map using --trait_path path/to/sample_to_trait.pickle. If omitted, the script assumes a single default trait.
