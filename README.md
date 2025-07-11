## Project Structure

```
/SuRe/
├── data/
│   ├── brca/
│   │   ├── mutation_counts.csv
│   │   ├── exposures.csv
│   │   └── sample_to_trait.pickle  (Optional)
│   └── pan/
│       ├── ...
├── models/
│   ├── brca/
│   │   ├── best_model.pth
│   │   ├── config.json
│   │   └── trait_map.json
│   └── pan/
│       ├── ...
├── sure/
│   ├── __init__.py
│   ├── __main__.py        # Main CLI entry point
│   ├── model.py           # SuRe neural network architecture
│   ├── data_loader.py     # Data loading
│   ├── train.py           # Training logic
│   └── infer.py           # Inference logic
│
└── requirements.txt
```

## Data Directory Setup

For the script to work correctly, the data should be organized in a subdirectory inside the `data/` folder (e.g., `data/brca/`). This directory must contain:
* `mutation_counts.csv`: A CSV file where columns are sample IDs and rows are the 96 mutation categories.
* `exposures.csv`: A CSV file with the ground-truth exposures. Columns are sample IDs and rows are signature names.

It can optionally contain:
* `sample_to_trait.pickle`: A Python pickle file containing a dictionary that maps sample IDs to their corresponding trait (e.g., cancer type). If this file is not present, the script will assume all samples belong to a single trait.

## Usage

The project is run via a command-line interface.

### Training a Model

The `train` command trains a new model. It automatically infers the number of signatures and traits from your data files.

**Example Command:**
```bash
python -m sure train --data_dir data/brca --num_experts 4 --hidden_units 100 --epochs 100 --batch_size 32 --learning_rate 0.1 --dropout_rate 0.2
```

This command will:
* Load data from the `data/brca` directory.
* Train a model.
* Save the best model (`best_model.pth`), its configuration (`config.json`), and the trait mapping (`trait_map.json`) to the `models/brca/` directory.
* Save a plot of the training and validation loss (`loss_curve.png`) to the `results/brca/` directory.

### Running Inference

The `infer` command uses a trained model to predict exposures for new data. It automatically loads the model's architecture from the saved configuration files.

**Example Command:**
```bash
python -m sure infer --model_path models/brca/best_model.pth --mutation_counts_path data/brca/new_unseen_counts.csv --trait_path data/brca/new_unseen_traits.pickle --output_file results/brca/predicted_exposures.csv
```
* The `--trait_path` is optional. If not provided, the script will assume all samples belong to a single trait.
