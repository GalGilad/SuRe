/sure_project/
├── data/
│   ├── brca/
│   │   ├── mutation_counts.csv
│   │   ├── exposures.csv
│   │   └── sample_to_trait.pickle  (Optional)
│   └── pan/
│       ├── ...
├── models/
│   └── brca/
│       ├── best_model.pth
│       ├── config.json
│       └── trait_map.json
├── results/
│   └── brca/
│       ├── loss_curve.png
│       └── evaluation_summary.png
│
├── sure/
│   ├── init.py
│   ├── main.py        # Main CLI entry point
│   ├── model.py           # SuRe neural network architecture
│   ├── data_loader.py     # Data loading and stratified splitting
│   ├── train.py           # Training logic
│   ├── infer.py           # Inference logic
│   └── evaluate.py        # Local evaluation logic
│
└── requirements.txt
