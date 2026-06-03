# Federated Learning — Client Selection Techniques and Model Pruning
### Machine Learning & Deep Learning — Politecnico di Torino

## Overview
This project explores advanced strategies for **Federated Learning (FL)** — a machine learning paradigm where a model is trained across multiple decentralized clients without sharing raw data. The focus is on three key challenges in FL:

1. **Client Selection** — which clients participate in each training round
2. **Domain Generalisation** — making models robust across heterogeneous data distributions
3. **Model Pruning** — reducing model size and communication cost without sacrificing performance

The goal was to improve the efficiency and generalisability of federated models under realistic heterogeneous data conditions.

## Techniques Implemented

### Client Selection Strategies
- **Random selection** (baseline)
- **Loss-based selection** — prioritising clients with higher local loss
- **Power-of-Choice** — biased client selection to improve convergence on heterogeneous data

### Domain Generalisation
- **FedAvg** (baseline federation algorithm)
- Experiments with data heterogeneity (non-IID distributions across clients)

### Model Pruning
- **Structured pruning** — removing entire filters/neurons to reduce model size
- Evaluated trade-off between model compression and accuracy retention

## Dataset
Experiments conducted on standard FL benchmarks with non-IID data partitioning to simulate realistic heterogeneous client distributions.

## How to Run

```bash
# Clone the repo
git clone https://github.com/siinvictus/FederatedLearningClientSelectionTechniquesandPruning.git
cd FederatedLearningClientSelectionTechniquesandPruning/MLDL23-FL-project-main

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
```

## Project Structure
```
MLDL23-FL-project-main/
├── main.py                  ← entry point
├── server.py                ← federation server logic
├── client.py                ← client training logic
├── datasets/                ← data loading and partitioning
├── models/                  ← model architectures
└── utils/                   ← helper functions
Report_MLDL_Federated Learning.pdf  ← full project report
```

## Tech Stack
Python, PyTorch, NumPy

## Authors
Silva Bashllari & collaborators — Politecnico di Torino, MLDL 2023

## License
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
