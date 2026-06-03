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

## Tech Stack
Python, PyTorch, NumPy etc.

## Authors
Silva Bashllari & collaborators — Politecnico di Torino, MLDL January 2024

## License
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)
