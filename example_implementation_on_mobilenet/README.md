# Advanced-AZ-NAS

# 🔍 Advanced AZ-NAS for MobileNet(example) on CIFAR-10

This project implements **Zero-Cost Neural Architecture Search (NAS)** using AZ-NAS ranking to discover efficient MobileNet architectures on the CIFAR-10 dataset. It includes both lightweight proxy-based search and optional real training-based evaluation.
# 📂 Note on This Repository Structure

    # ⚠️ Disclaimer: This folder serves only as an example implementation of the proposed AZ-NAS method applied to the MobileNet search space on CIFAR-10.
    It is not the main implementation used for running comprehensive experiments across multiple models and search spaces.
    For the full pipeline and generalizable NAS framework, please refer to the main file directory.

---

## 📦 Installation

Clone the repository and install the package in development mode:

```bash
git clone https://github.com/xzz17/Advanced-AZ-NAS.git
cd Advanced-AZ-NAS/MobileNet
pip install -e .
```
## 🚀 Zero-Cost NAS Search

Run the following Python script to start a random search using AZ-NAS ranking:

```python
from aznas import run

best_score, best_config = run(n_samples=100)
```
This will:

    Search over randomly sampled MobileNet configurations

    Rank them using AZ-NAS score (based on zero-cost proxies)

    Print the best configuration

    Save it to best_config.json for later use


## 🧪 Train the Best Structure

After the best structure is found and saved, you can train and evaluate it:

python examples/train_best.py

This script will:

    Load the structure from best_config.json

    Build the MobileNet model

    Train it on CIFAR-10 (default 10 epochs)

    Report test accuracy

## 🎯 Real NAS with TPE + Training (Optional)

If you want to perform architecture search using real training (not zero-cost), you can use Optuna's TPE sampler:
```
from aznas.train import tpe_search_with_training

study, results, time_cost = tpe_search_with_training(n_trials=30)
```
This function will:

    Use Optuna TPE to suggest MobileNet configurations

    Train each candidate for 5 epochs

    Track validation accuracy

    Output the best configuration found by actual training
