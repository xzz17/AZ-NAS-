import time
import numpy as np
import json
import torch
import gc

from .mobilenet import build_mobilenet_from_config
from .scores import (
    expressivity_score,
    progressivity_score,
    trainability_score,
    complexity_score,
    synflow_score
)
from .ranking import az_nas_ranking

def random_sample_config():
    """
    Randomly sample a MobileNet configuration.

    Returns:
        dict: A dictionary containing architecture hyperparameters.
    """
    import random
    return {
        'activation': random.choice(['relu', 'swish']),
        'kernel_size': random.choice([3, 5]),
        'expand_ratio': random.choice([3, 6]),
        'prune_ratio': random.choice([0.5, 0.75, 1.0]),
        'use_se': random.choice([True, False]),
        'num_blocks': random.choice([8, 12])
    }

def run_zc_random_search(n_samples=96, seed=123):
    """
    Run Zero-Cost NAS by randomly sampling architectures and ranking them with AZ-NAS.

    Args:
        n_samples (int): Number of architectures to evaluate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (aznas_scores, configs, scores_history)
            - aznas_scores: Final scores for each architecture (lower is better)
            - configs: Corresponding configuration list
            - scores_history: Raw proxy scores for all samples
    """
    random = __import__('random')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize proxy score history and config cache
    scores_history = {k: [] for k in ["expressivity", "progressivity", "trainability", "complexity", "synflow"]}
    configs = []
    config_cache = {}

    for i in range(n_samples):
        config = random_sample_config()
        config_key = str(config)
        if config_key in config_cache:
            print(f"Skipping duplicate config at sample {i}")
            continue
        config_cache[config_key] = True

        # Build model and evaluate proxy metrics
        model = build_mobilenet_from_config(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        scores = {
            "expressivity": expressivity_score(model),
            "progressivity": progressivity_score(model),
            "trainability": trainability_score(model),
            "complexity": complexity_score(model),
            "synflow": synflow_score(model)
        }

        # Save scores and config
        for k in scores:
            scores_history[k].append(scores[k])
        configs.append(config)

        # Free up memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Sample {i+1}/{n_samples} done")

    return az_nas_ranking(scores_history), configs, scores_history

def run(n_samples=300, seed=123):
    """
    Main entry for Zero-Cost NAS random search.

    Args:
        n_samples (int): Number of architecture samples to evaluate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (best_score, best_config)
    """
    print("Starting Zero-Cost NAS (Random Search) for MobileNet on CIFAR-10...")

    # Define ranking direction for each proxy
    proxy_ordering = {
        'expressivity': 'desc',
        'progressivity': 'asc',
        'trainability': 'abs1',
        'complexity': 'desc',
        'synflow': 'desc'
    }

    start = time.time()
    aznas_scores, zc_configs, scores_history = run_zc_random_search(n_samples=n_samples, seed=seed)
    end = time.time()
    print(f"\nTotal NAS Search Time: {end - start:.2f} seconds")

    # Get best architecture by AZ-NAS score
    best_idx = np.argmin(aznas_scores)
    best_config = zc_configs[best_idx]
    best_score = aznas_scores[best_idx]

    print("\n=== Best Zero-Cost NAS Structure ===")
    print(f"AZ-NAS Score: {best_score:.4f}")
    print(f"Config: {best_config}")

    # Save best config to file for later training
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
        print("✅ Saved best_config.json")

    return best_score, best_config
