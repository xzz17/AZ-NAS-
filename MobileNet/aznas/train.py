import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import gc

from .mobilenet import build_mobilenet_from_config
from .dataloader import get_cifar10_loaders


def train_and_evaluate(model, train_loader, test_loader, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
        acc = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += outputs.argmax(1).eq(targets).sum().item()
    final_acc = 100. * correct / len(test_loader.dataset)
    return final_acc


# Optuna search space definition
def sample_config(trial):
    return {
        'activation': trial.suggest_categorical('activation', ['relu', 'swish']),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
        'expand_ratio': trial.suggest_categorical('expand_ratio', [3, 6]),
        'prune_ratio': trial.suggest_categorical('prune_ratio', [0.5, 0.75, 1.0]),
        'use_se': trial.suggest_categorical('use_se', [True, False]),
        'num_blocks': trial.suggest_categorical('num_blocks', [8, 12])
    }


def evaluate_with_training(trial, results_list):
    config = sample_config(trial)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_mobilenet_from_config(config).to(device)
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)

    try:
        acc = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    except RuntimeError as e:
        print(f"⚠️ RuntimeError during training: {e}")
        acc = 0.0  # treat failure as 0 accuracy

    results_list.append({'config': config, 'accuracy': acc})
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return -acc


def tpe_search_with_training(n_trials=10, seed=123):
    results_list = []
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=15)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    start_time = time.time()
    study.optimize(lambda trial: evaluate_with_training(trial, results_list), n_trials=n_trials)
    end_time = time.time()
    search_time = end_time - start_time

    print("\n===== Best Architecture Found via TPE (with real training) =====")
    print(f"Best Config: {study.best_params}")
    print(f"Best Accuracy: {-study.best_value:.2f}%")
    print(f"TPE Search Time: {search_time:.2f} seconds")
    return study, results_list, search_time
