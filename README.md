# Advanced_AZ_NAS

**Advanced_AZ_NAS** is an automated model training and zero-cost NAS (Neural Architecture Search) ranking framework that supports both CNN and Transformer-based models. It integrates [Optuna](https://optuna.org/) for hyperparameter search and multiple zero-cost proxy metrics (like expressivity, trainability, synflow, etc.) to evaluate model quality without requiring full training.

## 🚀 Features

- 🔍 Automatic architecture search via Optuna (supports TPE or Random search)
- 🧠 Zero-cost proxy metrics to estimate model quality without full training
- 🧪 Supports both CNNs and Transformers (ResNet, MobileNet, BERT, etc.)
- 🏆 Final training & evaluation of top-ranked model
- 📦 Based on PyTorch, HuggingFace Transformers, and Optuna

## 📦 Installation

Install all dependencies via:

```bash
pip install -r requirements.txt
```


## 🧪 Quick Start

```bash
git clone https://github.com/xzz17/Advanced-AZ-NAS.git
cd Advanced_AZ_NAS
pip install -e .
```

## 📘 API: auto_train_model(...)

```python
auto_train_model(
    model,
    train_dataset,
    test_dataset,
    input_shape,
    num_trials=10,
    save_path="best_model.pth",
    checkpoint="bert-base-uncased",
    batch_size=64,
    tokenizer=None
) → (best_model, accuracy)
```

### 🔧 Parameters

| Parameter      | Type          | Description                                                               |
|----------------|---------------|---------------------------------------------------------------------------|
| `model`        | `nn.Module`   | Base model to start from (e.g., MobileNet or BERT)                           |
| `train_dataset`| `Dataset`     | Training set (HuggingFace or PyTorch Dataset)                             |
| `test_dataset` | `Dataset`     | Evaluation set                                                            |
| `input_shape`  | `tuple`       | Shape of dummy input for proxy computation (e.g., `(1, 128)`)             |
| `num_trials`   | `int`         | Number of trials for hyperparameter search                                |
| `save_path`    | `str`         | Path to save the best trained model                                       |
| `checkpoint`   | `str`         | (Optional) HuggingFace model checkpoint name (only required for Transformer models)                     |
| `batch_size`   | `int`         | Batch size for training                                                   |
| `tokenizer`    | `Tokenizer`   | (Optional) HuggingFace tokenizer (only required for Transformer models)   |

### 📤 Returns

- `best_model` — the best model found and trained  
- `accuracy` — final accuracy on the test set

## 📊 Zero-Cost Proxies

| Proxy         | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| **Expressivity**  | Entropy of intermediate features (higher = more expressive)             |
| **Progressivity** | Measures expressivity progression across layers                        |
| **Trainability**  | Gradient norm of loss w.r.t input (higher = easier to optimize)         |
| **Synflow**       | Synaptic Flow score, indicating parameter saliency                      |
| **Complexity**    | Model size in log-scale (#params)                                       |

## 🧪 Examples

We provide ready-to-run scripts for different model families:

### 🟦 MobileNet Example (CNN)

```bash
python examples/mobilenet_example.py
```

Loads pretrained MobileNetV2 on CIFAR-10 and runs NAS with zero-cost proxies.

### 🟦 ResNet Example (CNN)

```bash
python examples/resnet_example.py
```

Uses ResNet18 backbone and searches over block configs using CIFAR-10.

### 🟨 BERT Example (Transformer)

```bash
python examples/bert_example.py
```

Loads TinyBERT on the IMDb dataset and runs search + training using HuggingFace Trainer.



If you find this project helpful, feel free to ⭐ star the repo and contribute!

