
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


from datasets import load_dataset
from sklearn.metrics import accuracy_score


import optuna

import numpy as np


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, activation, use_se):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_se = use_se
        act_fn = nn.ReLU(inplace=True) if activation == 'relu' else nn.SiLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid()
            )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out += self.shortcut(x)
        return out

def build_mobilenet_from_config(config, num_classes=10):
    layers = []
    in_channels = 3
    base_channels = int(32 * config['prune_ratio'])

    for i in range(config['num_blocks']):
        scale_factor = 2 ** (i // 4)
        out_channels = int(base_channels * scale_factor)
        layers.append(MobileNetBlock(
            in_channels, out_channels, config['kernel_size'],
            config['expand_ratio'], config['activation'], config['use_se']))
        in_channels = out_channels

    layers.append(nn.AdaptiveAvgPool2d((1,1)))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(out_channels, num_classes))
    return nn.Sequential(*layers)





def get_model_type(model):
    model_name = model.__class__.__name__.lower()
    if "mobilenet" in model_name or "efficientnet" in model_name or "shufflenet" in model_name:
        return "cnn_mobilenet"
    elif "resnet" in model_name or "wideresnet" in model_name:
        return "cnn_resnet"
    elif "bert" in model_name or "transformer" in model_name:
        return "transformer"
    else:
        return "unknown"

def search_space(trial, model_type):
    if model_type == "cnn_resnet":
        return {
            "num_blocks": trial.suggest_int("num_blocks", 6, 20),
            "base_channels": trial.suggest_categorical("base_channels", [64, 128, 256]),
            "block_type": trial.suggest_categorical("block_type", ["basic", "bottleneck"]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.3)
        }
    elif model_type == "transformer":
        return {
            "num_hidden_layers": trial.suggest_int("num_hidden_layers", 2, 6),
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
            "num_attention_heads": trial.suggest_categorical("num_attention_heads", [2, 4, 8]),
            "intermediate_size": trial.suggest_categorical("intermediate_size", [512, 1024, 2048]),

            #"attention_probs_dropout_prob": trial.suggest_float("attention_probs_dropout_prob", 0.1, 0.3),
            #"hidden_dropout_prob": trial.suggest_float("hidden_dropout_prob", 0.1, 0.3),
            "classifier_dropout": trial.suggest_float("classifier_dropout", 0.1, 0.3),
        }
    else:
        return {
            'activation': trial.suggest_categorical('activation', ['relu', 'swish']),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            'expand_ratio': trial.suggest_categorical('expand_ratio', [3, 6]),
            'prune_ratio': trial.suggest_categorical('prune_ratio', [0.5, 0.75, 1.0]),
            'use_se': trial.suggest_categorical('use_se', [True, False]),
            'num_blocks': trial.suggest_categorical('num_blocks', [8, 12])
        }

#Resnet Zero Proxies
def expressivity_score_resnet(model, dummy_images):
    model.eval()
    with torch.no_grad():
        features = model.avgpool(model.layer3(model.layer2(model.layer1(F.relu(model.bn1(model.conv1(dummy_images)))))))
        features = features.view(features.size(0), -1)

        features_mean = features.mean(dim=0, keepdim=True)
        centered_features = features - features_mean
        cov_matrix = centered_features.T @ centered_features / (features.shape[0] - 1)

        eigvals = torch.linalg.eigvals(cov_matrix).real
        eigvals = eigvals / eigvals.sum()
        entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum().item()

    return entropy

def progressivity_score_resnet(model, dummy_images):
    model.eval()
    with torch.no_grad():
        layers_out = []
        out = F.relu(model.bn1(model.conv1(dummy_images)))
        layers_out.append(out.clone())
        out = model.layer1(out); layers_out.append(out.clone())
        out = model.layer2(out); layers_out.append(out.clone())
        out = model.layer3(out); layers_out.append(out.clone())

        sE_layers = []
        for layer_out in layers_out:
            features = model.avgpool(layer_out)
            features = features.view(features.size(0), -1)
            features_mean = features.mean(dim=0, keepdim=True)
            centered_features = features - features_mean
            cov_matrix = centered_features.T @ centered_features / (features.shape[0] - 1)

            eigvals = torch.linalg.eigvals(cov_matrix).real
            eigvals = eigvals / eigvals.sum()
            entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum().item()
            sE_layers.append(entropy)

        sP = min([sE_layers[i] - sE_layers[i-1] for i in range(1, len(sE_layers))])

    return sP

def trainability_score_resnet(model, dummy_images):
    model.train()
    dummy_input = dummy_images.clone().detach().requires_grad_()

    outputs = model(dummy_input)
    loss = outputs.norm()
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)

    sigma = grads.norm().item()
    return np.log(sigma + 1e-8)

def complexity_score(model):
    complexity = sum(p.numel() for p in model.parameters())
    return np.log(complexity + 1)

def synflow_score_resnet(model, dummy_images):
    model.eval()

    # Disable BatchNorm tracking (crucial for single-batch SynFlow)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.training = False  # 👈 强制关闭 BatchNorm 训练状态

    # Use constant input for SynFlow
    input_data = torch.ones_like(dummy_images)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.abs()

    model.zero_grad()
    output = model(input_data).sum()

    if torch.isnan(output) or torch.isinf(output):
        print("[SynFlow] Output is NaN or Inf.")
        return float('nan')

    output.backward()

    score = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"[SynFlow] Gradient NaN/Inf in: {name}")
                return float('nan')
            score += (param.grad * param).abs().sum().item()

    return score

#Tiny Bert

# **Zero-cost proxy: Expressivity**
def expressivity_score_transformer(model, tokenizer):
    model.eval()
    with torch.no_grad():
      dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(next(model.parameters()).device)
      attention_mask = torch.ones_like(dummy_input).to(next(model.parameters()).device)


      outputs = model.bert(dummy_input, attention_mask=attention_mask, output_hidden_states=True)
      hidden_states = outputs.hidden_states[-1]  # shape: (batch, seq_len, hidden_size)

      features_mean = hidden_states.mean(dim=1, keepdim=True)  # shape: (batch, 1, hidden_size)
      centered_features = hidden_states - features_mean

      cov_matrix = centered_features.squeeze(0).T @ centered_features.squeeze(0) / (hidden_states.shape[1] - 1)


      eigvals = torch.linalg.eigvals(cov_matrix).real
      eigvals = eigvals / eigvals.sum()

      entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum().item()

    return entropy

# **Zero-cost proxy: Progressivity**
def progressivity_score_transformer(model, tokenizer):
    model.eval()
    with torch.no_grad():
      device = next(model.parameters()).device
      dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).to(device)
      attention_mask = torch.ones_like(dummy_input).to(device)
      outputs = model.bert(dummy_input, attention_mask=attention_mask, output_hidden_states=True)
      hidden_states = outputs.hidden_states  

      sE_layers = []
      for layer_out in hidden_states[1:]:  
          features_mean = layer_out.mean(dim=1, keepdim=True)
          centered_features = layer_out - features_mean
          cov_matrix = centered_features.squeeze(0).T @ centered_features.squeeze(0) / (layer_out.shape[1] - 1)
          eigvals = torch.linalg.eigvals(cov_matrix).real
          eigvals = eigvals / eigvals.sum()
          entropy = -(eigvals * torch.log(eigvals + 1e-8)).sum().item()
          sE_layers.append(entropy)

      sP = min([sE_layers[i] - sE_layers[i-1] for i in range(1, len(sE_layers))])

    return sP

# **Zero-cost proxy: Trainability**
def trainability_score_transformer(model, tokenizer):
    model.train()
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128)).float().to(next(model.parameters()).device).requires_grad_()
    attention_mask = torch.ones_like(dummy_input)

    outputs = model.bert(dummy_input.long(), attention_mask=attention_mask, output_hidden_states=False)
    loss = outputs.last_hidden_state.norm()
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grads = torch.cat(grads)

    sigma = grads.norm().item()
    return np.log(sigma + 1e-8)


#  SynFlow **
def synflow_score_transformer(model, input_shape=(1, 128)):
    model.eval()
    device = 'cuda'
    model.to(device)
    input_ids = torch.ones(input_shape, dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)

    for param in model.parameters():
        param.data = param.data.abs()

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits.sum()

    outputs.backward()

    
    score = sum((param.grad * param).abs().sum().item() for param in model.parameters() if param.grad is not None)

    return score

#MobileNet

def expressivity_score_CNN(model, input_shape=(1,3,32,32), device='cuda'):
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        for layer in model:
            if isinstance(layer, (nn.Conv2d, MobileNetBlock)):
                features = layer(dummy_input)
                break
        
        prob = torch.softmax(features.flatten(), dim=0)
        entropy = -(prob * torch.log(prob + 1e-8)).sum().item()
    return entropy

def progressivity_score_CNN(model, input_shape=(1,3,32,32), device='cuda'):
    model.eval()
    entropies = []
    x = torch.randn(input_shape).to(device)

    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(x, torch.Tensor):
                prob = torch.softmax(x.flatten(), dim=0)
                entropy = -(prob * torch.log(prob + 1e-8)).sum().item()
                entropies.append(entropy)
    diffs = [entropies[i+1] - entropies[i] for i in range(len(entropies)-1)]
    return min(diffs) if diffs else 0

def trainability_score_CNN(model, input_shape=(1,3,32,32), device='cuda'):
    model.train()
    dummy_input = torch.randn(input_shape, requires_grad=True).to(device)
    output = model(dummy_input)
    output.sum().backward()
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    grads = torch.cat(grads)
    sigma = grads.norm().item()
    return np.log(sigma + 1e-8)


def synflow_score_CNN(model, input_shape=(1,3,32,32), device='cuda'):
    model.to(device).eval()
    for p in model.parameters():
        p.data = p.data.abs()
    input_tensor = torch.ones(input_shape).to(device)
    model.zero_grad()
    model(input_tensor).sum().backward()
    score = sum((p.grad * p.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
    return score


def az_nas_ranking(scores_dict):
    proxies = list(scores_dict.keys())
    m = len(scores_dict[proxies[0]])

    ranks = {}
    for proxy in proxies:
        scores = np.array(scores_dict[proxy])

        if len(scores) < 2:
            ranks[proxy] = np.ones_like(scores)
        else:
            if proxy in ["expressivity", "progressivity", "synflow"]:
                rank = np.argsort(np.argsort(-scores)) + 1
            elif proxy == "trainability":
                rank = np.argsort(np.argsort(abs(1 - scores))) + 1
            elif proxy == "complexity":
                rank = np.argsort(np.argsort(scores)) + 1

            ranks[proxy] = rank

    final_scores = []
    for i in range(m):
        score = sum([np.log(max(ranks[p][i] / m, 1e-8)) for p in proxies])
        final_scores.append(score)
    return final_scores

def az_nas_ranking_CNN(scores_dict, proxy_ordering=None):
    m = len(next(iter(scores_dict.values())))
    ranks = {}
    if proxy_ordering is None:
        proxy_ordering = {k: 'desc' for k in scores_dict}

    for k, scores in scores_dict.items():
        scores = np.array(scores)
        order = proxy_ordering.get(k, 'desc')
        if order == 'asc':
            r = np.argsort(np.argsort(scores)) + 1
        elif order == 'desc':
            r = np.argsort(np.argsort(-scores)) + 1
        elif order == 'abs1':  
            r = np.argsort(np.argsort(np.abs(1 - scores))) + 1
        else:
            raise ValueError(f"Unknown ordering for proxy '{k}': {order}")
        ranks[k] = r

    final_scores = [
        sum(np.log(max(ranks[p][i]/m, 1e-8)) for p in ranks)
        for i in range(m)
    ]
    return final_scores

proxy_ordering = {
    'expressivity': 'desc',
    'progressivity': 'asc',
    'trainability': 'abs1',
    'complexity': 'desc',
    'synflow': 'desc'
}

def zero_cost_proxies(model, input_shape, device='cuda', train_loader=None, train_dataset=None, test_dataset=None, tokenizer=None):
    model_type = get_model_type(model)
    global scores_history

    if model_type == "transformer":
        expressivity = expressivity_score_transformer(model, tokenizer)
        progressivity = progressivity_score_transformer(model, tokenizer)
        trainability = trainability_score_transformer(model, tokenizer)
        synflow = synflow_score_transformer(model)
    elif model_type == "cnn_resnet":
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        dummy_images, dummy_labels = next(iter(train_loader))
        dummy_images = dummy_images.to(device)
        expressivity = expressivity_score_resnet(model, dummy_images)
        progressivity = progressivity_score_resnet(model, dummy_images)
        trainability = trainability_score_resnet(model, dummy_images)
        synflow = synflow_score_resnet(model, dummy_images)
    else:
        expressivity = expressivity_score_CNN(model,input_shape, device=device)
        progressivity = progressivity_score_CNN(model,input_shape, device=device)
        trainability = trainability_score_CNN(model,input_shape, device=device)
        synflow = synflow_score_CNN(model,input_shape, device=device)

    complexity = complexity_score(model)

    scores_dict = {
      "expressivity": expressivity,  
      "complexity": complexity,
      "trainability": trainability,
      "synflow": synflow,
      "progressivity": progressivity
    }

    for key in scores_history:
        scores_history[key].append(scores_dict[key])
    return scores_history

def initialize_model_from_params(model_type, params):
    if model_type == "cnn_resnet":
        if params["num_blocks"] >= 10:
            return models.resnet50(pretrained=False, num_classes=10)  
        else:
            return models.resnet18(pretrained=False, num_classes=10)  
    elif model_type == "cnn_mobilenet":
        return models.mobilenet_v2(pretrained=False, num_classes=10)
    else:
        raise ValueError("Unsupported model type: " + model_type)




from transformers import Trainer, TrainingArguments

def train_and_evaluate(model, train_dataset, test_dataset, model_type, num_epochs=10, batch_size=64, device='cuda'):
    model.to(device)

    if model_type == "transformer":
        training_args = TrainingArguments(
          output_dir="./results",
          num_train_epochs=3,
          per_device_train_batch_size=8,
          per_device_eval_batch_size=8,
          evaluation_strategy="epoch",
          save_strategy="no",
          logging_dir="./logs",
          logging_strategy="steps",  
          logging_steps=10,  
          fp16=True if torch.cuda.is_available() else False,
          report_to="none",
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=lambda p: {"eval_accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))}
        )

        trainer.train()
        eval_results = trainer.evaluate()
        accuracy = eval_results["eval_accuracy"]  
    else:
        
        if model_type == "cnn_mobilenet":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        elif model_type == "cnn_resnet":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                optimizer.step()
                total_loss += loss.item()
                correct += outputs.argmax(1).eq(labels).sum().item()
            acc = 100. * correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += outputs.argmax(1).eq(labels).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test Accuracy ({model_type}): {accuracy:.4f}')
    return model, accuracy

scores_history = {
        "expressivity": [],
        "complexity": [],
        "trainability": [],
        "synflow": [],
        "progressivity": []
    }
def auto_train_model(model, train_dataset, test_dataset, input_shape, num_trials=10, save_path="best_model.pth", checkpoint="bert-base-uncased", batch_size=64, tokenizer=None):
    device = "cuda"
    print(f"Using device: {device}")
    model.to(device)
    model_type = get_model_type(model)
    print(f"Detected model type: {model_type}")
    params_history = []
    global scores_history
    scores_history.clear()
    scores_history = {
        "expressivity": [],
        "complexity": [],
        "trainability": [],
        "synflow": [],
        "progressivity": []
    }

    def objective(trial, tokenizer=tokenizer):
        params = search_space(trial, model_type)

        if model_type == "transformer":
            config = AutoConfig.from_pretrained(checkpoint, **params)
            model = AutoModelForSequenceClassification.from_config(config).to(device)
        elif model_type ==  "cnn_mobilenet":
            model = model = build_mobilenet_from_config(params).to(device)
        else:
            model = initialize_model_from_params(model_type, params).to(device)
        
        score = zero_cost_proxies(model, input_shape, device, train_dataset=train_dataset, test_dataset=test_dataset, tokenizer=tokenizer)
        params_history.append(params)
        return 0

    optuna.logging.set_verbosity(optuna.logging.CRITICAL) 
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=num_trials)

    if model_type =="cnn_mobilenet":
        final_scores = az_nas_ranking_CNN(scores_history, proxy_ordering)
    else:
        final_scores = az_nas_ranking(scores_history)


    best_index = np.argmin(final_scores)

    best_params = params_history[best_index]
    print("Best hyperparameters:", best_params)

    if model_type == "transformer":
        config = AutoConfig.from_pretrained(checkpoint, **best_params)
        best_model = AutoModelForSequenceClassification.from_config(config).to(device)
    elif model_type ==  "cnn_mobilenet":
        best_model = model = build_mobilenet_from_config(best_params).to(device)
    else:
        best_model = initialize_model_from_params(model_type, best_params).to(device)

    best_model, accuracy = train_and_evaluate(best_model, train_dataset, test_dataset, model_type, batch_size=batch_size)

    print("Final Model Accuracy:", accuracy)

    torch.save(best_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return best_model, accuracy
