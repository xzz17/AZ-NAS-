import torch
import torch.nn as nn
import numpy as np
from .mobilenet import MobileNetBlock

def expressivity_score(model, input_shape=(1,3,32,32), device='cuda'):
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

def progressivity_score(model, input_shape=(1,3,32,32), device='cuda'):
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

def trainability_score(model, input_shape=(1,3,32,32), device='cuda'):
    model.train()
    dummy_input = torch.randn(input_shape, requires_grad=True).to(device)
    output = model(dummy_input)
    output.sum().backward()
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    grads = torch.cat(grads)
    sigma = grads.norm().item()
    return np.log(sigma + 1e-8)

def complexity_score(model):
    total = sum(p.numel() for p in model.parameters())
    return np.log(total + 1)

def synflow_score(model, input_shape=(1,3,32,32), device='cuda'):
    model.to(device).eval()
    for p in model.parameters():
        p.data = p.data.abs()
    input_tensor = torch.ones(input_shape).to(device)
    model.zero_grad()
    model(input_tensor).sum().backward()
    score = sum((p.grad * p.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
    return score
