import torch
import torch.nn as nn
import numpy as np
from .mobilenet import MobileNetBlock

def expressivity_score(model, input_shape=(1,3,32,32), device='cuda'):
    """
    Measures the expressivity of the model based on entropy of the output
    after the first convolutional or MobileNetBlock layer.

    Higher entropy means the model can express more varied outputs.
    """
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        for layer in model:
            if isinstance(layer, (nn.Conv2d, MobileNetBlock)):
                features = layer(dummy_input)
                break
        # Compute softmax entropy
        prob = torch.softmax(features.flatten(), dim=0)
        entropy = -(prob * torch.log(prob + 1e-8)).sum().item()
    return entropy

def progressivity_score(model, input_shape=(1,3,32,32), device='cuda'):
    """
    Measures the minimum increase in entropy across layers (proxy for information flow).

    A more "progressive" network increases feature complexity gradually.
    """
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
    # Return the minimum difference between consecutive entropy values
    diffs = [entropies[i+1] - entropies[i] for i in range(len(entropies)-1)]
    return min(diffs) if diffs else 0

def trainability_score(model, input_shape=(1,3,32,32), device='cuda'):
    """
    Measures how easily the model can be trained by computing the gradient norm
    from a dummy backward pass.

    A higher gradient norm implies better trainability.
    """
    model.train()
    dummy_input = torch.randn(input_shape, requires_grad=True).to(device)
    output = model(dummy_input)
    output.sum().backward()
    grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    grads = torch.cat(grads)
    sigma = grads.norm().item()
    return np.log(sigma + 1e-8)

def complexity_score(model):
    """
    Measures model complexity using parameter count.

    Returns log(number of parameters), a proxy for size and cost.
    """
    total = sum(p.numel() for p in model.parameters())
    return np.log(total + 1)

def synflow_score(model, input_shape=(1,3,32,32), device='cuda'):
    """
    SynFlow: Zero-cost proxy measuring sensitivity of model output to weights.

    It computes the sum of gradient * weight (abs) after backward on all-one input.
    Higher score suggests more trainable and expressive parameters.
    """
    model.to(device).eval()

    # Ensure weights are positive
    for p in model.parameters():
        p.data = p.data.abs()

    # Forward + backward on constant input
    input_tensor = torch.ones(input_shape).to(device)
    model.zero_grad()
    model(input_tensor).sum().backward()

    # Compute SynFlow score: sum of |grad * weight|
    score = sum((p.grad * p.data).abs().sum().item() for p in model.parameters() if p.grad is not None)
    return score
