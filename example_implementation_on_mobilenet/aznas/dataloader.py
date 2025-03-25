import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import optuna
import time
import gc
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set a fixed seed for full reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Make CUDA deterministic (disable benchmark for consistent results)
cudnn.deterministic = True
cudnn.benchmark = False

def get_cifar10_loaders(batch_size=128):
    """
    Returns CIFAR-10 DataLoaders for training and testing.

    Args:
        batch_size (int): Batch size for both train and test loaders.

    Returns:
        train_loader (DataLoader): DataLoader for training set
        test_loader (DataLoader): DataLoader for test set
    """
    # Training data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Test data normalization (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Download CIFAR-10 dataset and create DataLoaders
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
