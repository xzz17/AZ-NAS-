from Advanced_AZ_NAS import auto_train_model
from Advanced_AZ_NAS import build_mobilenet_from_config

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision import models

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
              (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
              (0.2023, 0.1994, 0.2010)),
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

auto_train_model(
    model=models.mobilenet_v2(pretrained=True),
    train_dataset=trainset,
    test_dataset=testset,
    input_shape=(1, 3, 32, 32),
    num_trials=20,
    save_path="best_mobilenet.pth",
    checkpoint=None,
    batch_size=batch_size
)
