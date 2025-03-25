from Advanced_AZ_NAS import auto_train_model
from torchvision import models
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

batch_size = 64
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

auto_train_model(
    model=models.resnet18(pretrained=False),
    train_dataset=trainset,
    test_dataset=testset,
    input_shape=(1, 3, 32, 32),
    num_trials=10,
    save_path="best_resnet.pth",
    checkpoint=None,
    batch_size=batch_size
)
