import json
from aznas.train import train_and_evaluate
from aznas.mobilenet import build_mobilenet_from_config
from aznas.dataloader import get_cifar10_loaders

def main():
    # Load the best architecture config saved from Zero-Cost NAS search
    with open("best_config.json", "r") as f:
        best_config = json.load(f)
    print("✅ Loaded best_config:", best_config)

    # Build the model
    model = build_mobilenet_from_config(best_config)

    # Load CIFAR-10 dataset
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Train and evaluate
    acc = train_and_evaluate(model, train_loader, test_loader, epochs=10)
    print(f"\n🎯 Final Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
