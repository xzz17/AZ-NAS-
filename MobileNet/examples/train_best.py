import json
from aznas.train import train_and_evaluate
from aznas.mobilenet import build_mobilenet_from_config
from aznas.dataloader import get_cifar10_loaders

def main():
    # 读取 Zero-Cost 搜索保存的最优结构
    with open("best_config.json", "r") as f:
        best_config = json.load(f)
    print("✅ Loaded best_config:", best_config)

    # 构建模型
    model = build_mobilenet_from_config(best_config)

    # 加载 CIFAR-10 数据集
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # 训练并评估
    acc = train_and_evaluate(model, train_loader, test_loader, epochs=10)
    print(f"\n🎯 Final Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
