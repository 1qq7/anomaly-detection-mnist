import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
from model import DigitClassifier, device


# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 模型初始化并加载权重
model = DigitClassifier(input_dim=32*7*7, num_classes=10, latent_dim=128).to(device)
model.load_state_dict(torch.load('model/digit_classifier.pth'))
model.eval()

# 计算每个类别的重构误差
reconstruction_errors = {i: [] for i in range(10)}

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        _, encoded_feature, rec_feature = model(images)
        
        errors = torch.mean((encoded_feature - rec_feature) ** 2, dim=1)
        
        for i, label in enumerate(labels):
            reconstruction_errors[label.item()].append(errors[i].item())

# 计算每个类别的平均重构误差
mean_reconstruction_errors = {k: np.mean(v) for k, v in reconstruction_errors.items()}

# 将结果写入文件
with open('reconstruction_errors.txt', 'w') as f:
    for label, error in mean_reconstruction_errors.items():
        f.write(f"{label}: {error}\n")
