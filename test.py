import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import DigitClassifier, device
# 加载重构误差
reconstruction_errors = {}
with open('reconstruction_errors.txt', 'r') as f:
    for line in f:
        label, error = line.strip().split(': ')
        reconstruction_errors[int(label)] = float(error)

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 模型初始化并加载权重
model = DigitClassifier(input_dim=32*7*7, num_classes=10, latent_dim=128).to(device)
model.load_state_dict(torch.load('model/digit_classifier.pth'))
model.eval()

# 创建保存异常输入图像的目录
os.makedirs('anomalies', exist_ok=True)

# 判断测试样本是否为异常输入
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        logits, encoded_feature, rec_feature = model(images)
        
        errors = torch.mean((encoded_feature - rec_feature) ** 2, dim=1)
        
        for i, error in enumerate(errors):
            predicted_label = torch.argmax(logits[i]).item()
            if error.item() > 2 * reconstruction_errors[predicted_label]:  # 使用两倍的重构误差作为阈值
                print(f"Sample {batch_idx * 64 + i} is an outlier with predicted label {predicted_label} and reconstruction error {error.item()}")
                
                # 保存异常输入图像
                anomaly_image = images[i].cpu().numpy().squeeze()
                plt.imshow(anomaly_image, cmap='gray')
                plt.title(f"Pred: {predicted_label}, Error: {error.item():.4f}")
                plt.axis('off')
                plt.savefig(f'anomalies/anomaly_{batch_idx * 64 + i}.png')
                plt.close()
