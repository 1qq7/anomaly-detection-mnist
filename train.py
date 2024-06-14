import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import DigitClassifier, compute_loss, device


# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型初始化
model = DigitClassifier(input_dim=32*7*7, num_classes=10, latent_dim=128).to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_classification_loss = 0.0
    total_reconstruction_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits, encoded_feature, rec_feature = model(images)
        classification_loss, reconstruction_loss = compute_loss(logits, labels, encoded_feature, rec_feature)
        loss = classification_loss + reconstruction_loss
        
        loss.backward()
        optimizer.step()
        
        total_classification_loss += classification_loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Classification Loss: {total_classification_loss:.4f}, Reconstruction Loss: {total_reconstruction_loss:.4f}")

# 保存模型权重
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), 'model/digit_classifier.pth')
