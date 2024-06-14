import torch
from torchvision import transforms
from PIL import Image
import os
from model import DigitClassifier, device

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 模型初始化并加载权重
model = DigitClassifier(input_dim=32*7*7, num_classes=10, latent_dim=128).to(device)
model.load_state_dict(torch.load('model/digit_classifier.pth'))
model.eval()

# 加载重构误差
reconstruction_errors = {}
with open('reconstruction_errors.txt', 'r') as f:
    for line in f:
        label, error = line.strip().split(': ')
        reconstruction_errors[int(label)] = float(error)

# 判断字母图像是否为异常输入
letter_dir = 'letters'
with torch.no_grad():
    for filename in os.listdir(letter_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(letter_dir, filename)
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0).to(device)  # 预处理图像并添加batch维度
            
            logits, encoded_feature, rec_feature = model(image)
            error = torch.mean((encoded_feature - rec_feature) ** 2, dim=1).item()
            predicted_label = torch.argmax(logits).item()
            score = torch.softmax(logits, dim=1)[0,predicted_label].item()
            print(filename, predicted_label, score)
            print(error, reconstruction_errors[predicted_label])
            if error > 2 * reconstruction_errors[predicted_label]:
                print(f"Image {filename} is an outlier with predicted label {predicted_label} and reconstruction error {error}")
            print('-'*30)