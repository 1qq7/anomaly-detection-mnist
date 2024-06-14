import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 显示一些样本图片
def show_samples(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
    for i in range(num_samples):
        image, label = dataset[i]
        image = image.numpy().squeeze()  # 将图像转为numpy数组并去掉颜色通道维度
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

show_samples(train_dataset)
