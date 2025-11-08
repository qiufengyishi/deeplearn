import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use('Agg')  # 非交互式环境使用
import matplotlib.pyplot as plt
from PIL import Image

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 创建保存图像的目录
os.makedirs('../../zuoye2/plt', exist_ok=True)


class MNISTLoader:
    """MNIST数据集加载器"""

    @staticmethod
    def load_images(filename):
        """加载MNIST图像数据"""
        try:
            with open(filename, 'rb') as f:
                # 解析文件头：魔数、图像数量、行数、列数
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                # 读取图像数据并重塑
                images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
                return images
        except FileNotFoundError:
            print(f"错误：找不到文件 {filename}")
            return None
        except Exception as e:
            print(f"加载图像文件时出错: {e}")
            return None

    @staticmethod
    def load_labels(filename):
        """加载MNIST标签数据"""
        try:
            with open(filename, 'rb') as f:
                # 解析文件头：魔数、标签数量
                magic, num = struct.unpack('>II', f.read(8))
                # 读取标签数据
                labels = np.fromfile(f, dtype=np.uint8)
                return labels
        except FileNotFoundError:
            print(f"错误：找不到文件 {filename}")
            return None
        except Exception as e:
            print(f"加载标签文件时出错: {e}")
            return None


class DataProcessor:
    """数据预处理工具类"""

    @staticmethod
    def preprocess(images, labels):
        """
        数据预处理：归一化、维度调整并转换为Tensor

        参数:
            images: 原始图像数据
            labels: 原始标签数据

        返回:
            处理后的图像Tensor和标签Tensor
        """
        # 归一化到[0, 1]区间
        images = images.astype(np.float32) / 255.0
        # 调整维度为(batch, channel, height, width)
        images = images.reshape(-1, 1, 28, 28)
        # 转换为Tensor
        images_tensor = torch.from_numpy(images)
        labels_tensor = torch.from_numpy(labels).long()
        return images_tensor, labels_tensor


class FCNet(nn.Module):
    """全连接神经网络模型"""

    def __init__(self):
        super(FCNet, self).__init__()
        self.flatten = nn.Flatten()  # 展平操作
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)  # 使用nn.Flatten更清晰
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class CNNNet(nn.Module):
    """卷积神经网络模型"""

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)  # 使用nn.Flatten更清晰
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModelTrainer:
    """模型训练与测试工具类"""

    @staticmethod
    def train(model, train_loader, criterion, optimizer, device):
        """训练模型并返回损失和准确率"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        return avg_loss, accuracy

    @staticmethod
    def test(model, test_loader, device):
        """测试模型并返回准确率"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy


def run_experiment(model_name, model, train_loader, test_loader, epochs=5):
    """运行实验并记录结果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"\n开始训练{model_name}模型...")

    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = ModelTrainer.train(
            model, train_loader, criterion, optimizer, device
        )
        # 测试
        test_acc = ModelTrainer.test(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f'{model_name} - 轮次 [{epoch + 1}/{epochs}], '
              f'训练损失: {train_loss:.4f}, '
              f'训练准确率: {train_acc:.2f}%, '
              f'测试准确率: {test_acc:.2f}%')

    return train_losses, train_accuracies, test_accuracies


def visualize_predictions(model, test_loader, device, model_name):
    """可视化模型预测结果"""
    model.eval()
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 创建图形
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        img = images[i].cpu().squeeze()  # 移除通道维度
        ax.imshow(img, cmap='gray')
        ax.set_title(f'真实: {labels[i].item()}, 预测: {predicted[i].item()}')
        ax.axis('off')

    plt.suptitle(f'{model_name}预测结果', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plt/{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形避免内存占用


def main():
    """主函数：执行整个流程"""
    print("正在加载MNIST数据集...")
    # 加载数据
    train_images = MNISTLoader.load_images('../../zuoye2/archive/train-images.idx3-ubyte')
    train_labels = MNISTLoader.load_labels('../../zuoye2/archive/train-labels.idx1-ubyte')
    test_images = MNISTLoader.load_images('../../zuoye2/archive/t10k-images.idx3-ubyte')
    test_labels = MNISTLoader.load_labels('../../zuoye2/archive/t10k-labels.idx1-ubyte')

    # 检查数据加载情况
    if any(data is None for data in [train_images, train_labels, test_images, test_labels]):
        print("数据加载失败，请检查文件路径")
        return

    print(f"训练集大小: {train_images.shape}")
    print(f"测试集大小: {test_images.shape}")

    # 预处理数据
    train_images_tensor, train_labels_tensor = DataProcessor.preprocess(train_images, train_labels)
    test_images_tensor, test_labels_tensor = DataProcessor.preprocess(test_images, test_labels)

    # 创建数据加载器
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 训练参数
    epochs = 10

    print("=" * 60)
    print("MNIST手写数字识别模型性能比较")
    print("=" * 60)

    # 训练FC模型
    fc_model = FCNet()
    fc_train_losses, fc_train_accuracies, fc_test_accuracies = run_experiment(
        "全连接神经网络(FC)", fc_model, train_loader, test_loader, epochs
    )

    # 训练CNN模型
    cnn_model = CNNNet()
    cnn_train_losses, cnn_train_accuracies, cnn_test_accuracies = run_experiment(
        "卷积神经网络(CNN)", cnn_model, train_loader, test_loader, epochs
    )

    # 绘制性能对比曲线
    plt.figure(figsize=(15, 5))

    # 训练准确率曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, epochs + 1), fc_train_accuracies, 'b-', label='FC训练准确率', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_train_accuracies, 'r-', label='CNN训练准确率', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('准确率(%)')
    plt.title('训练准确率对比')
    plt.legend()
    plt.grid(True)

    # 测试准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(range(1, epochs + 1), fc_test_accuracies, 'b--', label='FC测试准确率', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_test_accuracies, 'r--', label='CNN测试准确率', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('准确率(%)')
    plt.title('测试准确率对比')
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(range(1, epochs + 1), fc_train_losses, 'b-', label='FC训练损失', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_train_losses, 'r-', label='CNN训练损失', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plt/mnist_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("性能对比图已保存: plt/mnist_performance_comparison.png")

    # 最终性能比较
    print("\n" + "=" * 60)
    print("最终性能比较结果")
    print("=" * 60)
    print(f"全连接神经网络(FC)最终测试准确率: {fc_test_accuracies[-1]:.2f}%")
    print(f"卷积神经网络(CNN)最终测试准确率: {cnn_test_accuracies[-1]:.2f}%")

    # 性能提升分析
    improvement = cnn_test_accuracies[-1] - fc_test_accuracies[-1]
    print(f"CNN相对于FC的性能提升: {improvement:.2f}%")

    # 保存模型
    torch.save(fc_model.state_dict(), 'plt/fc_model.pth')
    torch.save(cnn_model.state_dict(), 'plt/cnn_model.pth')

    # 可视化预测结果
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_predictions(fc_model, test_loader, device, "全连接神经网络")
    visualize_predictions(cnn_model, test_loader, device, "卷积神经网络")

    # 创建总结图表
    plt.figure(figsize=(10, 6))
    x = np.arange(epochs)
    width = 0.35

    plt.bar(x - width / 2, fc_test_accuracies, width, label='FC测试准确率', alpha=0.7)
    plt.bar(x + width / 2, cnn_test_accuracies, width, label='CNN测试准确率', alpha=0.7)

    plt.xlabel('训练轮次')
    plt.ylabel('测试准确率(%)')
    plt.title('FC vs CNN 测试准确率对比 (10轮训练)')
    plt.xticks(x, range(1, epochs + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plt/accuracy_comparison_bar.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()