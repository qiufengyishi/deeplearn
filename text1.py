import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time


# ==========================================
# 1. 数据处理模块 (Data Processing Module)
# ==========================================
class DataProcessor:
    """数据集加载与预处理封装类"""

    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.transform = self._get_transform()
        self.train_loader, self.test_loader = self._load_data()

    def _get_transform(self):
        """获取数据预处理管道"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
        ])

    def _load_data(self):
        """加载MNIST数据集并创建数据加载器"""
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )

        return (
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        )

    def get_loaders(self):
        """获取训练集和测试集加载器"""
        return self.train_loader, self.test_loader


# ==========================================
# 2. 模型管理模块 (Model Management Module)
# ==========================================
class ModelFactory:
    """模型创建工厂类"""

    @staticmethod
    def create_model(model_type):
        """根据模型类型创建对应模型实例"""
        if model_type == "googlenet":
            return MiniGoogleNet()
        elif model_type == "resnet":
            return MiniResNet()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


# 2.1 GoogleNet 相关模型定义
class InceptionBlock(nn.Module):
    """Inception模块实现"""

    def __init__(self, in_channels, c1, c2, c3, c4):
        super(InceptionBlock, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class MiniGoogleNet(nn.Module):
    """简化版GoogleNet模型"""

    def __init__(self):
        super(MiniGoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.inception1 = InceptionBlock(10, 16, (16, 32), (16, 16), 16)
        self.inception2 = InceptionBlock(80, 32, (32, 64), (16, 32), 32)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(160, 10)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 2.2 ResNet 相关模型定义
class ResidualBlock(nn.Module):
    """残差块实现"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MiniResNet(nn.Module):
    """简化版ResNet模型"""

    def __init__(self):
        super(MiniResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ==========================================
# 3. 训练器模块 (Trainer Module)
# ==========================================
class ModelTrainer:
    """模型训练与评估封装类"""

    def __init__(self, model, train_loader, test_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []
        self.acc_history = []

    def train_epoch(self):
        """训练单个epoch"""
        self.model.train()
        running_loss = 0.0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def evaluate(self):
        """在测试集上评估模型"""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total

    def run_training(self, epochs, model_name):
        """执行完整训练过程"""
        print(f"\n--- 开始训练 {model_name} ---")
        start_time = time.time()

        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            accuracy = self.evaluate()

            self.loss_history.append(avg_loss)
            self.acc_history.append(accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print(f"{model_name} 训练完成, 耗时: {time.time() - start_time:.1f}s")
        return self.loss_history, self.acc_history


# ==========================================
# 4. 主程序模块 (Main Program Module)
# ==========================================
class ExperimentRunner:
    """实验运行主类"""

    def __init__(self):
        # 配置参数
        self.config = {
            "batch_size": 64,
            "lr": 0.001,
            "epochs": 5,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        print(f"使用的设备: {self.config['device']}")

        # 初始化数据处理器
        self.data_processor = DataProcessor(batch_size=self.config["batch_size"])
        self.train_loader, self.test_loader = self.data_processor.get_loaders()

    def run_experiment(self):
        """运行模型对比实验"""
        # 训练GoogleNet
        googlenet = ModelFactory.create_model("googlenet")
        googlenet_trainer = ModelTrainer(
            googlenet,
            self.train_loader,
            self.test_loader,
            self.config["device"],
            self.config["lr"]
        )
        g_loss, g_acc = googlenet_trainer.run_training(
            self.config["epochs"],
            "GoogleNet (Inception)"
        )

        # 训练ResNet
        resnet = ModelFactory.create_model("resnet")
        resnet_trainer = ModelTrainer(
            resnet,
            self.train_loader,
            self.test_loader,
            self.config["device"],
            self.config["lr"]
        )
        r_loss, r_acc = resnet_trainer.run_training(
            self.config["epochs"],
            "ResNet (Residual)"
        )

        # 可视化结果
        self._visualize_results(g_loss, g_acc, r_loss, r_acc)

    def _visualize_results(self, g_loss, g_acc, r_loss, r_acc):
        """可视化训练损失和准确率曲线"""
        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.config["epochs"] + 1), g_loss, label='GoogleNet', marker='o')
        plt.plot(range(1, self.config["epochs"] + 1), r_loss, label='ResNet', marker='s')
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.config["epochs"] + 1), g_acc, label='GoogleNet', marker='o')
        plt.plot(range(1, self.config["epochs"] + 1), r_acc, label='ResNet', marker='s')
        plt.title('Test Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 启动实验
    experiment = ExperimentRunner()
    experiment.run_experiment()