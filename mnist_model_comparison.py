import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import time
import os

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的目录
result_dir = "./model_comparison_results"
os.makedirs(result_dir, exist_ok=True)

# 数据预处理
print("正在加载MNIST数据集...")

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print("数据加载完成！")
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"批次大小: {batch_size}")

# 实现GoogleNet模型（针对MNIST的调整版本）
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积 -> 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1卷积 -> 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3最大池化 -> 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        
        # 初始卷积层，针对MNIST调整
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 简化的Inception模块序列
        self.inception1 = Inception(16, 4, 4, 8, 2, 4, 4)
        self.inception2 = Inception(20, 8, 4, 12, 2, 6, 4)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.inception3 = Inception(30, 12, 6, 16, 3, 8, 6)
        self.inception4 = Inception(42, 16, 8, 20, 4, 10, 8)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类层
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(54, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 实现ResNet模型（针对MNIST的调整版本）
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 下采样层（用于匹配通道数和尺寸）
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        
        # 初始卷积层，针对MNIST调整
        self.in_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 残差层
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类层
        self.fc = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # 如果步长不为1或者输入通道数不等于输出通道数，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 创建ResNet18模型的函数
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2], num_classes)

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 遍历训练数据，不显示详细进度条
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'训练集 - 损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# 定义测试函数
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print(f'测试集 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%')
    
    return avg_loss, accuracy

# 定义计算模型参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定义计算模型FLOPs的函数
def count_flops(model, input_size):
    from thop import profile
    input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input,), verbose=False)
    return flops

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    num_epochs = 5
    learning_rate = 0.001
    
    print(f"\n超参数设置:")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"批次大小: {batch_size}")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 模型字典，用于统一管理
    models = {
        'GoogleNet': GoogleNet(),
        'ResNet18': resnet18()
    }
    
    # 训练结果字典
    results = {}
    
    # 训练和测试每个模型
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"正在训练 {model_name} 模型...")
        
        # 重置模型并移动到设备
        model = model.to(device)
        
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        start_time = time.time()
        train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
        training_time = time.time() - start_time
        
        # 测试模型
        test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
        
        # 计算参数量和FLOPs
        params = count_parameters(model)
        flops = count_flops(model, (1, 28, 28))
        
        # 保存结果
        results[model_name] = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'params': params,
            'flops': flops
        }
        
        print(f"\n{model_name} 模型统计:")
        print(f"参数量: {params:,}")
        print(f"FLOPs: {flops/1e6:.2f} M")
        print(f"训练时间: {training_time:.2f} 秒")
    
    # 可视化比较结果
    visualize_results(results, num_epochs)

# 定义可视化函数
def visualize_results(results, num_epochs):
    print(f"\n{'='*60}")
    print("正在生成可视化结果...")
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 1. 训练/测试准确率曲线
    plt.figure(figsize=(12, 6), dpi=300)
    
    for model_name, result in results.items():
        plt.plot(range(1, num_epochs+1), result['train_accuracies'], marker='o', label=f'{model_name} 训练准确率')
        # 测试准确率作为水平线
        plt.axhline(y=result['test_accuracy'], linestyle='--', label=f'{model_name} 测试准确率')
    
    plt.title('GoogleNet vs ResNet18 准确率曲线', fontsize=14)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '准确率曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 训练/测试损失曲线
    plt.figure(figsize=(12, 6), dpi=300)
    
    for model_name, result in results.items():
        plt.plot(range(1, num_epochs+1), result['train_losses'], marker='o', label=f'{model_name} 训练损失')
        # 测试损失作为水平线
        plt.axhline(y=result['test_loss'], linestyle='--', label=f'{model_name} 测试损失')
    
    plt.title('GoogleNet vs ResNet18 损失曲线', fontsize=14)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '损失曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 模型参数量对比柱状图
    plt.figure(figsize=(10, 6), dpi=300)
    
    model_names = list(results.keys())
    params = [result['params']/1e3 for result in results.values()]  # 转换为千个参数
    
    plt.bar(model_names, params, color=['blue', 'green'])
    
    # 添加数值标签
    for i, v in enumerate(params):
        plt.text(i, v+0.1, f'{v:.1f}K', ha='center', fontweight='bold')
    
    plt.title('GoogleNet vs ResNet18 参数量对比', fontsize=14)
    plt.xlabel('模型名称', fontsize=12)
    plt.ylabel('参数量 (千个)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '参数量对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 计算复杂度对比柱状图
    plt.figure(figsize=(10, 6), dpi=300)
    
    flops = [result['flops']/1e6 for result in results.values()]  # 转换为百万FLOPs
    
    plt.bar(model_names, flops, color=['blue', 'green'])
    
    # 添加数值标签
    for i, v in enumerate(flops):
        plt.text(i, v+0.5, f'{v:.1f}M', ha='center', fontweight='bold')
    
    plt.title('GoogleNet vs ResNet18 计算复杂度对比', fontsize=14)
    plt.xlabel('模型名称', fontsize=12)
    plt.ylabel('FLOPs (百万)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, '计算复杂度对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 测试准确率和训练时间对比
    plt.figure(figsize=(12, 6), dpi=300)
    
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    
    # 测试准确率
    ax1.bar([name + '_acc' for name in model_names], [result['test_accuracy'] for result in results.values()], color=['blue', 'green'], alpha=0.7, label='测试准确率')
    ax1.set_xlabel('模型名称', fontsize=12)
    ax1.set_ylabel('测试准确率 (%)', fontsize=12)
    ax1.tick_params(axis='y')
    
    # 训练时间
    ax2 = ax1.twinx()
    ax2.plot([name + '_time' for name in model_names], [result['training_time'] for result in results.values()], marker='o', color='red', label='训练时间')
    ax2.set_ylabel('训练时间 (秒)', fontsize=12)
    ax2.tick_params(axis='y')
    
    plt.title('GoogleNet vs ResNet18 测试准确率和训练时间对比', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, '准确率和训练时间对比.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存至: {result_dir}")
    print(f"生成的图片包括:")
    print(f"1. 准确率曲线.png")
    print(f"2. 损失曲线.png")
    print(f"3. 参数量对比.png")
    print(f"4. 计算复杂度对比.png")
    print(f"5. 准确率和训练时间对比.png")
    
    # 打印最终比较结果
    print(f"\n{'='*60}")
    print("最终模型性能比较结果:")
    print(f"{'='*60}")
    print(f"{'模型名称':<15} {'测试准确率':<15} {'参数量':<15} {'FLOPs (M)':<15} {'训练时间 (s)':<15}")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['test_accuracy']:<15.2f} {result['params']:<15,} {result['flops']/1e6:<15.2f} {result['training_time']:<15.2f}")

# 运行主函数
if __name__ == "__main__":
    main()
