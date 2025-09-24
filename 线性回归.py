import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 设置环境变量和绘图参数
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 文件路径处理
file_path = 'train.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误: 文件 '{file_path}' 不存在!")
    print(f"当前工作目录: {os.getcwd()}")
    # 尝试查找当前目录下的CSV文件
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        print("当前目录下的CSV文件有:")
        for f in csv_files:
            print(f"- {f}")
    exit(1)

# 读取数据并进行基本检查
data = pd.read_csv(file_path)
print('数据集形状:', data.shape)
print('数据中是否存在无穷大值：', data.isin([float('inf'), float('-inf')]).any().any())
print('数据中是否存在缺失值：', data.isnull().any().any())

# 处理缺失值
if data.isnull().any().any():
    print(f"缺失值处理前: {data.shape}")
    data = data.dropna()
    print(f"缺失值处理后: {data.shape}")

# 数据可视化 - 原始数据分布
plt.figure(figsize=(10, 6))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.5, s=30)
plt.xlabel('特征值')
plt.ylabel('目标值')
plt.title('原始数据分布')
plt.tight_layout()
plt.savefig('original_data.png')
plt.close()

# 准备数据并分割训练集和验证集
x = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).unsqueeze(1)

# 分割训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 初始化模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

# 学习率调度器 - 动态调整学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 记录训练过程中的参数和损失
w_list = []
b_list = []
train_loss_list = []
val_loss_list = []

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    # 计算平均训练损失
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)

    # 验证模式
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_loss_list.append(val_loss)

    # 调整学习率
    scheduler.step(val_loss)

    # 记录参数
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    w_list.append(w)
    b_list.append(b)

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 学习率: {current_lr:.8f}')

# 保存模型
torch.save(model.state_dict(), 'linear_regression_model.pth')
print("模型已保存为 'linear_regression_model.pth'")

# 绘制参数与损失关系图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(w_list, train_loss_list, label='训练损失')
plt.plot(w_list, val_loss_list, label='验证损失')
plt.xlabel('权重 (w)')
plt.ylabel('损失值')
plt.title('权重与损失的关系')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(b_list, train_loss_list, label='训练损失')
plt.plot(b_list, val_loss_list, label='验证损失')
plt.xlabel('偏置 (b)')
plt.ylabel('损失值')
plt.title('偏置与损失的关系')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('parameter_loss_relationship.png')
plt.close()

# 绘制训练过程中的损失变化
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_list, label='训练损失')
plt.plot(range(1, num_epochs + 1), val_loss_list, label='验证损失')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('训练过程中的损失变化')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png')
plt.close()

# 绘制回归结果
model.eval()
with torch.no_grad():
    y_pred = model(x)

plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y.numpy(), alpha=0.5, s=30, label='原始数据')
plt.plot(x.numpy(), y_pred.numpy(), 'r-', linewidth=2, label=f'回归直线: y = {w:.4f}x + {b:.4f}')
plt.xlabel('特征值')
plt.ylabel('目标值')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('regression_result.png')
plt.close()

print(f"最终模型参数: 权重 = {w:.6f}, 偏置 = {b:.6f}")
print(f"最终训练损失: {train_loss_list[-1]:.6f}")
print(f"最终验证损失: {val_loss_list[-1]:.6f}")
