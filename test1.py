import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建模拟数据集
np.random.seed(42)
n_samples = 1000
x = np.random.normal(2, 1, n_samples)
# 真实的函数关系：y = 2.5x + 1.2 + 噪声
y = 2.5 * x + 1.2 + np.random.normal(0, 0.5, n_samples)

# 创建DataFrame并保存为CSV
df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('train.csv', index=False)
print("数据集已创建并保存为 train.csv")
print(f"数据形状: {df.shape}")

# 2. 加载数据
data = pd.read_csv('train.csv')
x_data = data['x'].values.reshape(-1, 1)
y_data = data['y'].values.reshape(-1, 1)

# 转换为Tensor
x_tensor = torch.FloatTensor(x_data)
y_tensor = torch.FloatTensor(y_data)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)

print(f"训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")


# 3. 定义线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 使用正态分布初始化权重和偏置
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.normal_(self.linear.bias, mean=0, std=1)

    def forward(self, x):
        return self.linear(x)


# 4. 定义训练函数
def train_model(optimizer_class, optimizer_name, lr=0.01, epochs=1000):
    model = LinearModel()
    criterion = nn.MSELoss()

    # 选择优化器
    if optimizer_class.__name__ == 'SGD':
        optimizer = optimizer_class(model.parameters(), lr=lr)
    elif optimizer_class.__name__ == 'LBFGS':
        optimizer = optimizer_class(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    # 记录训练过程
    losses = []
    weights = []
    biases = []

    for epoch in range(epochs):
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            if loss.requires_grad:
                loss.backward()
            return loss

        if optimizer_class.__name__ == 'LBFGS':
            loss = optimizer.step(closure)
        else:
            # 前向传播
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 记录数据
        losses.append(loss.item())
        weights.append(model.linear.weight.item())
        biases.append(model.linear.bias.item())

    # 测试模型
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_loss = criterion(y_pred_test, y_test)

    return {
        'model': model,
        'losses': losses,
        'weights': weights,
        'biases': biases,
        'final_weight': model.linear.weight.item(),
        'final_bias': model.linear.bias.item(),
        'test_loss': test_loss.item(),
        'name': optimizer_name
    }


# 5. 使用不同的优化器进行训练
optimizers = [
    (torch.optim.SGD, 'SGD'),
    (torch.optim.Adam, 'Adam'),
    (torch.optim.RMSprop, 'RMSprop')
]

results = {}
print("开始训练不同的优化器...")

for optimizer_class, name in optimizers:
    print(f"训练 {name} 优化器...")
    results[name] = train_model(optimizer_class, name, lr=0.01, epochs=1000)

# 6. 可视化结果
# 6.1 损失函数下降曲线
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for name, result in results.items():
    plt.plot(result['losses'][:200], label=name, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('损失函数下降曲线 (前200轮)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
for name, result in results.items():
    plt.plot(result['losses'], label=name, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('损失函数下降曲线 (全部轮次)')
plt.legend()
plt.grid(True, alpha=0.3)

# 6.2 参数变化过程
plt.subplot(2, 2, 3)
for name, result in results.items():
    plt.plot(result['weights'][:200], label=f'{name} - w', alpha=0.8)
plt.axhline(y=2.5, color='red', linestyle='--', label='真实 w=2.5', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('权重 w 变化过程 (前200轮)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
for name, result in results.items():
    plt.plot(result['biases'][:200], label=f'{name} - b', alpha=0.8)
plt.axhline(y=1.2, color='red', linestyle='--', label='真实 b=1.2', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Bias Value')
plt.title('偏置 b 变化过程 (前200轮)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 参数空间可视化 (w和b的调节过程)
fig = plt.figure(figsize=(12, 6))

# 创建网格用于绘制损失曲面
w_range = np.linspace(1.5, 3.5, 50)
b_range = np.linspace(0, 2.5, 50)
W, B = np.meshgrid(w_range, b_range)

# 计算损失曲面
Z = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        w_val = w_range[i]
        b_val = b_range[j]
        y_pred = w_val * x_train.numpy() + b_val
        Z[j, i] = np.mean((y_pred - y_train.numpy()) ** 2)

# 绘制2D等高线图
ax2 = fig.add_subplot(121)
contour = ax2.contour(W, B, Z, levels=20, alpha=0.6)
plt.clabel(contour, inline=True, fontsize=8)
for name, result in results.items():
    ax2.plot(result['weights'][:100], result['biases'][:100], 'o-', label=name, markersize=3, linewidth=1)
ax2.plot(2.5, 1.2, 'r*', markersize=15, label='真实参数')
ax2.set_xlabel('Weight w')
ax2.set_ylabel('Bias b')
ax2.set_title('参数空间优化轨迹 (等高线)')
ax2.legend()

# 最终模型比较
ax3 = fig.add_subplot(122)
x_plot = np.linspace(x_train.min(), x_train.max(), 100)
plt.scatter(x_train.numpy(), y_train.numpy(), alpha=0.3, label='训练数据', s=10)

for name, result in results.items():
    w_final = result['final_weight']
    b_final = result['final_bias']
    y_plot = w_final * x_plot + b_final
    plt.plot(x_plot, y_plot, label=f'{name}: y={w_final:.3f}x+{b_final:.3f}', linewidth=2)

# 真实关系线
y_true = 2.5 * x_plot + 1.2
plt.plot(x_plot, y_true, 'k--', label='真实: y=2.5x+1.2', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('不同优化器的最终拟合结果')
plt.legend()

plt.tight_layout()
plt.savefig('parameter_space_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 学习率和epoch调节实验
def experiment_learning_rates():
    lrs = [0.001, 0.01, 0.1, 0.5]
    epochs_list = [500, 1000, 2000]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, lr in enumerate(lrs):
        if idx >= len(axes):
            break

        for epochs in epochs_list:
            result = train_model(torch.optim.Adam, f'Adam_lr{lr}_ep{epochs}', lr=lr, epochs=epochs)
            axes[idx].plot(result['losses'], label=f'epochs={epochs}')

        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(f'学习率 η = {lr}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')  # 使用对数坐标更好地观察损失变化

    plt.tight_layout()
    plt.savefig('learning_rate_experiment.png', dpi=300, bbox_inches='tight')
    plt.close()


print("\n进行学习率和epoch调节实验...")
experiment_learning_rates()

# 9. 最终结果汇总
print("\n" + "=" * 50)
print("最终模型性能比较:")
print("=" * 50)
for name, result in results.items():
    print(f"{name:8} | w = {result['final_weight']:.4f} | b = {result['final_bias']:.4f} | "
          f"测试损失 = {result['test_loss']:.6f}")

print(f"\n真实参数: w = 2.5000, b = 1.2000")

# 10. 不同优化器的收敛速度比较
plt.figure(figsize=(12, 8))

# 绘制收敛到特定阈值的时间
thresholds = [1.0, 0.5, 0.2, 0.1]
convergence_data = []

for name, result in results.items():
    convergence_epochs = []
    for threshold in thresholds:
        for epoch, loss in enumerate(result['losses']):
            if loss <= threshold:
                convergence_epochs.append(epoch)
                break
        else:
            convergence_epochs.append(None)
    convergence_data.append((name, convergence_epochs))

# 绘制柱状图
x_pos = np.arange(len(thresholds))
width = 0.25

for i, (name, epochs_list) in enumerate(convergence_data):
    valid_epochs = [e for e in epochs_list if e is not None]
    valid_thresholds = [thresholds[j] for j, e in enumerate(epochs_list) if e is not None]
    plt.bar(x_pos[:len(valid_epochs)] + i * width, valid_epochs, width, label=name, alpha=0.8)

plt.xlabel('损失阈值')
plt.ylabel('收敛所需轮数')
plt.title('不同优化器的收敛速度比较')
plt.xticks(x_pos + width, [f'loss ≤ {t}' for t in thresholds])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
plt.close()