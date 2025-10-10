import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from io import BytesIO, StringIO
from PIL import Image


# 加载数据集
def load_data(file_path):
    # 从字符串读取数据（如果没有本地文件）
    if isinstance(file_path, StringIO):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path)
    x = torch.tensor(data['x'].values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(data['y'].values, dtype=torch.float32).reshape(-1, 1)
    return x, y


# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入特征数为1，输出特征数为1

    def forward(self, x):
        return self.linear(x)


# 训练函数
def train_model(model, optimizer, criterion, train_loader, epochs, device):
    model.train()
    losses = []
    ws = []
    bs = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # 前向传播
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        # 记录参数
        ws.append(model.linear.weight.item())
        bs.append(model.linear.bias.item())
    return losses, ws, bs


# 将matplotlib图像转换为PIL Image（修复了tostring_rgb问题）
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


# 可视化损失
def plot_loss(losses_list, labels):
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(losses_list):
        plt.plot(losses, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Optimizers')
    plt.legend()
    img = fig_to_image(plt.gcf())
    plt.close()  # 关闭图像以释放资源
    return img


# 可视化参数变化
def plot_params(ws_list, bs_list, labels):
    fig, axes = plt.subplots(2, len(ws_list), figsize=(15, 8))
    for i, (ws, bs) in enumerate(zip(ws_list, bs_list)):
        axes[0, i].plot(ws, label='Weight')
        axes[0, i].set_title(labels[i] + ' - Weight')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Value')
        axes[0, i].legend()

        axes[1, i].plot(bs, label='Bias')
        axes[1, i].set_title(labels[i] + ' - Bias')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Value')
        axes[1, i].legend()
    plt.tight_layout()
    img = fig_to_image(plt.gcf())
    plt.close()
    return img


# 可视化不同学习率的损失
def plot_lr_loss(losses_dict, lr_values):
    plt.figure(figsize=(10, 6))
    for lr, losses in losses_dict.items():
        plt.plot(losses, label=f'LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Learning Rates')
    plt.legend()
    img = fig_to_image(plt.gcf())
    plt.close()
    return img


# 可视化不同epoch的损失
def plot_epoch_loss(losses_dict, epoch_values):
    plt.figure(figsize=(10, 6))
    for epoch, losses in losses_dict.items():
        plt.plot(losses, label=f'Epochs={epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Epoch Numbers')
    plt.legend()
    img = fig_to_image(plt.gcf())
    plt.close()
    return img


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建示例数据（如果没有train.csv文件）
    data_str = "x,y\n"
    for i in range(100):
        x = i / 10.0
        y = 2.5 * x + 3.0 + np.random.normal(0, 0.5)
        data_str += f"{x},{y}\n"
    file_path = StringIO(data_str)

    # 加载数据
    x, y = load_data(file_path)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型、损失函数
    model = LinearRegression().to(device)
    # 正态分布初始化参数
    nn.init.normal_(model.linear.weight, mean=0, std=1)
    nn.init.normal_(model.linear.bias, mean=0, std=1)
    criterion = nn.MSELoss()

    # 选择三种优化器
    optimizers = {
        'SGD': torch.optim.SGD(model.parameters(), lr=0.01),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.01),
        'Adagrad': torch.optim.Adagrad(model.parameters(), lr=0.01)
    }

    epochs = 100
    losses_list = []
    ws_list = []
    bs_list = []
    labels = []

    for name, optimizer in optimizers.items():
        # 重新初始化模型参数
        nn.init.normal_(model.linear.weight, mean=0, std=1)
        nn.init.normal_(model.linear.bias, mean=0, std=1)
        losses, ws, bs = train_model(model, optimizer, criterion, train_loader, epochs, device)
        losses_list.append(losses)
        ws_list.append(ws)
        bs_list.append(bs)
        labels.append(name)
        print(f"{name}优化器训练完成，最终损失: {losses[-1]:.6f}")

    # 可视化不同优化器的损失
    loss_img = plot_loss(losses_list, labels)
    loss_img.save('optimizer_losses.png')

    # 可视化不同优化器的参数变化
    params_img = plot_params(ws_list, bs_list, labels)
    params_img.save('parameter_changes.png')

    # 调节学习率
    lrs = [0.001, 0.01, 0.1]
    lr_losses = {}
    for lr in lrs:
        model = LinearRegression().to(device)
        nn.init.normal_(model.linear.weight, mean=0, std=1)
        nn.init.normal_(model.linear.bias, mean=0, std=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses, _, _ = train_model(model, optimizer, criterion, train_loader, epochs, device)
        lr_losses[lr] = losses
        print(f"学习率 {lr} 训练完成，最终损失: {losses[-1]:.6f}")

    lr_img = plot_lr_loss(lr_losses, lrs)
    lr_img.save('learning_rate_losses.png')

    # 调节epoch
    epochs_list = [50, 100, 200]
    epoch_losses = {}
    for epoch in epochs_list:
        model = LinearRegression().to(device)
        nn.init.normal_(model.linear.weight, mean=0, std=1)
        nn.init.normal_(model.linear.bias, mean=0, std=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses, _, _ = train_model(model, optimizer, criterion, train_loader, epoch, device)
        epoch_losses[epoch] = losses
        print(f"迭代次数 {epoch} 训练完成，最终损失: {losses[-1]:.6f}")

    epoch_img = plot_epoch_loss(epoch_losses, epochs_list)
    epoch_img.save('epoch_losses.png')

    # 选择性能最好的模型（基于上面的结果选择Adam）
    best_model = LinearRegression().to(device)
    nn.init.normal_(best_model.linear.weight, mean=0, std=1)
    nn.init.normal_(best_model.linear.bias, mean=0, std=1)
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=0.01)
    train_model(best_model, best_optimizer, criterion, train_loader, epochs, device)

    # 保存模型
    torch.save(best_model.state_dict(), 'best_linear_model.pth')
    print("训练性能最好的模型已保存为 best_linear_model.pth")
    print("所有可视化图像已保存为PNG文件")


if __name__ == '__main__':
    main()

