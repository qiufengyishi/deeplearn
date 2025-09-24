import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import os

# --- 1. 配置区 (Configuration) ---
# 在这里集中设置所有超参数和文件路径，方便修改
DATA_PATH = 'train.csv'
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
OUTPUT_IMAGE_PATH_1 = 'relationship_plot.png'
OUTPUT_IMAGE_PATH_2 = 'fit_and_loss.png'


# --- 2. 辅助函数 (Helper Functions) ---
def setup_environment():
    """设置绘图环境，确保中文正常显示"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(path):
    """加载CSV数据并进行预处理"""
    print(f"正在从 '{path}' 加载数据...")
    data = pd.read_csv(path)

    # 检查并处理异常值
    print(f'数据中是否存在无穷大值: {data.isin([float("inf"), float("-inf")]).any().any()}')
    print(f'数据中是否存在缺失值: {data.isnull().any().any()}')
    data = data.dropna()
    print(f"预处理完成，剩余有效数据 {len(data)} 条。")

    # 转换为PyTorch张量
    x = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).unsqueeze(1)
    return x, y


def build_model():
    """构建线性回归模型、损失函数和优化器"""
    model = nn.Linear(1, 1)  # 直接使用nn.Linear，更简洁
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, x, y, epochs):
    """训练模型并记录训练过程"""
    print("\n开始模型训练...")
    w_list, b_list, loss_list = [], [], []

    for epoch in range(epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录参数和损失
        w = model.weight.item()
        b = model.bias.item()
        w_list.append(w)
        b_list.append(b)
        loss_list.append(loss.item())

        # 打印训练日志
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, W: {w:.4f}, B: {b:.4f}')

    print("模型训练完成！")
    return w_list, b_list, loss_list


def plot_parameter_relationships(w_list, b_list, loss_list, save_path):
    """绘制参数w、b与Loss的关系图"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(w_list, loss_list, marker='.', linestyle='-', label='Loss')
    plt.xlabel('权重 (w)')
    plt.ylabel('损失值 (Loss)')
    plt.title('权重 w 与 Loss 的关系')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(b_list, loss_list, marker='.', linestyle='-', label='Loss', color='orange')
    plt.xlabel('偏置 (b)')
    plt.ylabel('损失值 (Loss)')
    plt.title('偏置 b 与 Loss 的关系')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n参数关系图已保存至: {save_path}")
    # plt.show() # 如果需要在脚本运行时显示图像，可以取消此行注释


def plot_fit_and_loss_curve(x, y, model, loss_list, save_path):
    """绘制数据拟合效果和损失下降曲线"""
    plt.figure(figsize=(12, 5))

    # 图1: 数据散点图与拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(x.numpy(), y.numpy(), s=15, alpha=0.6, label='原始数据')
    with torch.no_grad():
        y_pred = model(x)
    plt.plot(x.numpy(), y_pred.numpy(), 'r-', lw=2, label='拟合直线')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('数据拟合效果')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 图2: 损失函数下降曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, marker='.', linestyle='-', label='Loss')
    plt.xlabel('迭代次数 (Epoch)')
    plt.ylabel('损失值 (Loss)')
    plt.title('训练过程中的损失下降')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"拟合效果与损失曲线已保存至: {save_path}")
    # plt.show() # 如果需要在脚本运行时显示图像，可以取消此行注释


# --- 3. 主执行逻辑 (Main Execution) ---
if __name__ == '__main__':
    # 初始化环境
    setup_environment()

    # 数据加载与预处理
    x_train, y_train = load_and_preprocess_data(DATA_PATH)

    # 模型构建
    model, criterion, optimizer = build_model()

    # 模型训练
    weights, biases, losses = train_model(model, criterion, optimizer, x_train, y_train, NUM_EPOCHS)

    # 可视化
    plot_parameter_relationships(weights, biases, losses, OUTPUT_IMAGE_PATH_1)
    plot_fit_and_loss_curve(x_train, y_train, model, losses, OUTPUT_IMAGE_PATH_2)

    print("\n所有任务完成！")