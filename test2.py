import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim

# 设置环境变量和matplotlib参数（沿用test5的中文配置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['figure.dpi'] = 300  # 图片清晰度

# 读取数据集（参考test5/test6的数据读取方式）
data = pd.read_csv('train.csv')
print('数据基本信息：')
data.info()

# 数据预处理（删除缺失值）
data = data.dropna(subset=['y'])  # 确保y列无缺失值
x = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data['y'].values, dtype=torch.float32).unsqueeze(1)


# 定义模型（使用test6的nn.Module训练方法）
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 线性层

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.0001)  # 优化器

# 存储训练过程数据（参考test5的列表存储方式）
w_list = []
b_list = []
loss_list = []
epochs = 100

# 训练模型（test6的训练循环逻辑）
for epoch in range(epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录参数和损失
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    w_list.append(w)
    b_list.append(b)
    loss_list.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 打印最终参数（参考test5的输出方式）
print(f"训练后得到的w值: {w:.4f}")
print(f"训练后得到的b值: {b:.4f}")

# 绘图（复用test5的绘图逻辑和中文显示）
plt.figure(figsize=(12, 5))

# 绘制w和loss关系（含x轴扩展）
plt.subplot(1, 2, 1)
plt.plot(w_list, loss_list)
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('w 和 loss 之间的关系')
min_w, max_w = min(w_list), max(w_list)
w_margin = (max_w - min_w) * 0.1
plt.xlim(min_w - w_margin, max_w + w_margin)

# 绘制b和loss关系（含x轴扩展）
plt.subplot(1, 2, 2)
plt.plot(b_list, loss_list)
plt.xlabel('b')
plt.ylabel('Loss')
plt.title('b 和 loss 之间的关系')
min_b, max_b = min(b_list), max(b_list)
b_margin = (max_b - min_b) * 0.1
plt.xlim(min_b - b_margin, max_b + b_margin)

plt.tight_layout()
plt.savefig('combined_relationship_plot.png')  # 保存图片