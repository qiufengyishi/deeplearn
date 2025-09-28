import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 1. 数据加载与预处理
data = pd.read_csv('train.csv').dropna()
x = torch.tensor(data['x'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)

# 创建数据集和数据加载器
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 模型定义（简洁方式）
model = nn.Linear(1, 1)

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# 4. 训练过程与记录
w_history = []
loss_history = []
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

    # 计算平均损失
    avg_loss = epoch_loss / len(dataset)
    loss_history.append(avg_loss)
    w_history.append(model.weight.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, W: {model.weight.item():.4f}')

# 5. 可视化结果
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['figure.dpi'] = 150

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), w_history, 'b-')
plt.title('权重w的变化')
plt.xlabel('迭代次数')
plt.ylabel('w值')

plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), loss_history, 'r-')
plt.title('损失值的变化')
plt.xlabel('迭代次数')
plt.ylabel('损失值')

plt.tight_layout()
plt.show()
