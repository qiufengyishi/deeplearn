import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from torch import nn, optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

data = pd.read_csv('train.csv')
print('数据基本信息：')
data.info()


data = data.dropna(subset=['y'])
x = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data['y'].values, dtype=torch.float32).unsqueeze(1)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

w_list = []
b_list = []
loss_list = []
epochs = 100

for epoch in range(epochs):
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    w = model.linear.weight.item()
    b = model.linear.bias.item()
    w_list.append(w)
    b_list.append(b)
    loss_list.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print(f"训练后得到的w值: {w:.4f}")
print(f"训练后得到的b值: {b:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w_list, loss_list)
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('w 和 loss 之间的关系')
min_w, max_w = min(w_list), max(w_list)
w_margin = (max_w - min_w) * 0.1
plt.xlim(min_w - w_margin, max_w + w_margin)

plt.subplot(1, 2, 2)
plt.plot(b_list, loss_list)
plt.xlabel('b')
plt.ylabel('Loss')
plt.title('b 和 loss 之间的关系')
min_b, max_b = min(b_list), max(b_list)
b_margin = (max_b - min_b) * 0.1
plt.xlim(min_b - b_margin, max_b + b_margin)

plt.tight_layout()
plt.savefig('combined_relationship_plot.png')