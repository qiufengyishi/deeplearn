import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

data = pd.read_csv('train.csv')
print('数据中是否存在无穷大值：', data.isin([float('inf'), float('-inf')]).any().any())
print('数据中是否存在缺失值：', data.isnull().any().any())

data = data.dropna()

x = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).unsqueeze(1)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

w_list = []
b_list = []
loss_list = []

num_epochs = 100
for epoch in range(num_epochs):
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.rcParams['figure.dpi'] = 300


plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(w_list, loss_list)
plt.xlabel('w')
plt.xticks(rotation=45)
plt.ylabel('Loss')
plt.title('w 和 Loss 的关系')

plt.subplot(1, 2, 2)
plt.plot(b_list, loss_list)
plt.xlabel('b')
plt.xticks(rotation=45)
plt.ylabel('Loss')
plt.title('b 和 Loss 的关系')

plt.tight_layout()
plt.savefig('relationship_plot1.png')
