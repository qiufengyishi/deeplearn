import pandas as pd
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

data = pd.read_csv('train.csv')

print('数据基本信息：')
data.info()

rows, columns = data.shape

if rows < 100 and columns < 20:
    print('数据全部内容信息：')
    print(data.to_csv(sep='\t', na_rep='nan'))
else:
    print('数据前几行内容信息：')
    print(data.head().to_csv(sep='\t', na_rep='nan'))

data = data.dropna()

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

w_values = []
loss_values = []

num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    w_values.append(model.linear.weight.item())
    loss_values.append(loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"训练后得到的w值: {w:.4f}")
print(f"训练后得到的b值: {b:.4f}")


plt.rcParams['figure.dpi'] = 180

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(range(1, num_epochs+1), w_values)
plt.xlabel('Epoch')
plt.ylabel('w 值')
plt.title('w 的变化')

plt.subplot(2, 1, 2)
plt.plot(range(1, num_epochs+1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss 值')
plt.title('Loss 的变化')

plt.tight_layout()
plt.show()