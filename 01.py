import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义自定义数据集类
class CountryDataset(Dataset):
    def __init__(self, data):
        self.data = data.drop(columns=['Country', 'Region', 'GDP per Capita', 'Data Quality'])
        self.data = self.data.dropna()
        self.x = self.data.drop(columns=['Total Ecological Footprint']).values
        self.y = self.data['Total Ecological Footprint'].values

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 定义 5 层神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 7)
        self.fc2 = nn.Linear(7, 6)
        self.fc3 = nn.Linear(6, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 读取数据集
data = pd.read_csv('/mnt/countries.csv')

# 创建数据集和数据加载器
dataset = CountryDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# 初始化模型、损失函数和优化器
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
best_loss = float('inf')
train_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    train_losses.append(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), '/mnt/best_model.pt')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 可视化训练过程
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('/mnt/training_loss.png')
plt.show()