import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 数据预处理
# ----------------------------
# 定义输入输出序列
input_str = "dlearn"
target_str = "lanrla"

# 收集所有唯一字符并创建映射表
all_chars = sorted(list(set(input_str + target_str)))
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
vocab_size = len(all_chars)

# 序列长度
seq_len = len(input_str)
print(f"字符映射表: {char_to_idx}")
print(f"输入序列: {input_str} -> 编码: {[char_to_idx[c] for c in input_str]}")
print(f"目标序列: {target_str} -> 编码: {[char_to_idx[c] for c in target_str]}")


# 字符串转张量函数
def str_to_tensor(s):
    return torch.tensor([char_to_idx[c] for c in s], dtype=torch.long)


# 转换输入输出为张量
input_tensor = str_to_tensor(input_str)  # 形状: (seq_len,)
target_tensor = str_to_tensor(target_str)  # 形状: (seq_len,)

# ----------------------------
# 训练参数配置
# ----------------------------
hidden_size = 64  # 隐藏层维度
epochs = 20  # 训练轮次
learning_rate = 0.005
print_interval = 1  # 打印间隔


# ----------------------------
# 1. 使用nn.RNNCell实现模型
# ----------------------------
class RNNCellModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNCellModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 字符嵌入层
        self.rnn_cell = nn.RNNCell(hidden_size, hidden_size)  # RNN单元
        self.fc = nn.Linear(hidden_size, vocab_size)  # 输出层

    def forward(self, input_seq):
        batch_size = 1
        hidden = self.init_hidden(batch_size)  # 初始化隐藏状态
        outputs = []

        for char in input_seq:
            # 处理每个时间步的字符
            embed = self.embedding(char.unsqueeze(0))  # (1, hidden_size)
            hidden = self.rnn_cell(embed, hidden)  # 更新隐藏状态
            output = self.fc(hidden)  # 计算输出
            outputs.append(output)

        return torch.cat(outputs, dim=0), hidden  # 拼接所有输出

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


# 初始化模型、损失函数和优化器
rnn_cell_model = RNNCellModel(vocab_size, hidden_size)
criterion_cell = nn.CrossEntropyLoss()
optimizer_cell = optim.Adam(rnn_cell_model.parameters(), lr=learning_rate)

# 记录训练过程数据
cell_train_data = {
    'losses': [],  # 每轮损失
    'predictions': [],  # 预测结果 (epoch, pred_str)
    'hidden_states': []  # 隐藏状态记录
}

# 训练RNNCell模型
print("\n----- 开始训练 RNNCell 模型 -----")
for epoch in range(epochs):
    optimizer_cell.zero_grad()  # 重置梯度

    # 前向传播
    outputs, hidden = rnn_cell_model(input_tensor)
    loss = criterion_cell(outputs, target_tensor)

    # 反向传播与优化
    loss.backward()
    optimizer_cell.step()

    # 记录训练数据
    cell_train_data['losses'].append(loss.item())

    # 定期记录预测结果和隐藏状态
    if (epoch + 1) % print_interval == 0:
        _, predicted_idx = torch.max(outputs, dim=1)
        predicted_str = ''.join([idx_to_char[idx.item()] for idx in predicted_idx])
        cell_train_data['predictions'].append((epoch + 1, predicted_str))
        cell_train_data['hidden_states'].append(hidden.detach().numpy())
        print(f"RNNCell - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, 预测: {predicted_str}")


# ----------------------------
# 2. 使用nn.RNN实现模型
# ----------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 字符嵌入层
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # RNN层
        self.fc = nn.Linear(hidden_size, vocab_size)  # 输出层

    def forward(self, input_seq):
        # 调整输入形状: (seq_len,) -> (1, seq_len)
        input_seq = input_seq.unsqueeze(0)

        # 嵌入层
        embed = self.embedding(input_seq)  # (1, seq_len, hidden_size)

        # 初始化隐藏状态
        hidden = self.init_hidden()

        # RNN前向传播
        outputs, hidden = self.rnn(embed, hidden)  # (1, seq_len, hidden_size)

        # 输出层 (去除batch维度)
        outputs = self.fc(outputs).squeeze(0)  # (seq_len, vocab_size)
        return outputs, hidden

    def init_hidden(self):
        # 隐藏状态形状: (num_layers, batch_size, hidden_size)
        return torch.zeros(1, 1, self.hidden_size)


# 初始化模型、损失函数和优化器
rnn_model = RNNModel(vocab_size, hidden_size)
criterion_rnn = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=learning_rate)

# 记录训练过程数据
rnn_train_data = {
    'losses': [],  # 每轮损失
    'predictions': [],  # 预测结果 (epoch, pred_str)
    'hidden_states': []  # 隐藏状态记录
}

# 训练RNN模型
print("\n----- 开始训练 RNN 模型 -----")
for epoch in range(epochs):
    optimizer_rnn.zero_grad()  # 重置梯度

    # 前向传播
    outputs, hidden = rnn_model(input_tensor)
    loss = criterion_rnn(outputs, target_tensor)

    # 反向传播与优化
    loss.backward()
    optimizer_rnn.step()

    # 记录训练数据
    rnn_train_data['losses'].append(loss.item())

    # 定期记录预测结果和隐藏状态
    if (epoch + 1) % print_interval == 0:
        _, predicted_idx = torch.max(outputs, dim=1)
        predicted_str = ''.join([idx_to_char[idx.item()] for idx in predicted_idx])
        rnn_train_data['predictions'].append((epoch + 1, predicted_str))
        rnn_train_data['hidden_states'].append(hidden.detach().numpy())
        print(f"RNN      - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, 预测: {predicted_str}")

# ----------------------------
# 结果可视化与分析
# ----------------------------
# 1. 绘制损失曲线
plt.figure(figsize=(14, 6))

# RNNCell损失曲线
plt.subplot(1, 2, 1)
plt.plot(cell_train_data['losses'], label='RNNCell Loss')
plt.title('RNNCell 训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# RNN损失曲线
plt.subplot(1, 2, 2)
plt.plot(rnn_train_data['losses'], label='RNN Loss', color='orange')
plt.title('RNN 训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# 2. 打印最终预测结果
print("\n----- 最终预测结果 -----")

# RNNCell最终预测
with torch.no_grad():
    outputs, _ = rnn_cell_model(input_tensor)
    _, pred_idx = torch.max(outputs, dim=1)
    cell_final_pred = ''.join([idx_to_char[idx.item()] for idx in pred_idx])
print(f"RNNCell 最终预测: {cell_final_pred} (目标: {target_str})")

# RNN最终预测
with torch.no_grad():
    outputs, _ = rnn_model(input_tensor)
    _, pred_idx = torch.max(outputs, dim=1)
    rnn_final_pred = ''.join([idx_to_char[idx.item()] for idx in pred_idx])
print(f"RNN 最终预测: {rnn_final_pred} (目标: {target_str})")

# 3. 打印训练过程中的预测变化
print("\n----- 训练过程预测变化 -----")
print("RNNCell 预测变化:")
for epoch, pred in cell_train_data['predictions']:
    print(f"Epoch {epoch}: {pred}")

print("\nRNN 预测变化:")
for epoch, pred in rnn_train_data['predictions']:
    print(f"Epoch {epoch}: {pred}")