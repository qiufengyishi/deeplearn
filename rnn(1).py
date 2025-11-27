import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import os
import random
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置类 ====================
class Config:
    def __init__(self):
        self.seed = 1234
        self.batch_size = 64
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.output_dim = 5  # 0-4共5类
        self.n_layers = 2
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.max_len = 60
        self.min_freq = 2
        self.train_path = "train.tsv"
        self.test_path = "test.tsv"
        self.output_path = "RNNsubmission.csv"
        self.use_cpu = False  # 可强制使用CPU排查GPU问题
        self.debug_mode = True  # 调试模式，开启同步报错


# ==================== 工具类 ====================
class Utils:
    @staticmethod
    def set_seed(seed):
        """设置全局随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and not Config().use_cpu:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_debug_env(config):
        """设置调试环境，解决CUDA异步报错问题"""
        if config.debug_mode and torch.cuda.is_available():
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
        print("调试环境已设置，CUDA同步报错开启")

    @staticmethod
    def clean_text(text):
        """文本清洗：转小写并保留字母数字"""
        return re.sub(r'[^a-z0-9\s]', '', str(text).lower())

    @staticmethod
    def validate_labels(series):
        """验证标签是否在0-4范围内，并过滤异常值"""
        valid_mask = series.between(0, 4)
        invalid_count = len(series) - valid_mask.sum()
        if invalid_count > 0:
            print(f"警告：发现{invalid_count}个异常标签，已过滤")
            return series[valid_mask]
        return series


# ==================== 数据处理模块 ====================
class VocabBuilder:
    """词表构建与管理类"""

    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.stoi = None
        self.itos = None
        self.vocab_size = None

    def build_from_tokens(self, tokens_list):
        counter = Counter(token for tokens in tokens_list for token in tokens)
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        self.stoi.update({word: idx + 2 for idx, (word, freq) in enumerate(counter.items())
                          if freq >= self.min_freq})
        self.itos = {v: k for k, v in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        # 验证词表大小
        assert self.vocab_size > 2, "词表过小，请检查min_freq设置"
        print(f"词表构建完成，大小: {self.vocab_size}")
        return self

    def text_to_indices(self, text, max_len):
        """文本转索引序列（截断/填充）"""
        tokens = text.split()
        indices = [self.stoi.get(token, self.stoi[self.unk_token]) for token in tokens[:max_len]]
        # 确保索引不越界
        indices = [idx if idx < self.vocab_size else self.stoi[self.unk_token] for idx in indices]
        padded = indices + [self.stoi[self.pad_token]] * (max_len - len(indices))
        assert len(padded) == max_len, "序列长度不匹配"
        assert max(padded) < self.vocab_size, "索引超出词表范围"
        return padded


class ReviewDataset(Dataset):
    """影评数据集类"""

    def __init__(self, df, vocab, max_len, is_test=False):
        self.texts = df['Phrase'].apply(Utils.clean_text).values
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.labels = df['Sentiment'].values if not is_test else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.vocab.text_to_indices(self.texts[idx], self.max_len)
        x = torch.tensor(indices, dtype=torch.long)

        if self.is_test:
            return x
        else:
            # 确保标签是long类型且在有效范围
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            assert label.item() in [0, 1, 2, 3, 4], f"无效标签值: {label.item()}"
            return x, label


class DataManager:
    """数据加载与管理类"""

    def __init__(self, config):
        self.config = config
        self.vocab = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_raw_data(self):
        """加载原始TSV数据并验证"""
        if not all(os.path.exists(f) for f in [self.config.train_path, self.config.test_path]):
            raise FileNotFoundError("缺少数据文件（train.tsv/test.tsv）")

        train_df_full = pd.read_csv(self.config.train_path, sep='\t')
        test_df = pd.read_csv(self.config.test_path, sep='\t')

        # 验证并清理标签
        train_df_full['Sentiment'] = Utils.validate_labels(train_df_full['Sentiment'])
        train_df_full = train_df_full.dropna(subset=['Sentiment'])

        self.test_df = test_df
        self.train_df, self.val_df = train_test_split(
            train_df_full, test_size=0.2, random_state=self.config.seed
        )

        print(f"数据加载完成 - 训练集: {len(self.train_df)} | 验证集: {len(self.val_df)} | 测试集: {len(self.test_df)}")
        return self

    def build_vocab(self):
        """构建词表"""
        tokens_list = self.train_df['Phrase'].apply(Utils.clean_text).apply(lambda x: x.split()).values
        self.vocab = VocabBuilder(self.config.min_freq).build_from_tokens(tokens_list)
        return self

    def create_dataloaders(self):
        """创建数据加载器"""
        generator = torch.Generator().manual_seed(self.config.seed)

        train_loader = DataLoader(
            ReviewDataset(self.train_df, self.vocab, self.config.max_len),
            batch_size=self.config.batch_size, shuffle=True, generator=generator,
            drop_last=True  # 丢弃最后一个不完整批次
        )
        val_loader = DataLoader(
            ReviewDataset(self.val_df, self.vocab, self.config.max_len),
            batch_size=self.config.batch_size, shuffle=False,
            drop_last=False
        )
        test_loader = DataLoader(
            ReviewDataset(self.test_df, self.vocab, self.config.max_len, is_test=True),
            batch_size=self.config.batch_size, shuffle=False,
            drop_last=False
        )
        return train_loader, val_loader, test_loader


# ==================== 模型模块 ====================
class BiLSTMClassifier(nn.Module):
    """双向LSTM文本分类模型"""

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            vocab_size, config.embedding_dim,
            padding_idx=0,
            _weight=torch.randn(vocab_size, config.embedding_dim)  # 显式初始化权重
        )
        self.lstm = nn.LSTM(
            config.embedding_dim, config.hidden_dim, config.n_layers,
            bidirectional=True, batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            proj_size=0
        )
        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text):
        # 验证输入维度
        assert text.dim() == 2, f"输入维度错误，应为2D，实际: {text.dim()}"
        batch_size, seq_len = text.shape

        embedded = self.dropout(self.embedding(text))
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 验证LSTM输出维度
        assert hidden.shape == (self.config.n_layers * 2, batch_size, self.config.hidden_dim), \
            f"LSTM隐藏层维度错误: {hidden.shape}"

        # 拼接双向最后一层输出
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(self.dropout(hidden_cat))

        # 验证输出维度
        assert output.shape == (batch_size, self.config.output_dim), \
            f"输出维度错误: {output.shape}"
        return output


# ==================== 训练器模块 ====================
class ModelTrainer:
    def __init__(self, config, vocab_size):
        self.config = config

        # 设备选择
        if config.use_cpu or not torch.cuda.is_available():
            self.device = torch.device('cpu')
            print("使用CPU运行（强制或无GPU）")
        else:
            self.device = torch.device('cuda')
            print(f"使用GPU运行: {torch.cuda.get_device_name(0)}")

        # 模型初始化
        self.model = BiLSTMClassifier(vocab_size, config).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-8  # 防止数值不稳定
        )

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数总数: {total_params:,}")

    def train_epoch(self, train_loader):
        """单轮训练"""
        self.model.train()
        total_loss, total_correct = 0, 0

        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(self.device), labels.to(self.device)

            # 验证批次数据
            assert texts.dtype == torch.long, "输入类型错误"
            assert labels.dtype == torch.long, "标签类型错误"

            self.optimizer.zero_grad()
            preds = self.model(texts)

            # 验证预测输出
            assert preds.shape[0] == labels.shape[0], "批次大小不匹配"

            loss = self.criterion(preds, labels)
            loss.backward()

            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (preds.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / (len(train_loader) * train_loader.batch_size)
        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """单轮验证"""
        self.model.eval()
        total_loss, total_correct = 0, 0

        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                preds = self.model(texts)
                loss = self.criterion(preds, labels)

                total_loss += loss.item()
                total_correct += (preds.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / len(val_loader.dataset)
        return avg_loss, accuracy

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print("\n开始训练...")
        best_val_acc = 0

        for epoch in range(self.config.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"Best Val Acc: {best_val_acc:.4f}")

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self

    def predict(self, test_loader, test_df):
        """测试集预测并保存结果"""
        print("\n生成测试集预测...")
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for texts in test_loader:
                texts = texts.to(self.device)
                preds = self.model(texts).argmax(1)
                all_preds.extend(preds.cpu().numpy())

        # 确保预测结果有效
        all_preds = [p if p in [0, 1, 2, 3, 4] else 2 for p in all_preds]

        pd.DataFrame({
            'PhraseId': test_df['PhraseId'],
            'Sentiment': all_preds
        }).to_csv(self.config.output_path, index=False)
        print(f"预测完成！结果已保存至 {self.config.output_path}")
        print(f"预测样本数: {len(all_preds)}")


# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 可选：强制使用CPU排查问题
    # config.use_cpu = True

    # 设置调试环境和随机种子
    Utils.setup_debug_env(config)
    Utils.set_seed(config.seed)

    try:
        # 数据准备
        data_manager = DataManager(config).load_raw_data().build_vocab()
        train_loader, val_loader, test_loader = data_manager.create_dataloaders()

        # 模型训练与预测
        trainer = ModelTrainer(config, data_manager.vocab.vocab_size)
        trainer.train(train_loader, val_loader)
        trainer.predict(test_loader, data_manager.test_df)

        print("\n程序执行完成！")

    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()