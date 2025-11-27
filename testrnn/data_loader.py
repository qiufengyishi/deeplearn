import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import re

class MovieReviewDataset(Dataset):
    def __init__(self, phrases, sentiments, word_to_idx, max_length=50):
        self.phrases = phrases
        self.sentiments = sentiments
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        sentiment = self.sentiments[idx]
        
        # 文本预处理和编码
        tokens = self.preprocess_text(phrase)
        encoded = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in tokens]
        
        # 填充或截断到固定长度
        if len(encoded) < self.max_length:
            encoded = encoded + [self.word_to_idx['<PAD>']] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
            
        return torch.tensor(encoded), torch.tensor(sentiment)
    
    def preprocess_text(self, text):
        # 简单的文本预处理
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return tokens

class DataProcessor:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        # 构建词汇表
        word_counts = Counter()
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words = text.split()
            word_counts.update(words)
            
        # 过滤低频词
        vocab = [word for word, count in word_counts.items() if count >= self.min_freq]
        vocab = ['<PAD>', '<UNK>'] + vocab
        
        # 创建映射
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        return self.vocab_size
    
    def load_data(self, train_path, test_path):
        # 加载训练数据
        train_df = pd.read_csv(train_path, sep='\t')
        train_phrases = train_df['Phrase'].tolist()
        train_sentiments = train_df['Sentiment'].tolist()
        
        # 构建词汇表
        self.build_vocab(train_phrases)
        
        # 加载测试数据
        test_df = pd.read_csv(test_path, sep='\t')
        test_phrases = test_df['Phrase'].tolist()
        
        return train_phrases, train_sentiments, test_phrases
    
    def create_data_loaders(self, train_phrases, train_sentiments, batch_size=32, max_length=50):
        # 创建训练和验证集
        train_phrases, val_phrases, train_sentiments, val_sentiments = train_test_split(
            train_phrases, train_sentiments, test_size=0.2, random_state=42, stratify=train_sentiments
        )
        
        # 创建数据集
        train_dataset = MovieReviewDataset(train_phrases, train_sentiments, self.word_to_idx, max_length)
        val_dataset = MovieReviewDataset(val_phrases, val_sentiments, self.word_to_idx, max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader