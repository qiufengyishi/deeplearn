import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          dropout=dropout,
                          batch_first=True)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        # hidden: [batch_size, hidden_dim * num_directions]
        
        return self.fc(hidden)

class TextRNNClassifier:
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=5, 
                 n_layers=2, bidirectional=True, dropout=0.5, pad_idx=0):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TextRNN(vocab_size, embedding_dim, hidden_dim, output_dim, 
                           n_layers, bidirectional, dropout, pad_idx)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        for batch in train_loader:
            text, labels = batch
            text, labels = text.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(text)
            loss = self.criterion(predictions, labels)
            
            acc = self.calculate_accuracy(predictions, labels)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(train_loader), epoch_acc / len(train_loader)
    
    def evaluate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text, labels = batch
                text, labels = text.to(self.device), labels.to(self.device)
                
                predictions = self.model(text)
                loss = self.criterion(predictions, labels)
                acc = self.calculate_accuracy(predictions, labels)
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
        return epoch_loss / len(val_loader), epoch_acc / len(val_loader)
    
    def calculate_accuracy(self, predictions, labels):
        max_preds = predictions.argmax(dim=1)
        correct = (max_preds == labels).float()
        return correct.sum() / len(correct)
    
    def predict(self, text_loader):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in text_loader:
                text = batch[0].to(self.device)
                batch_preds = self.model(text)
                predictions.extend(batch_preds.argmax(dim=1).cpu().numpy())
                
        return predictions