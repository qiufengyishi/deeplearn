import argparse
import torch
from data_loader import DataProcessor
from model import TextRNNClassifier
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='TextRNN for Movie Review Sentiment Analysis')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    
    # 数据预处理
    print("Loading and preprocessing data...")
    processor = DataProcessor(min_freq=5)
    train_phrases, train_sentiments, test_phrases = processor.load_data(
        'train.tsv/train.tsv', 
        'test.tsv/test.tsv'
    )
    
    print(f"Vocabulary size: {processor.vocab_size}")
    print(f"Training samples: {len(train_phrases)}")
    print(f"Test samples: {len(test_phrases)}")
    
    # 创建数据加载器
    train_loader, val_loader = processor.create_data_loaders(
        train_phrases, train_sentiments, 
        batch_size=args.batch_size, max_length=50
    )
    
    # 初始化模型
    print("Initializing model...")
    classifier = TextRNNClassifier(
        vocab_size=processor.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=5,
        n_layers=args.n_layers,
        bidirectional=True,
        dropout=args.dropout,
        pad_idx=0
    )
    
    # 训练模型
    print("Starting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = classifier.train_epoch(train_loader)
        val_loss, val_acc = classifier.evaluate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
    
    # 保存模型
    torch.save({
        'model_state_dict': classifier.model.state_dict(),
        'word_to_idx': processor.word_to_idx,
        'vocab_size': processor.vocab_size
    }, 'textrnn_model.pth')
    
    print("Model saved as textrnn_model.pth")
    
    # 绘制训练曲线（只保存图片，不显示）
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，释放内存
        
        print("Training curves saved as training_curves.png")
        
    except Exception as e:
        print(f"Warning: Could not save training curves: {e}")
        print("Training completed successfully, but visualization failed.")
    
    # 打印最终训练结果摘要
    print("\n=== Training Summary ===")
    print(f"Final Training Accuracy: {train_accs[-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {val_accs[-1]*100:.2f}%")
    print(f"Best Validation Accuracy: {max(val_accs)*100:.2f}%")
    
    # 检查模型文件是否成功创建
    if os.path.exists('textrnn_model.pth'):
        file_size = os.path.getsize('textrnn_model.pth') / (1024 * 1024)  # MB
        print(f"Model file size: {file_size:.2f} MB")
        print("Model training completed successfully!")
    else:
        print("Warning: Model file was not created!")

if __name__ == '__main__':
    main()