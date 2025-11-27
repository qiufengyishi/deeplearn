import torch
import pandas as pd
from data_loader import DataProcessor, MovieReviewDataset
from model import TextRNNClassifier
from torch.utils.data import DataLoader

def predict_sentiment():
    try:
        # 加载模型
        checkpoint = torch.load('textrnn_model.pth', map_location=torch.device('cpu'))
        
        # 重新创建数据处理器以获取词汇表
        processor = DataProcessor()
        processor.word_to_idx = checkpoint['word_to_idx']
        processor.vocab_size = checkpoint['vocab_size']
        
        # 重新创建模型
        classifier = TextRNNClassifier(
            vocab_size=processor.vocab_size,
            embedding_dim=100,
            hidden_dim=128,
            output_dim=5,
            n_layers=2,
            bidirectional=True,
            dropout=0.5,
            pad_idx=0
        )
        
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载测试数据并清理空值
        test_df = pd.read_csv('test.tsv/test.tsv', sep='\t')
        # 移除空值
        test_df = test_df.dropna(subset=['Phrase'])
        test_phrases = test_df['Phrase'].tolist()
        
        print(f"Processing {len(test_phrases)} test samples...")
        
        # 创建测试数据集
        test_dataset = MovieReviewDataset(test_phrases, [0]*len(test_phrases), 
                                        processor.word_to_idx, max_length=50)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 进行预测
        print("Making predictions...")
        predictions = classifier.predict(test_loader)
        
        # 保存结果
        result_df = test_df.copy()
        result_df['Sentiment'] = predictions
        result_df.to_csv('predictions.csv', index=False)
        
        print("Predictions saved to predictions.csv")
        
        # 显示一些预测示例
        print("\nPrediction examples:")
        for i in range(min(5, len(test_phrases))):
            print(f"Phrase: {test_phrases[i][:50]}...")
            print(f"Predicted Sentiment: {predictions[i]}")
            print("-" * 50)
            
        # 统计预测结果分布
        sentiment_counts = pd.Series(predictions).value_counts().sort_index()
        print("\nSentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            print(f"{sentiment_labels[sentiment]}: {count} samples")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def predict_single_text(text, model_path='textrnn_model.pth'):
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 重新创建数据处理器
        processor = DataProcessor()
        processor.word_to_idx = checkpoint['word_to_idx']
        
        # 重新创建模型
        classifier = TextRNNClassifier(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=100,
            hidden_dim=128,
            output_dim=5,
            n_layers=2,
            bidirectional=True,
            dropout=0.5,
            pad_idx=0
        )
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 预处理文本
        dataset = MovieReviewDataset([text], [0], processor.word_to_idx, max_length=50)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 预测
        prediction = classifier.predict(loader)[0]
        sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        
        return sentiment_labels[prediction], prediction
        
    except Exception as e:
        print(f"Error during single text prediction: {e}")
        return "Error", -1

if __name__ == '__main__':
    # 批量预测测试集
    predict_sentiment()
    
    # 单文本预测示例
    test_texts = [
        "This movie is absolutely fantastic and well worth watching!",
        "Terrible acting and boring storyline.",
        "It's an okay movie, nothing special.",
        "One of the best films I've ever seen!",
        "Waste of time and money."
    ]
    
    print(f"\nSingle text predictions:")
    for test_text in test_texts:
        sentiment, score = predict_single_text(test_text)
        print(f"Text: {test_text}")
        print(f"Sentiment: {sentiment} (Score: {score})")
        print("-" * 60)