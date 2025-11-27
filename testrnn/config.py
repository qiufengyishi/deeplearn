class Config:
    # 数据配置
    DATA_PATH = {
        'train': 'train.tsv/train.tsv',
        'test': 'test.tsv/test.tsv'
    }
    
    # 模型配置
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 5  # 5个情感类别
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    # 训练配置
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    MAX_LENGTH = 50  # 最大序列长度
    
    # 词汇表配置
    MIN_FREQ = 5  # 最小词频
    
    # 保存路径
    MODEL_SAVE_PATH = 'textrnn_model.pth'
    PREDICTIONS_SAVE_PATH = 'predictions.csv'