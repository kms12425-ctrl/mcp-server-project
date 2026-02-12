import sys
import os
# 1. 确保项目根目录加入搜索路径（适配tools模块）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入依赖（保留你的原有导入，仅适配数据加载器）
from dataloader import StockDataLoader  # 导入不可修改的StockDataLoader
from models import build_lstm_model, build_gru_model

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def train_and_compare(ticker='AAPL', lookback=30, epochs=50, batch_size=32, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    
    # ========== 核心修改1：适配get_data()的返回值 ==========
    # 原代码：(X, y), scaler = StockDataLoader(...).get_data()
    # 新代码：X, y, scaler = StockDataLoader(...).get_data()
    loader = StockDataLoader(
        ticker=ticker,
        sequence_length=lookback,  # 原lookback对应这里的sequence_length
        period='1y',  # 可根据需求调整，比如'6mo'/'2y'
        interval='1d'
    )
    X, y, scaler = loader.get_data(use_cache=True)  # 直接接收三个返回值
    
    # ========== 核心修改2：对齐数据形状（y的维度） ==========
    # 原代码中y是(N,1)，数据加载器返回的y是(N,)，需要reshape保持一致
    y = y.reshape(-1, 1)
    
    # 数据集划分（你的原有逻辑，无需修改）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # 构建模型（你的原有逻辑，无需修改）
    input_shape = (X.shape[1], X.shape[2])
    lstm = build_lstm_model(input_shape)
    gru = build_gru_model(input_shape)

    # 回调函数（你的原有逻辑，无需修改）
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cp_lstm = ModelCheckpoint(os.path.join(out_dir, 'best_lstm.h5'), monitor='val_loss', save_best_only=True)
    cp_gru = ModelCheckpoint(os.path.join(out_dir, 'best_gru.h5'), monitor='val_loss', save_best_only=True)

    # 训练模型（你的原有逻辑，无需修改）
    history_lstm = lstm.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[es, cp_lstm], verbose=1)
    lstm_loss = lstm.evaluate(X_test, y_test, verbose=0)

    history_gru = gru.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=epochs, batch_size=batch_size, callbacks=[es, cp_gru], verbose=1)
    gru_loss = gru.evaluate(X_test, y_test, verbose=0)

    # 保存scaler（你的原有逻辑，无需修改）
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    # 绘图对比（你的原有逻辑，无需修改）
    plt.figure(figsize=(8,4))
    plt.plot(history_lstm.history['loss'], label='LSTM loss')
    plt.plot(history_lstm.history.get('val_loss', []), label='LSTM val_loss')
    plt.plot(history_gru.history['loss'], label='GRU loss')
    plt.plot(history_gru.history.get('val_loss', []), label='GRU val_loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'training_loss.png'))

    # 选择最优模型（注意：lstm_loss/gru_loss是列表，比较第一个值即MSE）
    if lstm_loss[0] < gru_loss[0]:
        winner = 'lstm'
        winner_path = os.path.join(out_dir, 'best_lstm.h5')
    else:
        winner = 'gru'
        winner_path = os.path.join(out_dir, 'best_gru.h5')

    return {
        'winner': winner, 
        'winner_path': winner_path, 
        'lstm_loss': lstm_loss, 
        'gru_loss': gru_loss,
        'scaler_path': os.path.join(out_dir, 'scaler.joblib'),
        'loss_plot_path': os.path.join(out_dir, 'training_loss.png')
    }

# 测试调用（新增：方便你验证）
if __name__ == "__main__":
    # 测试训练AAPL模型
    result = train_and_compare(ticker='AAPL', lookback=30, epochs=10)  # 先少训练几轮验证
    print("训练结果：")
    print(f"最优模型：{result['winner']}")
    print(f"LSTM测试损失（MSE, MAE）：{result['lstm_loss']}")
    print(f"GRU测试损失（MSE, MAE）：{result['gru_loss']}")
    print(f"最优模型路径：{result['winner_path']}")