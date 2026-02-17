from core.dataloader import ParquetDataLoader
from core.models import build_lstm_model, build_gru_model
import pandas as pd  # 新增导入
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # 新增导入
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import sys
import os

# 1. 【关键】将项目根目录加入搜索路径，必须在导入 core 模块之前执行
# 这样无论是作为工具被调用，还是单独运行此脚本，都能找到 core 包
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# 2. 改为绝对导入，确保在 Server 环境下能正确找到模块


def train_and_compare(ticker='AAPL', lookback=30, epochs=50, batch_size=32, out_dir='models', period='5y'):
    out_dir = os.path.abspath(out_dir)  # 确保使用绝对路径
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[Dataset] Using Parquet Dataset (QFQ, Full History) for ticker: {ticker}")

    loader = ParquetDataLoader(
        ticker=ticker,
        sequence_length=lookback,
        period=period,
        interval='1d'
    )
    X, y, scaler = loader.get_data(use_cache=True)

    if loader.df is not None:
        start_date = loader.df.index.min()
        end_date = loader.df.index.max()
        print(
            f"[Data Info] Loaded {len(loader.df)} rows from {start_date} to {end_date}")

    y = y.reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False)

    input_shape = (X.shape[1], X.shape[2])
    lstm = build_lstm_model(input_shape)
    gru = build_gru_model(input_shape)

    es = EarlyStopping(monitor='val_loss', patience=5,
                       restore_best_weights=True)
    lstm_path = os.path.join(out_dir, 'best_lstm.keras')
    gru_path = os.path.join(out_dir, 'best_gru.keras')

    cp_lstm = ModelCheckpoint(
        lstm_path, monitor='val_loss', save_best_only=True)
    cp_gru = ModelCheckpoint(gru_path, monitor='val_loss', save_best_only=True)

    history_lstm = lstm.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, callbacks=[es, cp_lstm], verbose=1)
    lstm_loss = lstm.evaluate(X_test, y_test, verbose=0)

    history_gru = gru.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=epochs, batch_size=batch_size, callbacks=[es, cp_gru], verbose=1)
    gru_loss = gru.evaluate(X_test, y_test, verbose=0)

    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    plt.figure(figsize=(8, 4))
    plt.plot(history_lstm.history['loss'], label='LSTM loss')
    plt.plot(history_lstm.history.get('val_loss', []), label='LSTM val_loss')
    plt.plot(history_gru.history['loss'], label='GRU loss')
    plt.plot(history_gru.history.get('val_loss', []), label='GRU val_loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'training_loss.png'))

    # 比较MSE (evaluate返回列表时取第一个值)
    val_l = lstm_loss[0] if isinstance(lstm_loss, list) else lstm_loss
    val_g = gru_loss[0] if isinstance(gru_loss, list) else gru_loss

    if val_l < val_g:
        winner = 'LSTM'
        winner_path = lstm_path
        winner_loss = val_l
    else:
        winner = 'GRU'
        winner_path = gru_path
        winner_loss = val_g

    return {
        'winner': winner,
        'winner_path': winner_path,
        'winner_loss': float(winner_loss),
        'lstm_loss': float(val_l),
        'gru_loss': float(val_g),
        'scaler_path': os.path.join(out_dir, 'scaler.joblib'),
        'loss_plot_path': os.path.join(out_dir, 'training_loss.png')
    }


def run_analysis(ticker='AAPL', period='5y'):
    """
    全流程函数：训练模型 -> 择优 -> 预测下一天股价
    供 MCP Tool 调用
    """
    # 1. 训练并择优
    print(f"[INFO] Starting analysis for {ticker}...")
    result = train_and_compare(ticker=ticker, period=period)

    winner_path = result['winner_path']
    print(f"[INFO] Winner model: {result['winner']} (saved at {winner_path})")

    # 2. 加载最优模型
    model = load_model(winner_path)

    # 3. 获取最新的序列数据进行推理
    loader = ParquetDataLoader(
        ticker=ticker, sequence_length=30, period=period)
    last_seq_scaled, scaler, last_date = loader.get_latest_sequence()

    # 4. 预测
    pred_scaled = model.predict(last_seq_scaled)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    # 5. 格式化结果
    next_date = last_date + pd.Timedelta(days=1)

    return {
        "ticker": ticker,
        "winner_model": result['winner'],
        "accuracy_mse": result['winner_loss'],
        "last_actual_date": str(last_date.date()),
        "next_prediction_date": str(next_date.date()),
        "predicted_price": float(pred_price),
        "plot_path": result['loss_plot_path']
    }


if __name__ == "__main__":
    # 测试训练
    res = run_analysis(ticker='AAPL')
    print("最终预测结果：", res)
