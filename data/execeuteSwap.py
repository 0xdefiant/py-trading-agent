import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'currentData', 'utils'))
import pandas as pd
import numpy as np
import torch

# Fallback LSTMModel definition (if import fails)
try:
    from ml_trading_algo import LSTMModel
except ImportError:
    import torch.nn as nn
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)

def get_latest_eth_ohlcv_csv():
    data_dir = os.path.join(os.path.dirname(__file__), 'currentData')
    files = [f for f in os.listdir(data_dir) if f.lower().startswith('WETH_USDC') and f.endswith('.csv')]
    if not files:
        # fallback: use any csv with ETH in symbol
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)), reverse=True)
    return os.path.join(data_dir, files[0]) if files else None

def load_ohlcv(csv_path):
    df = pd.read_csv(csv_path)
    if 'close' not in df.columns:
        raise ValueError('No close column in OHLCV data')
    return df

def normalize_close(df):
    mean = df['close'].mean()
    std = df['close'].std()
    df['close_norm'] = (df['close'] - mean) / std
    return df, mean, std

def main():
    # 1. Load latest ETH OHLCV data
    csv_path = get_latest_eth_ohlcv_csv()
    if not csv_path:
        print('No ETH OHLCV CSV found.')
        return
    df = load_ohlcv(csv_path)
    df, mean, std = normalize_close(df)
    seq_len = 24
    if len(df) < seq_len:
        print(f'Not enough data for sequence length {seq_len}')
        return
    # 2. Prepare input sequence
    input_seq = df['close_norm'].values[-seq_len:]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # shape: (1, seq_len, 1)
    # 3. Load model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'ethereum_lstm.pth'))
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    # 4. Predict next close (normalized)
    with torch.no_grad():
        pred_norm = model(input_tensor).item()
    pred_close = pred_norm * std + mean
    last_close = df['close'].values[-1]
    # 5. Trade signal logic
    threshold = 0.001  # 0.1% move
    if pred_close > last_close * (1 + threshold):
        action = 'BUY'
    elif pred_close < last_close * (1 - threshold):
        action = 'SELL'
    else:
        action = 'HOLD'
    print(f"Last close: {last_close:.6f}, Predicted next close: {pred_close:.6f}")
    print(f"Trade signal: {action}")

if __name__ == "__main__":
    main()
