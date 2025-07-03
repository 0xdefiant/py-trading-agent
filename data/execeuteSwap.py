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
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3, classification=False):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.classification = classification
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            if self.classification:
                return torch.softmax(out, dim=2)
            else:
                return out.squeeze(-1)

def get_latest_eth_ohlcv_csv():
    data_dir = os.path.join(os.path.dirname(__file__), 'currentData')
    # Only select files that start with 'WETH_USDC'
    files = [f for f in os.listdir(data_dir) if f.startswith('WETH_USDC') and f.endswith('.csv')]
    if not files:
        # fallback: use any csv
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
    print(f"Selected OHLCV CSV: {csv_path}")
    df = load_ohlcv(csv_path)
    # Use the exact feature columns and order as in training
    feature_cols = [
        'open', 'high', 'low', 'close',
        'Volume ETH', 'Volume USD',
        'return', 'log_return', 'SMA_24', 'volatility_24'
    ]
    # Check for missing columns
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")
    seq_len = 24  # Model's expected input length
    if len(df) < seq_len:
        print(f'Not enough data for sequence length {seq_len}')
        return
    # 2. Prepare input sequence (most recent first)
    input_df = df[feature_cols].iloc[:seq_len].copy()
    # Normalize each feature column
    means = input_df.mean()
    stds = input_df.std().replace(0, 1)  # avoid division by zero
    input_df_norm = (input_df - means) / stds
    print("Input sequence (normalized, most recent 5 rows):\n", input_df_norm.head())
    input_tensor = torch.tensor(input_df_norm.values, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, num_features)
    # 3. Load model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'ethereum_lstm_1h.pth'))
    print(f"Loading model from: {model_path}")
    model = LSTMModel(input_size=len(feature_cols), output_size=3, classification=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    # 4. Predict next class (classification)
    with torch.no_grad():
        logits = model(input_tensor)
        # If logits are 3D (batch, seq, class), squeeze to (batch, class)
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred_class = int(np.argmax(probs))
        # Print detailed model output
        print(f"Raw logits: {logits.cpu().numpy().flatten()}")
        print(f"Class probabilities: {probs}")
        print(f"Predicted class: {pred_class} (probability: {probs[pred_class]:.4f})")
    last_close = input_df['close'].iloc[0]  # Most recent close
    # 5. Trade signal logic (0=HOLD, 1=BUY, 2=SELL)
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action = action_map.get(pred_class, 'HOLD')
    print(f"Last close (most recent): {last_close:.6f}")
    print(f"Trade signal: {action}")

if __name__ == "__main__":
    main()
