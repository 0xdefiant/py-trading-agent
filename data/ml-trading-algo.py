import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import click

# =========================
# Tips for Efficient and Accurate ML Training
# =========================
# - Use more historical data for training (longer timeframes, more tokens/pairs).
# - Tune hyperparameters: seq_len, hidden_size, num_layers, batch_size, learning rate, epochs.
# - Try both LSTM and GRU, and compare validation/test MSE.
# - Use early stopping or validation loss to avoid overfitting.

# =========================
# 1. Data Loading & Preprocessing
# =========================
class CryptoDataset(Dataset):
    def __init__(self, df, seq_len=24, feature_col='close'):
        self.seq_len = seq_len
        self.data = df[feature_col].values.astype(np.float32)
        self.X, self.y = self.create_sequences(self.data, seq_len)

    def create_sequences(self, data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_token_csvs(token_dir):
    files = glob.glob(os.path.join(token_dir, '*.csv'))
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        if 'close' in df.columns:
            df_list.append(df)
    if not df_list:
        raise ValueError(f'No valid CSVs found in {token_dir}')
    return pd.concat(df_list, ignore_index=True)

# =========================
# 2. Model Definitions
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# =========================
# 3. Training & Evaluation
# =========================
def train_model(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            X, y = X.unsqueeze(-1).to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return model

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(-1).to(device)
            output = model(X)
            preds.extend(output.cpu().numpy())
            trues.extend(y.numpy())
    preds, trues = np.array(preds), np.array(trues)
    mse = np.mean((preds - trues) ** 2)
    print(f"Test MSE: {mse:.6f}")
    return preds, trues

# =========================
# 4. CLI Entry Point
# =========================
@click.command()
@click.option('--token', required=True, type=str, help='Token folder name (e.g., bitcoin, ripple, ethereum)')
@click.option('--model', default='lstm', type=click.Choice(['lstm', 'gru']), help='Model type to use (lstm or gru)')
@click.option('--seq-len', default=24, type=int, help='Sequence length (number of hours for input window)')
@click.option('--batch-size', default=64, type=int, help='Batch size for training')
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--device', default='cpu', type=str, help='Device to use (cpu or cuda)')
@click.option('--save-model', default=None, type=str, help='Path or filename to save trained model weights (e.g., bitcoin_lstm.pth)')
def cli(token, model, seq_len, batch_size, epochs, lr, device, save_model):
    token_dir = os.path.join(os.path.dirname(__file__), 'tokenData', token)
    df = load_token_csvs(token_dir)
    # Normalize close price
    df['close_norm'] = (df['close'] - df['close'].mean()) / df['close'].std()
    dataset = CryptoDataset(df, seq_len=seq_len, feature_col='close_norm')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    if model == 'lstm':
        net = LSTMModel()
    else:
        net = GRUModel()
    print(f"Training {model.upper()} on {token} ({len(df)} rows)...")
    net = train_model(net, train_loader, epochs=epochs, lr=lr, device=device)
    # Save model weights if requested
    if save_model:
        # If save_model is just a filename, save to /data/models/<filename>
        if not os.path.isabs(save_model) and not os.path.dirname(save_model):
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(models_dir, save_model)
        else:
            save_path = save_model
        torch.save(net.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
    print("Evaluating on test set...")
    preds, trues = evaluate_model(net, test_loader, device=device)
    # Optionally: plot predictions vs. true values
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.plot(trues, label='True')
        plt.plot(preds, label='Predicted')
        plt.title(f'{model.upper()} Prediction on {token}')
        plt.legend()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    cli()
