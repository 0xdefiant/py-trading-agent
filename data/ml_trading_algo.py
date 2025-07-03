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
    def __init__(self, df, seq_len=24, feature_cols=None, label_mode='regression', return_threshold_buy=0.09, return_threshold_sell=-0.02):
        self.seq_len = seq_len
        # Use all numeric columns except unix (and not object dtype)
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ['unix'] and df[col].dtype != 'O' and col != 'close']
            # Always include 'close' as a feature
            if 'close' in df.columns:
                feature_cols = ['close'] + [c for c in feature_cols if c != 'close']
        self.feature_cols = feature_cols
        self.data = df[feature_cols].values.astype(np.float32)
        self.label_mode = label_mode
        self.return_threshold_buy = return_threshold_buy
        self.return_threshold_sell = return_threshold_sell
        if label_mode == 'classification':
            self.labels = self.create_labels(df['close'].values, seq_len, return_threshold_buy, return_threshold_sell)
        else:
            self.labels = df['close'].values.astype(np.float32)
        self.X, self.y = self.create_sequences(self.data, self.labels, seq_len)

    def create_sequences(self, data, labels, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len - 1):
            X.append(data[i:i+seq_len])
            y.append(labels[i+seq_len])
        return np.array(X), np.array(y)

    def create_labels(self, close_prices, seq_len, threshold_buy, threshold_sell):
        # Label: 0=hold, 1=buy, 2=sell
        labels = np.zeros(len(close_prices))
        for i in range(len(close_prices) - 1):
            ret = (close_prices[i+1] - close_prices[i]) / close_prices[i]
            if ret > threshold_buy:
                labels[i] = 1  # buy
            elif ret < threshold_sell:
                labels[i] = 2  # sell
            else:
                labels[i] = 0  # hold
        labels[-1] = 0  # last label is hold
        return labels.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_token_csvs(token_dir, pattern='*.csv'):
    files = glob.glob(os.path.join(token_dir, pattern))
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        # If 'date' column exists, sort by it; otherwise, sort by 'unix' if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        elif 'unix' in df.columns:
            df = df.sort_values('unix').reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        if 'close' in df.columns:
            df_list.append(df)
    if not df_list:
        raise ValueError(f'No valid CSVs found in {token_dir} with pattern {pattern}')
    return pd.concat(df_list, ignore_index=True)

# =========================
# 2. Model Definitions
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, classification=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.classification = classification
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        if self.classification:
            return out  # logits
        return out.squeeze(-1)

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, classification=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.classification = classification
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        if self.classification:
            return out  # logits
        return out.squeeze(-1)

# =========================
# 3. Training & Evaluation
# =========================
def train_model(model, dataloader, epochs=10, lr=1e-3, device='cpu', classification=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for X, y in dataloader:
            X = X.to(device)
            if classification:
                y = y.long().to(device)
            else:
                X = X.unsqueeze(-1)
                y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            if classification:
                loss = criterion(output, y)
                preds = output.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            else:
                loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        if classification:
            acc = correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Acc: {acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return model

def evaluate_model(model, dataloader, device='cpu', classification=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            if classification:
                y = y.long().to(device)
            else:
                X = X.unsqueeze(-1)
                y = y.to(device)
            output = model(X)
            if classification:
                pred = output.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(y.cpu().numpy())
            else:
                preds.extend(output.cpu().numpy())
                trues.extend(y.cpu().numpy())
    preds, trues = np.array(preds), np.array(trues)
    if classification:
        acc = np.mean(preds == trues)
        print(f"Test Accuracy: {acc:.4f}")
    else:
        mse = np.mean((preds - trues) ** 2)
        print(f"Test MSE: {mse:.6f}")
    return preds, trues

# =========================
# 4. CLI Entry Point
# =========================
def train_token(token, model='lstm', seq_len=24, batch_size=64, epochs=10, lr=1e-3, device='cpu', save_model=None, plot=True, classification=True, return_threshold_buy=0.09, return_threshold_sell=-0.02, features_only=False):
    token_dir = os.path.join(os.path.dirname(__file__), 'tokenData', token)
    # Look for both _features_1h.csv and _features_d.csv
    freq_files = [("1h", f"{token}_features_1h.csv"), ("d", f"{token}_features_d.csv")]
    for freq, fname in freq_files:
        fpath = os.path.join(token_dir, fname)
        if not os.path.exists(fpath):
            continue
        print(f"\n=== Training on {token} ({freq}) ===")
        df = pd.read_csv(fpath)
        # Normalize all numeric features (float or int only)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_timedelta64_dtype(df[col]) and col not in ['unix']]
        for col in numeric_cols:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        feature_cols = [col for col in df.columns if col not in ['unix'] and df[col].dtype != 'O' and col != 'close']
        if 'close' in df.columns:
            feature_cols = ['close'] + [c for c in feature_cols if c != 'close']
        dataset = CryptoDataset(df, seq_len=seq_len, feature_cols=feature_cols, label_mode='classification' if classification else 'regression', return_threshold_buy=return_threshold_buy, return_threshold_sell=return_threshold_sell)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        input_size = len(feature_cols)
        output_size = 3 if classification else 1
        if model == 'lstm':
            net = LSTMModel(input_size=input_size, output_size=output_size, classification=classification)
        else:
            net = GRUModel(input_size=input_size, output_size=output_size, classification=classification)
        print(f"Training {model.upper()} on {token} {freq} ({len(df)} rows)...")
        net = train_model(net, train_loader, epochs=epochs, lr=lr, device=device, classification=classification)
        # Save model weights if requested or by default
        if save_model:
            if not os.path.isabs(save_model) and not os.path.dirname(save_model):
                models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
                os.makedirs(models_dir, exist_ok=True)
                base, ext = os.path.splitext(save_model)
                save_path = os.path.join(models_dir, f"{token}_{model}_{freq}{ext if ext else '.pth'}")
            else:
                base, ext = os.path.splitext(save_model)
                save_path = f"{base}_{freq}{ext if ext else '.pth'}"
        else:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
            os.makedirs(models_dir, exist_ok=True)
            save_path = os.path.join(models_dir, f"{token}_{model}_{freq}.pth")
        torch.save(net.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
        print("Evaluating on test set...")
        preds, trues = evaluate_model(net, test_loader, device=device, classification=classification)
        # Optionally: plot predictions vs. true values
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                if classification:
                    plt.plot(trues, label='True')
                    plt.plot(preds, label='Predicted')
                    plt.title(f'{model.upper()} Buy/Sell/Hold Classification on {token} {freq}')
                else:
                    plt.plot(trues, label='True')
                    plt.plot(preds, label='Predicted')
                    plt.title(f'{model.upper()} Prediction on {token} {freq}')
                plt.legend()
                plt.show()
            except ImportError:
                pass
    return None, None, None

@click.group()
def main():
    pass

@main.command()
@click.option('--token', required=True, type=str, help='Token folder name (e.g., bitcoin, ripple, ethereum)')
@click.option('--model', default='lstm', type=click.Choice(['lstm', 'gru']), help='Model type to use (lstm or gru)')
@click.option('--seq-len', default=24, type=int, help='Sequence length (number of hours for input window)')
@click.option('--batch-size', default=64, type=int, help='Batch size for training')
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--device', default='cpu', type=str, help='Device to use (cpu or cuda)')
@click.option('--save-model', default=None, type=str, help='Path or filename to save trained model weights (e.g., bitcoin_lstm.pth)')
@click.option('--return-threshold-buy', default=0.09, type=float, help='Return threshold for buy signal (e.g., 0.09 for 9%)')
@click.option('--return-threshold-sell', default=-0.02, type=float, help='Return threshold for sell signal (e.g., -0.02 for -2%)')
@click.option('--features-only', is_flag=True, help='Only use files ending with _features.csv in the token folder')
def train(token, model, seq_len, batch_size, epochs, lr, device, save_model, return_threshold_buy, return_threshold_sell, features_only):
    train_token(token, model=model, seq_len=seq_len, batch_size=batch_size, epochs=epochs, lr=lr, device=device, save_model=save_model, plot=True, classification=True, return_threshold_buy=return_threshold_buy, return_threshold_sell=return_threshold_sell, features_only=features_only)

@main.command('train-all')
@click.option('--model', default='lstm', type=click.Choice(['lstm', 'gru']), help='Model type to use (lstm or gru)')
@click.option('--seq-len', default=24, type=int, help='Sequence length (number of hours for input window)')
@click.option('--batch-size', default=64, type=int, help='Batch size for training')
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--lr', default=1e-3, type=float, help='Learning rate')
@click.option('--device', default='cpu', type=str, help='Device to use (cpu or cuda)')
@click.option('--save-model', default=None, type=str, help='Path or filename to save trained model weights (e.g., bitcoin_lstm.pth)')
@click.option('--return-threshold-buy', default=0.09, type=float, help='Return threshold for buy signal (e.g., 0.09 for 9%)')
@click.option('--return-threshold-sell', default=-0.02, type=float, help='Return threshold for sell signal (e.g., -0.02 for -2%)')
@click.option('--features-only', is_flag=True, help='Only use files ending with _features.csv in the token folder')
def train_all(model, seq_len, batch_size, epochs, lr, device, save_model, return_threshold_buy, return_threshold_sell, features_only):
    token_data_dir = os.path.join(os.path.dirname(__file__), 'tokenData')
    token_folders = [f for f in os.listdir(token_data_dir) if os.path.isdir(os.path.join(token_data_dir, f))]
    for token in token_folders:
        print(f"\n=== Training on {token} ===")
        if save_model:
            # Save each model with token name prefix
            base, ext = os.path.splitext(save_model)
            save_model_name = f"{token}_{model}{ext if ext else '.pth'}"
        else:
            save_model_name = None
        try:
            train_token(token, model=model, seq_len=seq_len, batch_size=batch_size, epochs=epochs, lr=lr, device=device, save_model=save_model_name, plot=False, classification=True, return_threshold_buy=return_threshold_buy, return_threshold_sell=return_threshold_sell, features_only=features_only)
        except Exception as e:
            print(f"Error training {token}: {e}")
    print("\nAll tokens processed.")

if __name__ == "__main__":
    main()
