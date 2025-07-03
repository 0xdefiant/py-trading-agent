# Trading Algorithm CLI

A modular Python trading algorithm framework for training and evaluating ML models (LSTM/GRU) on historical crypto data.

## Requirements

- Python 3.8+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

Run the ML trading algorithm from the command line using a token folder (e.g., bitcoin, ripple, ethereum):

### Example: Train LSTM on Bitcoin
```sh
python data/ml_trading_algo.py --token bitcoin --model lstm --seq-len 24 --batch-size 64 --epochs 10 --lr 0.001 --device cpu --save-model bitcoin_lstm.pth
```

### Example: Train GRU on Ethereum
```sh
python data/ml_trading_algo.py --token ethereum --model gru --seq-len 24 --batch-size 64 --epochs 10 --lr 0.001 --device cpu --save-model ethereum_gru.pth
```

## CLI Options

| Option                  | Type    | Default      | Description                                                                                   |
|-------------------------|---------|--------------|-----------------------------------------------------------------------------------------------|
| --token TOKEN           | str     | (required)   | Token folder name (e.g., bitcoin, ripple, ethereum)                                           |
| --model [lstm|gru]      | str     | lstm         | Model type to use (lstm or gru)                                                               |
| --seq-len INT           | int     | 24           | Sequence length (number of hours for input window)                                            |
| --batch-size INT        | int     | 64           | Batch size for training                                                                       |
| --epochs INT            | int     | 10           | Number of training epochs                                                                     |
| --lr FLOAT              | float   | 0.001        | Learning rate                                                                                 |
| --device [cpu|cuda]     | str     | cpu          | Device to use (cpu or cuda)                                                                   |
| --save-model PATH       | str     | None         | Path or filename to save trained model weights (e.g., bitcoin_lstm.pth)                       |

## Example Commands

# --- Basic Model Training Examples ---

# LSTM on Bitcoin
```sh
python data/ml_trading_algo.py --token bitcoin --model lstm --seq-len 24 --batch-size 64 --epochs 10 --lr 0.001 --device cpu --save-model bitcoin_lstm.pth
```

# GRU on Ripple
```sh
python data/ml_trading_algo.py --token ripple --model gru --seq-len 24 --batch-size 64 --epochs 10 --lr 0.001 --device cpu --save-model ripple_gru.pth
```

# LSTM on Ethereum (custom sequence length and epochs)
```sh
python data/ml_trading_algo.py --token ethereum --model lstm --seq-len 48 --batch-size 32 --epochs 20 --lr 0.0005 --device cpu --save-model ethereum_lstm.pth
```

# GRU on Uniswap (using GPU if available)
```sh
python data/ml_trading_algo.py --token uniswap --model gru --seq-len 24 --batch-size 64 --epochs 10 --lr 0.001 --device cuda --save-model uniswap_gru.pth
```

# --- Save Model Example ---

# Save trained LSTM model for Aave
```sh
python data/ml_trading_algo.py --token aave --model lstm --save-model aave_lstm.pth
```

# --- Custom Example: All-in-one ---
# (GRU, 48-hour window, 30 epochs, batch size 128, learning rate 0.0005, save model)
```sh
python data/ml_trading_algo.py --token bitcoin --model gru --seq-len 48 --batch-size 128 --epochs 30 --lr 0.0005 --save-model bitcoin_gru.pth
```

## Token Examples

Below are example commands for each token available in `/data/tokenData/`:

### Bitcoin
```sh
python data/ml_trading_algo.py --token bitcoin --model lstm --save-model bitcoin_lstm.pth
```

### Ripple (XRP)
```sh
python data/ml_trading_algo.py --token ripple --model gru --save-model ripple_gru.pth
```

### Ethereum
```sh
python data/ml_trading_algo.py --token ethereum --model lstm --save-model ethereum_lstm.pth
```

### Uniswap
```sh
python data/ml_trading_algo.py --token uniswap --model gru --save-model uniswap_gru.pth
```

### Aave
```sh
python data/ml_trading_algo.py --token aave --model lstm --save-model aave_lstm.pth
```

### Chainlink
```sh
python data/ml_trading_algo.py --token chainlink --model gru --save-model chainlink_gru.pth
```

## Notes
- The script will automatically load and concatenate all CSVs in the specified token folder under `/data/tokenData/<token>/`.
- Model weights are saved to `/data/models/` if only a filename is provided for `--save-model`.
- The script prints training loss per epoch and test MSE after training.
- If matplotlib is installed, a plot of true vs. predicted values is shown after evaluation.
- You can tune hyperparameters (sequence length, batch size, epochs, learning rate) for best results.
- For more options, run:

```bash
python data/ml_trading_algo.py --help
```
