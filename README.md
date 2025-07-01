# Trading Algorithm CLI

A modular Python trading algorithm framework for backtesting strategies on historical crypto data.

## Requirements

- Python 3.8+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```

## Usage

Run the trading algorithm from the command line using either a single CSV file or a directory of CSVs (for multi-token/pair analysis):

### Single File Example
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy macd --plot
```

### Directory Example (all pairs for a token)
```sh
python data/trading-algo.py --dir data/tokenData/ripple --strategy ema_crossover --fast 10 --slow 30 --stop-loss 0.03 --take-profit 0.07 --position-size 0.5 --plot
```

## CLI Options

| Option                  | Type    | Default      | Description                                                                                   |
|-------------------------|---------|--------------|-----------------------------------------------------------------------------------------------|
| --csv PATH              | Path    | (required*)  | Path to a single market data CSV file                                                         |
| --dir PATH              | Path    | (required*)  | Path to a directory of CSVs for multi-token/pair analysis                                     |
| --strategy STR          | Choice  | sma_crossover| Trading strategy: sma_crossover, ema_crossover, rsi, macd, bollinger                          |
| --fast INT              | int     | 10           | Fast window for SMA/EMA (used in crossover strategies)                                        |
| --slow INT              | int     | 30           | Slow window for SMA/EMA (used in crossover strategies)                                        |
| --rsi-window INT        | int     | 14           | RSI window (number of periods for RSI calculation)                                            |
| --rsi-buy INT           | int     | 30           | RSI value below which to generate a buy signal                                                |
| --rsi-sell INT          | int     | 70           | RSI value above which to generate a sell signal                                               |
| --bb-window INT         | int     | 20           | Bollinger Bands window (number of periods for SMA/STD)                                        |
| --bb-num-std FLOAT      | float   | 2.0          | Number of standard deviations for Bollinger Bands                                             |
| --stop-loss FLOAT       | float   | 0.05         | Stop loss threshold (fraction, e.g. 0.03 = 3% loss triggers exit)                             |
| --take-profit FLOAT     | float   | 0.1          | Take profit threshold (fraction, e.g. 0.07 = 7% gain triggers exit)                           |
| --position-size FLOAT   | float   | 1.0          | Fraction of capital to use per trade (0-1, e.g. 0.5 = 50% of capital per trade)               |
| --plot / --no-plot      | flag    | --plot       | Show plots of results (equity curve, buy/sell signals)                                        |
| --initial-balance FLOAT | float   | 10000.0      | Initial balance for backtest simulation                                                       |

*Either --csv or --dir is required. If both are provided, --dir takes priority.

### Advanced Option Details

- **--stop-loss FLOAT**: Sets the maximum loss per trade as a fraction of entry price. For example, `--stop-loss 0.03` means a trade will be closed if it loses 3% or more.
- **--take-profit FLOAT**: Sets the profit target per trade as a fraction of entry price. For example, `--take-profit 0.07` means a trade will be closed if it gains 7% or more.
- **--position-size FLOAT**: Controls what fraction of your simulated capital is used for each trade. `--position-size 1.0` means all-in, `0.5` means half your capital per trade. Useful for risk management and portfolio simulation.
- **--fast / --slow**: Used for moving average crossovers (SMA/EMA). `--fast` is the short window, `--slow` is the long window.
- **--rsi-window / --rsi-buy / --rsi-sell**: Used for RSI-based strategies. Adjust these to tune sensitivity and thresholds for overbought/oversold signals.
- **--bb-window / --bb-num-std**: Used for Bollinger Bands. Adjust window and number of standard deviations to control band width and signal frequency.
- **--plot / --no-plot**: By default, plots are shown. Use `--no-plot` to suppress plots (useful for batch runs or automation).
- **--initial-balance**: Set your starting simulated capital for the backtest.

You can combine these options to fine-tune your strategy and risk management for any token or trading pair.

## Example Commands

# --- Basic Strategy Examples ---

# SMA Crossover on Bitcoin (single file)
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy sma_crossover --fast 10 --slow 30 --plot
```

# EMA Crossover on all Ripple pairs (directory)
```sh
python data/trading-algo.py --dir data/tokenData/ripple --strategy ema_crossover --fast 12 --slow 26 --plot
```

# RSI Strategy on Bitcoin (custom thresholds)
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy rsi --rsi-window 14 --rsi-buy 25 --rsi-sell 75 --plot
```

# MACD on all Ethereum pairs
```sh
python data/trading-algo.py --dir data/tokenData/ethereum --strategy macd --plot
```

# Bollinger Bands on Uniswap (custom window and std)
```sh
python data/trading-algo.py --csv data/tokenData/uniswap/Bitstamp_UNIUSD_1h.csv --strategy bollinger --bb-window 15 --bb-num-std 2.5 --plot
```

# --- Advanced Risk Management Examples ---

# SMA Crossover with stop-loss and take-profit
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy sma_crossover --fast 10 --slow 30 --stop-loss 0.02 --take-profit 0.05 --plot
```

# EMA Crossover with position sizing (50% capital per trade)
```sh
python data/trading-algo.py --dir data/tokenData/chainlink --strategy ema_crossover --fast 8 --slow 21 --position-size 0.5 --plot
```

# MACD with custom initial balance
```sh
python data/trading-algo.py --csv data/tokenData/aave/Bitstamp_AAVEBTC_1h.csv --strategy macd --initial-balance 50000 --plot
```

# --- Batch/Automation and No-Plot Examples ---

# Run RSI on all tokens in a folder, no plots (for automation)
```sh
python data/trading-algo.py --dir data/tokenData/ethereum --strategy rsi --no-plot
```

# Run Bollinger Bands on all Uniswap pairs, no plots, custom risk
```sh
python data/trading-algo.py --dir data/tokenData/uniswap --strategy bollinger --bb-window 20 --bb-num-std 2 --stop-loss 0.03 --take-profit 0.08 --position-size 0.3 --no-plot
```

# --- Multi-Token Batch Example (run for each token folder) ---
# (You can run this command for each token folder to backtest all pairs for that token)
```sh
python data/trading-algo.py --dir data/tokenData/bitcoin --strategy macd --no-plot
python data/trading-algo.py --dir data/tokenData/ripple --strategy macd --no-plot
python data/trading-algo.py --dir data/tokenData/ethereum --strategy macd --no-plot
python data/trading-algo.py --dir data/tokenData/uniswap --strategy macd --no-plot
python data/trading-algo.py --dir data/tokenData/aave --strategy macd --no-plot
python data/trading-algo.py --dir data/tokenData/chainlink --strategy macd --no-plot
```

# --- Custom Example: All-in-one ---
# (EMA crossover, 30% capital per trade, stop-loss 4%, take-profit 10%, no plots, $25,000 initial balance)
```sh
python data/trading-algo.py --dir data/tokenData/bitcoin --strategy ema_crossover --fast 10 --slow 30 --stop-loss 0.04 --take-profit 0.1 --position-size 0.3 --initial-balance 25000 --no-plot
```

## Token Examples

Below are example commands for each token available in `/data/tokenData/`:

### Bitcoin
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy macd --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/bitcoin --strategy ema_crossover --plot
```

### Ripple (XRP)
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/ripple/Bitstamp_XRPUSD_1h.csv --strategy rsi --rsi-window 14 --rsi-buy 30 --rsi-sell 70 --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/ripple --strategy bollinger --bb-window 20 --bb-num-std 2 --plot
```

### Ethereum
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/ethereum/Gemini_ETHUSD_1h.csv --strategy macd --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/ethereum --strategy ema_crossover --plot
```

### Uniswap
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/uniswap/Bitstamp_UNIUSD_1h.csv --strategy sma_crossover --fast 10 --slow 30 --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/uniswap --strategy rsi --plot
```

### Aave
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/aave/Bitstamp_AAVEBTC_1h.csv --strategy macd --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/aave --strategy bollinger --plot
```

### Chainlink
**Single File:**
```sh
python data/trading-algo.py --csv data/tokenData/chainlink/Bitstamp_LINKBTC_1h.csv --strategy ema_crossover --plot
```
**All Pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/chainlink --strategy sma_crossover --plot
```

## Notes
- Plots will display the equity curve and buy/sell signals.
- Results are printed as summary statistics (final balance, total return, Sharpe ratio, max drawdown).
- You can extend the code to add more strategies or analytics.
- All token data is now under `/data/tokenData/` with a subfolder for each token (e.g., `/data/tokenData/bitcoin/`, `/data/tokenData/ripple/`, etc.). # py-trading-agent
