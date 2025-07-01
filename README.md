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

| Option              | Type      | Default      | Description                                                        |
|---------------------|-----------|--------------|--------------------------------------------------------------------|
| --csv PATH          | Path      | (required*)  | Path to a single market data CSV file                              |
| --dir PATH          | Path      | (required*)  | Path to a directory of CSVs for multi-token/pair analysis          |
| --strategy STR      | Choice    | sma_crossover| Trading strategy: sma_crossover, ema_crossover, rsi, macd, bollinger |
| --fast INT          | int       | 10           | Fast window for SMA/EMA (if applicable)                            |
| --slow INT          | int       | 30           | Slow window for SMA/EMA (if applicable)                            |
| --rsi-window INT    | int       | 14           | RSI window (if applicable)                                         |
| --rsi-buy INT       | int       | 30           | RSI buy threshold                                                  |
| --rsi-sell INT      | int       | 70           | RSI sell threshold                                                 |
| --bb-window INT     | int       | 20           | Bollinger Bands window                                             |
| --bb-num-std FLOAT  | float     | 2.0          | Bollinger Bands number of std deviations                           |
| --stop-loss FLOAT   | float     | 0.05         | Stop loss threshold (fraction, e.g. 0.03 = 3%)                     |
| --take-profit FLOAT | float     | 0.1          | Take profit threshold (fraction, e.g. 0.07 = 7%)                   |
| --position-size FLOAT| float    | 1.0          | Fraction of capital to use per trade (0-1)                         |
| --plot / --no-plot  | flag      | --plot       | Show plots of results                                              |
| --initial-balance FLOAT| float  | 10000.0      | Initial balance for backtest                                       |

*Either --csv or --dir is required. If both are provided, --dir takes priority.

## Example Commands

**SMA Crossover:**
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy sma_crossover --fast 10 --slow 30 --plot
```

**EMA Crossover on all Ripple pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/ripple --strategy ema_crossover --fast 10 --slow 30 --stop-loss 0.03 --take-profit 0.07 --position-size 0.5 --plot
```

**RSI Strategy:**
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy rsi --rsi-window 14 --rsi-buy 30 --rsi-sell 70 --plot
```

**MACD on all Ethereum pairs:**
```sh
python data/trading-algo.py --dir data/tokenData/ethereum --strategy macd --plot
```

**Bollinger Bands:**
```sh
python data/trading-algo.py --csv data/tokenData/bitcoin/Bitstamp_BTCUSD_1h.csv --strategy bollinger --bb-window 20 --bb-num-std 2 --plot
```

## Notes
- Plots will display the equity curve and buy/sell signals.
- Results are printed as summary statistics (final balance, total return, Sharpe ratio, max drawdown).
- You can extend the code to add more strategies or analytics.
- All token data is now under `/data/tokenData/` with a subfolder for each token (e.g., `/data/tokenData/bitcoin/`, `/data/tokenData/ripple/`, etc.). # py-trading-agent
