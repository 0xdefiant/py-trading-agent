import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from typing import List, Dict, Any
import glob

# =========================
# 1. Data Loading & Cleaning
# =========================
def load_market_data(csv_path: str) -> pd.DataFrame:
    """
    Loads and cleans historical market data from a CSV file.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Cleaned DataFrame with parsed columns.
    """
    df = pd.read_csv(csv_path)
    # Parse date, ensure numeric columns, sort by date
    df['date'] = pd.to_datetime(df['date'])
    # Use only columns present in the file
    numeric_cols = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    # Note: Some Gemini files may have different volume columns (e.g., Volume ETH, Volume BTC, Volume USD)
    return df

def load_market_data_from_dir(directory: str) -> pd.DataFrame:
    """
    Loads and concatenates all CSVs in a directory (recursively) for multi-token/pair analysis.
    Args:
        directory (str): Path to the directory containing CSV files.
    Returns:
        pd.DataFrame: Combined DataFrame of all market data.
    """
    all_files = glob.glob(os.path.join(directory, '**', '*.csv'), recursive=True)
    df_list = []
    for file in all_files:
        df = load_market_data(file)
        df['source_file'] = os.path.basename(file)
        df_list.append(df)
    if not df_list:
        raise ValueError(f"No CSV files found in directory: {directory}")
    combined = pd.concat(df_list, ignore_index=True)
    return combined

# =========================
# 2. Technical Indicators
# =========================
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()

def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD and Signal line."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line})

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    sma_ = sma(series, window)
    std = series.rolling(window=window).std()
    upper = sma_ + num_std * std
    lower = sma_ - num_std * std
    return pd.DataFrame({'upper': upper, 'lower': lower, 'sma': sma_})

# =========================
# 3. Signal Generation
# =========================
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates buy/sell signals based on indicator logic.
    Args:
        df (pd.DataFrame): DataFrame with price and indicator columns.
    Returns:
        pd.DataFrame: DataFrame with signal column (1=buy, -1=sell, 0=hold).
    """
    signals = pd.Series(0, index=df.index)
    # Example: SMA crossover
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        signals[df['sma_fast'] > df['sma_slow']] = 1
        signals[df['sma_fast'] < df['sma_slow']] = -1
    # Add more logic as needed
    df['signal'] = signals
    return df

# =========================
# 4. Backtesting & Evaluation
# =========================
def backtest(df: pd.DataFrame, initial_balance: float = 10000.0) -> Dict[str, Any]:
    """
    Simulates trades and evaluates performance metrics.
    Args:
        df (pd.DataFrame): DataFrame with 'close' and 'signal' columns.
        initial_balance (float): Starting capital.
    Returns:
        Dict[str, Any]: Performance metrics (total return, Sharpe, max drawdown, etc).
    """
    balance = initial_balance
    position = 0  # 1 for long, 0 for out
    entry_price = 0
    equity_curve = []
    for i, row in df.iterrows():
        if row['signal'] == 1 and position == 0:
            position = 1
            entry_price = row['close']
        elif row['signal'] == -1 and position == 1:
            balance *= row['close'] / entry_price
            position = 0
        equity_curve.append(balance if position == 0 else balance * (row['close'] / entry_price))
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    return {
        'final_balance': equity_curve[-1],
        'total_return': (equity_curve[-1] / initial_balance) - 1,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown.max() if len(max_drawdown) > 0 else 0
    }

# =========================
# 5. Risk Management
# =========================
def apply_risk_management(df: pd.DataFrame, stop_loss: float = 0.05, take_profit: float = 0.1, position_size: float = 1.0) -> pd.DataFrame:
    """
    Applies stop loss, take profit, and position sizing to signals.
    Args:
        df (pd.DataFrame): DataFrame with 'close' and 'signal' columns.
        stop_loss (float): Stop loss threshold (fraction).
        take_profit (float): Take profit threshold (fraction).
        position_size (float): Fraction of capital to use per trade (0-1).
    Returns:
        pd.DataFrame: DataFrame with adjusted signals and position sizing.
    """
    df = df.copy()
    df['position_size'] = position_size
    # Implement stop loss/take profit logic in backtest for accuracy
    return df

# =========================
# 6. Main & Extensibility
# =========================
# Remove the old main() function and direct file loading.
# if __name__ == "__main__":
#     main()

# Instead, the CLI is now the only entry point:
# Usage:
#   python trading-algo.py --csv path/to/file.csv --strategy macd --plot
#   python trading-algo.py --dir path/to/folder --strategy macd --plot

# =========================
# 7. Plotting & Visualization
# =========================
def plot_equity_curve(equity_curve: List[float], title: str = 'Equity Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title(title)
    plt.xlabel('Trade/Time Step')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_price_with_signals(df: pd.DataFrame, title: str = 'Price & Signals'):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['close'], label='Close Price')
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    plt.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='g', label='Buy', alpha=0.8)
    plt.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='r', label='Sell', alpha=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# 8. Advanced Features: Strategy Selection & Parameterization
# =========================
def run_strategy(df: pd.DataFrame, strategy: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Runs the selected strategy with given parameters.
    Args:
        df (pd.DataFrame): Market data.
        strategy (str): Strategy name.
        params (dict): Strategy parameters.
    Returns:
        pd.DataFrame: DataFrame with signals.
    """
    if strategy == 'sma_crossover':
        df['sma_fast'] = sma(df['close'], params.get('fast', 10))
        df['sma_slow'] = sma(df['close'], params.get('slow', 30))
        df = generate_signals(df)
    elif strategy == 'ema_crossover':
        df['ema_fast'] = ema(df['close'], params.get('fast', 10))
        df['ema_slow'] = ema(df['close'], params.get('slow', 30))
        signals = pd.Series(0, index=df.index)
        signals[df['ema_fast'] > df['ema_slow']] = 1
        signals[df['ema_fast'] < df['ema_slow']] = -1
        df['signal'] = signals
    elif strategy == 'rsi':
        df['rsi'] = rsi(df['close'], params.get('window', 14))
        df['signal'] = 0
        df.loc[df['rsi'] < params.get('rsi_buy', 30), 'signal'] = 1
        df.loc[df['rsi'] > params.get('rsi_sell', 70), 'signal'] = -1
    elif strategy == 'macd':
        macd_df = macd(df['close'])
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['signal'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1
    elif strategy == 'bollinger':
        bb_df = bollinger_bands(df['close'], params.get('window', 20), params.get('num_std', 2.0))
        df['bb_upper'] = bb_df['upper']
        df['bb_lower'] = bb_df['lower']
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return df

# =========================
# 9. CLI Entry Point (Enhanced for Multi-Token)
# =========================
@click.command()
@click.option('--csv', 'csv_path', required=False, type=click.Path(exists=True), help='Path to market data CSV.')
@click.option('--dir', 'csv_dir', required=False, type=click.Path(exists=True, file_okay=False), help='Path to directory of market data CSVs (for multi-token/pair analysis).')
@click.option('--strategy', default='sma_crossover', type=click.Choice(['sma_crossover', 'ema_crossover', 'rsi', 'macd', 'bollinger']), help='Trading strategy to use.')
@click.option('--fast', default=10, type=int, help='Fast window for SMA/EMA (if applicable).')
@click.option('--slow', default=30, type=int, help='Slow window for SMA/EMA (if applicable).')
@click.option('--rsi-window', default=14, type=int, help='RSI window (if applicable).')
@click.option('--rsi-buy', default=30, type=int, help='RSI buy threshold.')
@click.option('--rsi-sell', default=70, type=int, help='RSI sell threshold.')
@click.option('--bb-window', default=20, type=int, help='Bollinger Bands window.')
@click.option('--bb-num-std', default=2.0, type=float, help='Bollinger Bands number of std deviations.')
@click.option('--stop-loss', default=0.05, type=float, help='Stop loss threshold (fraction).')
@click.option('--take-profit', default=0.1, type=float, help='Take profit threshold (fraction).')
@click.option('--position-size', default=1.0, type=float, help='Fraction of capital to use per trade (0-1).')
@click.option('--plot/--no-plot', default=True, help='Show plots of results.')
@click.option('--initial-balance', default=10000.0, type=float, help='Initial balance for backtest.')
def cli(csv_path, csv_dir, strategy, fast, slow, rsi_window, rsi_buy, rsi_sell, bb_window, bb_num_std, stop_loss, take_profit, position_size, plot, initial_balance):
    """Run trading algorithm backtest from the command line."""
    # Prefer directory if both are provided
    if csv_dir:
        df = load_market_data_from_dir(csv_dir)
    elif csv_path:
        df = load_market_data(csv_path)
    else:
        raise click.UsageError('You must provide either --csv or --dir.')
    params = {'fast': fast, 'slow': slow, 'window': rsi_window, 'rsi_buy': rsi_buy, 'rsi_sell': rsi_sell, 'bb_window': bb_window, 'num_std': bb_num_std}
    df = run_strategy(df, strategy, params)
    df = apply_risk_management(df, stop_loss, take_profit, position_size)
    results = backtest_with_risk(df, initial_balance, stop_loss, take_profit, position_size)
    summary = {k: v for k, v in results.items() if k != 'equity_curve'}
    click.echo(f'Backtest Results: {summary}')
    if plot:
        plot_equity_curve(results["equity_curve"])
        plot_price_with_signals(df)

# =========================
# 4. Backtesting & Evaluation (Enhanced for Risk)
# =========================
def backtest_with_risk(df: pd.DataFrame, initial_balance: float = 10000.0, stop_loss: float = 0.05, take_profit: float = 0.1, position_size: float = 1.0) -> Dict[str, Any]:
    """
    Simulates trades and evaluates performance metrics, including stop loss, take profit, and position sizing.
    Args:
        df (pd.DataFrame): DataFrame with 'close' and 'signal' columns.
        initial_balance (float): Starting capital.
        stop_loss (float): Stop loss threshold (fraction).
        take_profit (float): Take profit threshold (fraction).
        position_size (float): Fraction of capital to use per trade (0-1).
    Returns:
        Dict[str, Any]: Performance metrics and equity curve.
    """
    balance = initial_balance
    position = 0  # 1 for long, 0 for out
    entry_price = 0
    equity_curve = []
    for i, row in df.iterrows():
        if row['signal'] == 1 and position == 0:
            position = 1
            entry_price = row['close']
            trade_size = balance * position_size
        elif row['signal'] == -1 and position == 1:
            # Close position
            pnl = (row['close'] - entry_price) / entry_price
            balance += trade_size * pnl
            position = 0
        elif position == 1:
            # Check stop loss/take profit
            pnl = (row['close'] - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                balance += trade_size * pnl
                position = 0
        equity_curve.append(balance if position == 0 else balance + trade_size * ((row['close'] - entry_price) / entry_price))
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    max_drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    return {
        'final_balance': equity_curve[-1],
        'total_return': (equity_curve[-1] / initial_balance) - 1,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown.max() if len(max_drawdown) > 0 else 0,
        'equity_curve': equity_curve
    }

if __name__ == "__main__":
    cli()
