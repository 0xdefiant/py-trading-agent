import os
import glob
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

def add_features(df):
    # Basic returns
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']).diff()
    # 24-period simple moving average
    df['SMA_24'] = df['close'].rolling(window=24).mean()
    # 24-period rolling volatility
    df['volatility_24'] = df['return'].rolling(window=24).std()
    return df

def process_token_folder(token_folder):
    # Only include raw CSVs, skip any _features.csv files
    csv_files = [f for f in glob.glob(os.path.join(token_folder, '*.csv')) if not f.endswith('_features.csv')]
    if not csv_files:
        print(f"No CSV files found in {token_folder}")
        return
    # Separate hourly and daily files
    hourly_files = [f for f in csv_files if f.rstrip('.csv').endswith('1h')]
    daily_files = [f for f in csv_files if f.rstrip('.csv').endswith('d')]
    for freq, files, suffix in [("hourly", hourly_files, "_features_1h"), ("daily", daily_files, "_features_d")]:
        if not files:
            continue
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        if not dfs:
            print(f"No valid {freq} data in {token_folder}")
            continue
        df = pd.concat(dfs, ignore_index=True)
        # --- Merge all USD stablecoin columns into a single 'Volume USD' column ---
        usd_cols = [col for col in df.columns if any(stable in col.lower() for stable in ['usd', 'usdt', 'usdc', 'dai'])]
        if usd_cols:
            df['Volume USD'] = df[usd_cols].sum(axis=1)
            # Drop all original stablecoin columns except for the new 'Volume USD'
            for col in usd_cols:
                if col != 'Volume USD':
                    df = df.drop(columns=col)
        # Sort by date if present, but do not drop duplicates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        # Add features
        df = add_features(df)
        # Only drop the first max_lookback rows (168 for weekly rolling features)
        max_lookback = 168
        df = df.iloc[max_lookback:].reset_index(drop=True)
        # Optionally, drop rows where the target (e.g., 'close', 'return', 'log_return') is still NaN
        df = df[df['close'].notna() & df['return'].notna() & df['log_return'].notna()]
        df = df.reset_index(drop=True)
        # Save
        token_name = os.path.basename(token_folder)
        out_path = os.path.join(token_folder, f'{token_name}{suffix}.csv')
        # Explicitly remove the output file if it exists to ensure overwrite
        if os.path.exists(out_path):
            os.remove(out_path)
        df.to_csv(out_path, index=False)
        print(f"Saved enriched {freq} data for {token_name} to {out_path}")

def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'tokenData')
    for token in os.listdir(base_dir):
        token_folder = os.path.join(base_dir, token)
        if os.path.isdir(token_folder):
            print(f"Processing {token_folder}...")
            process_token_folder(token_folder)

if __name__ == '__main__':
    main()
