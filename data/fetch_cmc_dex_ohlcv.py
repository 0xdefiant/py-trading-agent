import os
import requests
import pandas as pd
import click
from datetime import datetime, timezone, timedelta
import numpy as np
from technical_indicators import add_features
import json

def get_current_eth_price():
    # Use CoinGecko public API to get current ETH price in USD
    try:
        resp = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd')
        resp.raise_for_status()
        data = resp.json()
        return float(data['ethereum']['usd'])
    except Exception as e:
        print(f"Warning: Could not fetch ETH price from CoinGecko: {e}")
        return None

CMC_DEX_OHLCV_URL = "https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical"

# Load DexPairs.json
DEX_PAIRS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DexPairs.json'))
def load_dex_pairs():
    try:
        with open(DEX_PAIRS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load DexPairs.json: {e}")
        return {}

def default_time_start():
    # 128 hours ago, as unix seconds string
    return str(int((datetime.now(timezone.utc) - timedelta(hours=128)).timestamp()))

def default_time_end():
    # now, as unix seconds string
    return str(int(datetime.now(timezone.utc).timestamp()))

@click.command()
@click.option('--api-key', default=None, help='CoinMarketCap API key (or set CMC_API_KEY env variable)')
@click.option('--contract-address', required=False, help='DEX pair contract address (overrides --dex/--pair)')
@click.option('--dex', required=False, help='DEX/network name as in DexPairs.json (e.g., eth, base, arbitrum, uni, bsc-pancake, optimism)')
@click.option('--pair', required=False, help='Trading pair as in DexPairs.json (e.g., btc/eth, eth/usdc, etc.)')
@click.option('--network-id', default=1, required=False, help='Network ID (e.g., 1 for Ethereum mainnet)')
@click.option('--network-slug', required=False, help='Network slug (e.g., ethereum)')
@click.option('--time-period', default='hourly', type=click.Choice(['hourly', 'daily']), show_default=True, help='Time period (hourly or daily)')
@click.option('--interval',
    default='1h',
    type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '8h', '12h', 'daily', 'weekly', 'monthly']),
    show_default=True,
    help='Interval for sampling time_period. Valid values: 1m, 5m, 15m, 30m, 1h, 4h, 8h, 12h, daily, weekly, monthly.'
)
@click.option('--time-start', default=default_time_start, show_default='128h ago (unix seconds)', help='Start time (unix seconds or ISO8601)')
@click.option('--time-end', default=default_time_end, show_default='now (unix seconds)', help='End time (unix seconds or ISO8601)')
@click.option('--count', default=128, type=int, help='Limit the number of time periods to return (max 500)')
@click.option('--aux', default=None, help='Comma-separated list of supplemental data fields to return')
@click.option('--convert-id', default=None, help='Comma-separated list of currency IDs for conversion')
@click.option('--skip-invalid', default=True, is_flag=True, show_default=True, help='Skip invalid lookups (default: true)')
@click.option('--reverse-order', default=False, is_flag=True, show_default=True, help='Reverse the order of the spot pair (default: false)')
def fetch_dex_ohlcv(api_key, contract_address, dex, pair, network_id, network_slug, time_period, interval, time_start, time_end, count, aux, convert_id, skip_invalid, reverse_order):
    api_key = api_key or os.environ.get('CMC_API_KEY')
    if not api_key:
        raise click.ClickException('CoinMarketCap API key required. Use --api-key or set CMC_API_KEY env variable.')
    # Load contract address from DexPairs.json if not provided
    dex_network_id = None
    if not contract_address and dex and pair:
        dex_pairs = load_dex_pairs()
        dex_info = dex_pairs.get(dex)
        if not dex_info:
            print(f"DEX/network '{dex}' not found in DexPairs.json. Available: {list(dex_pairs.keys())}")
            return
        pairs = dex_info.get('pairs', {})
        contract_address = pairs.get(pair)
        dex_network_id = dex_info.get('network_id')
        if not contract_address:
            print(f"Pair '{pair}' not found for DEX '{dex}'. Available pairs: {list(pairs.keys())}")
            return
        print(f"Using contract address for {dex} {pair}: {contract_address}")
        if not network_id and dex_network_id:
            network_id = dex_network_id
    params = {
        'time_period': time_period,
        'interval': interval,
        'time_start': time_start,
        'time_end': time_end,
        'skip_invalid': str(skip_invalid).lower(),
        'reverse_order': str(reverse_order).lower()
    }
    if contract_address:
        params['contract_address'] = contract_address
    if network_id:
        params['network_id'] = network_id
    if network_slug:
        params['network_slug'] = network_slug
    if count:
        params['count'] = count
    if aux:
        params['aux'] = aux
    if convert_id:
        params['convert_id'] = convert_id
    headers = {'X-CMC_PRO_API_KEY': api_key}
    # --- Fetch normal order ---
    params['reverse_order'] = 'false'
    print(f"Requesting DEX OHLCV with params: {params}")
    resp = requests.get(CMC_DEX_OHLCV_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()['data']
    if not data or not data[0]['quotes']:
        print('No OHLCV data returned.')
        return
    quotes = data[0]['quotes']
    base_symbol = data[0].get('base_asset_symbol', 'BASE')
    quote_symbol = data[0].get('quote_asset_symbol', 'QUOTE')
    symbol = f"{base_symbol}/{quote_symbol}"
    name = data[0].get('name', 'pair')
    network_id = data[0].get('network_id', 'net')
    most_recent_time_close = max(q.get('time_close', '') for q in quotes if 'time_close' in q)
    safe_name = name.replace('/', '_').replace(' ', '_')
    safe_time_close = most_recent_time_close.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')
    output_filename = f"{safe_name}_{network_id}_{safe_time_close}.csv"
    eth_price = get_current_eth_price()
    if eth_price is None or eth_price <= 0:
        print('Could not fetch a valid ETH price. ETH Volume column will be 0.')
    rows = []
    for q in quotes:
        quote = q['quote'][0]
        time_open = q['time_open']
        try:
            dt = datetime.fromisoformat(time_open.replace('Z', '+00:00'))
            unix = int(dt.timestamp())
        except Exception:
            unix = ''
        def fmt(val):
            try:
                return float(val)
            except Exception:
                return 0.0
        price = fmt(quote['close'])
        usd_volume = fmt(quote.get('volume', 0.0))
        base_volume = usd_volume / price if price > 0 else 0.0
        quote_volume = usd_volume
        weth_volume = usd_volume / eth_price if eth_price and eth_price > 0 else 0.0
        rows.append({
            'unix': unix,
            'open': fmt(quote['open']),
            'high': fmt(quote['high']),
            'low': fmt(quote['low']),
            'close': price,
            f'{base_symbol} Volume': base_volume,
            f'{quote_symbol} Volume': quote_volume,
            'WETH Volume': weth_volume
        })
    df = pd.DataFrame(rows)
    # Sort so most recent is at the top
    df = df.sort_values('unix', ascending=False).reset_index(drop=True)
    # --- Add technical indicators ---
    df = add_features(df)
    # Only drop the first max_lookback rows (168 for weekly rolling features)
    max_lookback = 168
    if len(df) > max_lookback:
        df = df.iloc[max_lookback:].reset_index(drop=True)
    # Optionally, drop rows where the target (e.g., 'close', 'return', 'log_return') is still NaN
    df = df[df['close'].notna() & df['return'].notna() & df['log_return'].notna()]
    df = df.reset_index(drop=True)
    # Format all numeric columns (except unix) to 6 decimal places and fill missing with 0.000000
    for col in df.columns:
        if col != 'unix' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "0.000000")
    # Only keep unix, open, high, low, close, <base_symbol> Volume, <quote_symbol> Volume, WETH Volume, and technical indicators
    keep_cols = ['unix', 'open', 'high', 'low', 'close', f'{base_symbol} Volume', f'{quote_symbol} Volume', 'WETH Volume']
    tech_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + tech_cols]
    # Drop duplicate columns, keeping the first occurrence (especially for 'WETH Volume')
    df = df.loc[:, ~df.columns.duplicated()]
    current_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'currentData'))
    os.makedirs(current_data_dir, exist_ok=True)
    output_path = os.path.join(current_data_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    fetch_dex_ohlcv() 