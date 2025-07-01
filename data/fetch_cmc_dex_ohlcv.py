import os
import requests
import pandas as pd
import click
from datetime import datetime, timezone, timedelta

CMC_DEX_OHLCV_URL = "https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical"

def default_time_start():
    # 24 hours ago, as unix seconds string
    return str(int((datetime.now(timezone.utc) - timedelta(days=2)).timestamp()))

def default_time_end():
    # now, as unix seconds string
    return str(int(datetime.now(timezone.utc).timestamp()))

@click.command()
@click.option('--api-key', default=None, help='CoinMarketCap API key (or set CMC_API_KEY env variable)')
@click.option('--contract-address', required=False, help='DEX pair contract address')
@click.option('--network-id', required=False, help='Network ID (e.g., 1 for Ethereum mainnet)')
@click.option('--network-slug', required=False, help='Network slug (e.g., ethereum)')
@click.option('--time-period', default='hourly', type=click.Choice(['hourly', 'daily']), show_default=True, help='Time period (hourly or daily)')
@click.option('--interval',
    default='1h',
    type=click.Choice(['1m', '5m', '15m', '30m', '1h', '4h', '8h', '12h', 'daily', 'weekly', 'monthly']),
    show_default=True,
    help='Interval for sampling time_period. Valid values: 1m, 5m, 15m, 30m, 1h, 4h, 8h, 12h, daily, weekly, monthly.'
)
@click.option('--time-start', default=default_time_start, show_default='24h ago (unix seconds)', help='Start time (unix seconds or ISO8601)')
@click.option('--time-end', default=default_time_end, show_default='now (unix seconds)', help='End time (unix seconds or ISO8601)')
@click.option('--count', default=100, type=int, help='Limit the number of time periods to return (max 500)')
@click.option('--aux', default=None, help='Comma-separated list of supplemental data fields to return')
@click.option('--convert-id', default=None, help='Comma-separated list of currency IDs for conversion')
@click.option('--skip-invalid', default=True, is_flag=True, show_default=True, help='Skip invalid lookups (default: true)')
@click.option('--reverse-order', default=False, is_flag=True, show_default=True, help='Reverse the order of the spot pair (default: false)')
def fetch_dex_ohlcv(api_key, contract_address, network_id, network_slug, time_period, interval, time_start, time_end, count, aux, convert_id, skip_invalid, reverse_order):
    api_key = api_key or os.environ.get('CMC_API_KEY')
    if not api_key:
        raise click.ClickException('CoinMarketCap API key required. Use --api-key or set CMC_API_KEY env variable.')
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
    # --- New: Determine output filename from API metadata ---
    name = data[0].get('name', 'pair')
    network_id = data[0].get('network_id', 'net')
    # Find the most recent time_close in quotes
    most_recent_time_close = max(q.get('time_close', '') for q in quotes if 'time_close' in q)
    # Clean up name for filename (remove slashes and spaces)
    safe_name = name.replace('/', '_').replace(' ', '_')
    # Clean up time_close for filename (remove colons and dashes)
    safe_time_close = most_recent_time_close.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')
    output_filename = f"{safe_name}_{network_id}_{safe_time_close}.csv"
    rows = []
    for q in quotes:
        quote = q['quote'][0]  # Only one quote per time period
        # Convert ISO8601 to unix seconds and to 'YYYY-MM-DD HH:MM:SS'
        time_open = q['time_open']
        try:
            dt = datetime.fromisoformat(time_open.replace('Z', '+00:00'))
            unix = int(dt.timestamp())
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            unix = ''
            date_str = time_open
        def fmt(val):
            try:
                return f"{float(val):.6f}"
            except Exception:
                return val
        rows.append({
            'unix': unix,
            'date': date_str,
            'symbol': symbol,
            'open': fmt(quote['open']),
            'high': fmt(quote['high']),
            'low': fmt(quote['low']),
            'close': fmt(quote['close']),
            f'Volume {base_symbol}': fmt(quote.get('volume', 0.0)),
            f'Volume {quote_symbol}': fmt(0.0)  # Not available from API
        })
    # Ensure column order
    columns = ['unix', 'date', 'symbol', 'open', 'high', 'low', 'close', f'Volume {base_symbol}', f'Volume {quote_symbol}']
    df = pd.DataFrame(rows, columns=columns)
    current_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'currentData'))
    os.makedirs(current_data_dir, exist_ok=True)
    output_path = os.path.join(current_data_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    fetch_dex_ohlcv() 