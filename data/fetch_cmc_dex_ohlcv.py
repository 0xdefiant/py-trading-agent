import os
import requests
import pandas as pd
import click
from datetime import datetime, timezone

CMC_DEX_OHLCV_URL = "https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical"

@click.command()
@click.option('--api-key', default=None, help='CoinMarketCap API key (or set CMC_API_KEY env variable)')
@click.option('--contract-address', required=False, help='DEX pair contract address')
@click.option('--network-id', required=False, help='Network ID (e.g., 1 for Ethereum mainnet)')
@click.option('--network-slug', required=False, help='Network slug (e.g., ethereum)')
@click.option('--time-period', default='hourly', type=click.Choice(['hourly', 'daily']), show_default=True, help='Time period (hourly or daily)')
@click.option('--interval', default='1h', help='Interval (e.g., 1h, 1d, 1w)')
@click.option('--time-start', required=True, help='Start time (unix seconds or ISO8601)')
@click.option('--time-end', required=True, help='End time (unix seconds or ISO8601)')
@click.option('--count', default=None, type=int, help='Limit the number of time periods to return (max 500)')
@click.option('--aux', default=None, help='Comma-separated list of supplemental data fields to return')
@click.option('--convert-id', default=None, help='Comma-separated list of currency IDs for conversion')
@click.option('--skip-invalid', default=True, is_flag=True, show_default=True, help='Skip invalid lookups (default: true)')
@click.option('--reverse-order', default=False, is_flag=True, show_default=True, help='Reverse the order of the spot pair (default: false)')
@click.option('--output', default='dex_ohlcv.csv', help='Output CSV file')
def fetch_dex_ohlcv(api_key, contract_address, network_id, network_slug, time_period, interval, time_start, time_end, count, aux, convert_id, skip_invalid, reverse_order, output):
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
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")

if __name__ == "__main__":
    fetch_dex_ohlcv() 