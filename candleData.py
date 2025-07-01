import os
import time
from datetime import datetime, timedelta, UTC
from coinbase.rest import RESTClient
from json import dumps

# Helper to get UNIX timestamp
now = datetime.now(UTC)
def unix(dt):
    return int(dt.timestamp())

def fetch_candles(client, product_id, granularity, num_candles):
    end = now
    if granularity == 'ONE_MINUTE':
        delta = timedelta(minutes=num_candles)
    elif granularity == 'FIVE_MINUTE':
        delta = timedelta(minutes=5*num_candles)
    elif granularity == 'ONE_HOUR':
        delta = timedelta(hours=num_candles)
    elif granularity == 'ONE_DAY':
        delta = timedelta(days=num_candles)
    else:
        raise ValueError('Unsupported granularity')
    start = end - delta
    candles = client.get_public_candles(
        product_id=product_id,
        start=str(unix(start)),
        end=str(unix(end)),
        granularity=granularity,
        limit=num_candles
    )
    return candles.to_dict()['candles']

def analyze_candles(candles):
    closes = [float(c['close']) for c in candles]
    if not closes:
        return None
    avg_close = sum(closes) / len(closes)
    last_close = closes[-1]
    first_close = closes[0]
    trend = (last_close - first_close) / first_close * 100 if first_close != 0 else 0
    is_good_buy = trend > 0 and last_close > avg_close
    return {
        'avg_close': avg_close,
        'last_close': last_close,
        'trend_percent': trend,
        'is_good_buy': is_good_buy
    }

def main():
    api_key = os.environ["COINBASE_API_KEY"]
    api_secret = os.environ["COINBASE_API_SECRET"]
    client = RESTClient(api_key=api_key, api_secret=api_secret)
    product_id = "BTC-USD"  # Change as needed
    intervals = [
        ("ONE_MINUTE", 15),
        ("FIVE_MINUTE", 15),
        ("ONE_HOUR", 10),
    ]
    print(f"Analyzing {product_id}...")
    for granularity, num in intervals:
        print(f"\nInterval: {granularity}, Candles: {num}")
        candles = fetch_candles(client, product_id, granularity, num)
        print(f"Fetched {len(candles)} candles.")
        # Print first 3 and last 3 candles for inspection
        if candles:
            print("  First 3 candles:")
            for c in candles[:3]:
                print(f"    {c}")
            if len(candles) > 6:
                print("  ...")
            print("  Last 3 candles:")
            for c in candles[-3:]:
                print(f"    {c}")
        analysis = analyze_candles(candles)
        if analysis:
            print(f"  Avg Close: {analysis['avg_close']:.2f}")
            print(f"  Last Close: {analysis['last_close']:.2f}")
            print(f"  Trend: {analysis['trend_percent']:.2f}%")
            print(f"  Good Buy? {'YES' if analysis['is_good_buy'] else 'NO'}")
        else:
            print("  No candle data available.")

if __name__ == "__main__":
    main() 