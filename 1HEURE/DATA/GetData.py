import requests
import pandas as pd
import time

def get_binance_klines(symbol, interval, start_time, end_time):
    url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    while start_time < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move to the next timestamp after the last one
        time.sleep(1)  # To prevent hitting the rate limit
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    df['Adj Close'] = df['Close']  # Assuming Adjusted Close is the same as Close
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Example usage
start_time = int(pd.Timestamp('2024-05-01').timestamp() * 1000)
end_time = int(pd.Timestamp('2024-05-27').timestamp() * 1000)
df = get_binance_klines('ETHUSDT', '1h', start_time, end_time)
save_to_csv(df, 'ETH_USD.csv')
