
import pandas as pd
import ta
from time import sleep

# Configuration parameters
tp = 0.012  # Take profit threshold
sl = 0.009  # Stop loss threshold
volume = 10  # Volume for one order
qty = 100  # Amount of concurrent opened positions

# Read historical data from CSV file
def read_historical_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

# Function to get candles for the needed symbol from the historical data
def klines(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df.astype(float)

# Strategy function to generate signals
def rsi_signal(kl):
    rsi = ta.momentum.RSIIndicator(kl['Close']).rsi()
    ema = ta.trend.ema_indicator(kl['Close'], window=200)
    if len(rsi) < 2 or len(ema) < 1:
        return 'none'
    if rsi.iloc[-2] < 30 and rsi.iloc[-1] > 30:
        return 'up'
    if rsi.iloc[-2] > 70 and rsi.iloc[-1] < 70:
        return 'down'
    else:
        return 'none'

# Main loop to process historical data
def main(file_path):
    data = read_historical_data(file_path)
    while True:
        # Simulate live trading by iterating through the historical data
        for current_time, current_data in data.iterrows():
            kl = klines(data[:current_time])
            signal = rsi_signal(kl)

            # Simulate order logic
            if signal == 'up':
                print(f"Time: {current_time} - Found BUY signal. Placing order.")
                # Place buy order logic here
            elif signal == 'down':
                print(f"Time: {current_time} - Found SELL signal. Placing order.")
                # Place sell order logic here

            # Wait for the next data point (simulate real-time delay)
            sleep(1)

        print('Waiting 3 min before next iteration')
        sleep(180)

if __name__ == "__main__":
    file_path = '/home/touf/trade_workshop/BTC_USD.csv'  # Update with the path to your CSV file
    main(file_path)
