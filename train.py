import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from binance.client import Client
import time
import keys  # Import your API keys from keys.py

class RealTimeTradeBot:
    def __init__(self):
        self._stacks = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 10.0}
        self._initial_stacks = self._stacks.copy()
        self.transaction_log = []
        self.model = None  # Will be loaded later
        self.stop_loss = 0.95  # Stop loss at 5% loss
        self.take_profit = 1.05  # Take profit at 5% gain
        self.api_key = keys.api_key
        self.api_secret = keys.api_secret
        self.client = Client(self.api_key, self.api_secret)
        self.results = []  # Store results for visualization

    def main_engine(self):
        self.load_model('trading_model.pkl')
        self.trade_real_time()

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def fetch_market_data(self, symbol):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)
            ticker = self.client.get_ticker(symbol=symbol)
            data = {
                'symbol': symbol,
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'last_price': float(ticker['lastPrice']),
                'high_price': float(ticker['highPrice']),
                'low_price': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'timestamp': pd.Timestamp.now()
            }
            print(data)  # Print the data to inspect its structure
            return data
        except Exception as e:
            print(f"Exception fetching market data: {e}")
            return None

    def preprocess_data(self, market_data):
        df = pd.DataFrame([market_data])
        df['price_change'] = df['last_price'] - df['bid_price']
        df['high_low_diff'] = df['high_price'] - df['low_price']
        return df[['price_change', 'high_low_diff', 'volume']]

    def trade_real_time(self):
        symbol_map = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT'
        }

        while True:
            for asset in symbol_map:
                market_data = self.fetch_market_data(symbol_map[asset])
                if market_data:
                    features = self.preprocess_data(market_data)
                    prediction = self.model.predict(features)[0]

                    if prediction == 1:  # Buy signal
                        self.buy(asset, market_data['last_price'])
                    else:  # Sell signal
                        self.sell(asset, market_data['last_price'])

            # Sleep for a certain period before fetching the next data point
            time.sleep(60)  # Fetch data every 60 seconds

    def buy(self, asset, price):
        if asset == 'ETH':
            buy_amount = self._stacks['USDT'] * 0.1
            if buy_amount > 0:
                self._stacks['USDT'] -= buy_amount
                self._stacks['ETH'] += buy_amount / price
                self.transaction_log.append(f"buy {asset} {buy_amount / price} at {price}")
                self.results.append({'action': 'buy', 'pair': asset, 'amount': buy_amount / price, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"buy {asset} {buy_amount / price} at {price}")
        elif asset == 'BTC':
            buy_amount = self._stacks['USDT'] * 0.1
            if buy_amount > 0:
                self._stacks['USDT'] -= buy_amount
                self._stacks['BTC'] += buy_amount / price
                self.transaction_log.append(f"buy {asset} {buy_amount / price} at {price}")
                self.results.append({'action': 'buy', 'pair': asset, 'amount': buy_amount / price, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"buy {asset} {buy_amount / price} at {price}")

    def sell(self, asset, price):
        if asset == 'ETH':
            sell_amount = self._stacks['ETH'] * 0.25
            if sell_amount > 0:
                self._stacks['USDT'] += sell_amount * price
                self._stacks['ETH'] -= sell_amount
                self.transaction_log.append(f"sell {asset} {sell_amount} at {price}")
                self.results.append({'action': 'sell', 'pair': asset, 'amount': sell_amount, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {asset} {sell_amount} at {price}")
        elif asset == 'BTC':
            sell_amount = self._stacks['BTC'] * 0.25
            if sell_amount > 0:
                self._stacks['USDT'] += sell_amount * price
                self._stacks['BTC'] -= sell_amount
                self.transaction_log.append(f"sell {asset} {sell_amount} at {price}")
                self.results.append({'action': 'sell', 'pair': asset, 'amount': sell_amount, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {asset} {sell_amount} at {price}")

    def show_profit(self):
        usdt_value = self._stacks['USDT']
        if 'ETH' in self._stacks and self._stacks['ETH'] > 0:
            market_data = self.fetch_market_data('ETHUSDT')
            if market_data:
                usdt_value += self._stacks['ETH'] * market_data['last_price']
        if 'BTC' in self._stacks and self._stacks['BTC'] > 0:
            market_data = self.fetch_market_data('BTCUSDT')
            if market_data:
                usdt_value += self._stacks['BTC'] * market_data['last_price']

        initial_value = self._initial_stacks['USDT']
        profit = usdt_value - initial_value
        print(f"Initial USD: {initial_value}")
        print(f"Current USD: {usdt_value}")
        print(f"Profit: {profit}")
        print("\nTransaction Log:")
        for log in self.transaction_log:
            print(log)
        self.results.append({'initial_usd': initial_value, 'current_usd': usdt_value, 'profit': profit})

if __name__ == "__main__":
    engine = RealTimeTradeBot()
    engine.main_engine()
    # Save results to a file
    pd.DataFrame(engine.results).to_csv('results.csv', index=False)
