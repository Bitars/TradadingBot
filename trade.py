import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from binance.client import Client
import time
import keys  # Import your API keys from keys.py

class RealTimeTradeBot:
    def __init__(self):
        self._stacks = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 20.0}
        self._initial_stacks = self._stacks.copy()
        self.transaction_log = []
        self.model = None  # Will be loaded later
        self.stop_loss = 0.95  # Stop loss at 5% loss
        self.take_profit = 1.05  # Take profit at 5% gain
        self.api_key = keys.api
        self.api_secret = keys.secret
        self.client = Client(self.api_key, self.api_secret)
        self.results = []  # Store results for visualization
        self.cooldown = 60  # Cooldown period in seconds
        self.last_trade_time = 0  # Timestamp of the last trade
        self.min_trade_amount = 10  # Minimum trade amount in USDT
        self.lot_size_filter = self.get_lot_size_filter()

    def get_lot_size_filter(self):
        info = self.client.get_exchange_info()
        lot_size_filter = {}
        for symbol_info in info['symbols']:
            symbol = symbol_info['symbol']
            for filt in symbol_info['filters']:
                if filt['filterType'] == 'LOT_SIZE':
                    lot_size_filter[symbol] = {
                        'minQty': float(filt['minQty']),
                        'maxQty': float(filt['maxQty']),
                        'stepSize': float(filt['stepSize'])
                    }
                    break
        return lot_size_filter

    def main_engine(self):
        self.load_model('trading_model.pkl')
        self.trade_real_time()

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def fetch_market_data(self, symbol):
        try:
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
            current_time = time.time()
            if current_time - self.last_trade_time < self.cooldown:
                time.sleep(self.cooldown - (current_time - self.last_trade_time))
                continue

            for asset in symbol_map:
                market_data = self.fetch_market_data(symbol_map[asset])
                if market_data:
                    features = self.preprocess_data(market_data)
                    prediction = self.model.predict(features)[0]

                    if prediction == 1:  # Buy signal
                        self.buy(asset, market_data['last_price'])
                    else:  # Sell signal
                        self.sell(asset, market_data['last_price'])

                    self.last_trade_time = current_time

            # Sleep for a certain period before fetching the next data point
            time.sleep(60)  # Fetch data every 60 seconds

    def adjust_quantity(self, quantity, symbol):
        lot_size = self.lot_size_filter[symbol]
        min_qty = lot_size['minQty']
        max_qty = lot_size['maxQty']
        step_size = lot_size['stepSize']

        if quantity < min_qty:
            quantity = min_qty
        elif quantity > max_qty:
            quantity = max_qty
        # Adjust quantity to the nearest step size
        quantity = round(quantity // step_size * step_size, int(-np.log10(step_size)))

        return quantity

    def buy(self, asset, price):
        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        if usdt_balance < self.min_trade_amount:
            print(f"Not enough USDT to buy {asset}. Minimum trade amount is {self.min_trade_amount} USDT.")
            return

        buy_amount_usdt = self.min_trade_amount
        quantity = buy_amount_usdt / price
        symbol = f"{asset}USDT"
        quantity = self.adjust_quantity(quantity, symbol)

        try:
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
            self._stacks['USDT'] -= buy_amount_usdt
            self._stacks[asset] += quantity

            self.transaction_log.append(f"buy {asset} {quantity} at {price}")
            self.results.append({'action': 'buy', 'pair': asset, 'amount': quantity, 'price': price, 'total_usdt': self._stacks['USDT']})
            print(f"buy {asset} {quantity} at {price}")
        except Exception as e:
            print(f"Exception placing buy order: {e}")

    def sell(self, asset, price):
        symbol = f"{asset}USDT"
        if asset == 'ETH':
            quantity = self._stacks['ETH'] * 0.25
            if quantity * price < self.min_trade_amount:
                print(f"Not enough {asset} to sell. Minimum trade amount is {self.min_trade_amount} USDT.")
                return
            quantity = self.adjust_quantity(quantity, symbol)

            try:
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                self._stacks['USDT'] += quantity * price
                self._stacks['ETH'] -= quantity
                self.transaction_log.append(f"sell {asset} {quantity} at {price}")
                self.results.append({'action': 'sell', 'pair': asset, 'amount': quantity, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {asset} {quantity} at {price}")
            except Exception as e:
                print(f"Exception placing sell order: {e}")
        elif asset == 'BTC':
            quantity = self._stacks['BTC'] * 0.25
            if quantity * price < self.min_trade_amount:
                print(f"Not enough {asset} to sell. Minimum trade amount is {self.min_trade_amount} USDT.")
                return
            quantity = self.adjust_quantity(quantity, symbol)

            try:
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                self._stacks['USDT'] += quantity * price
                self._stacks['BTC'] -= quantity
                self.transaction_log.append(f"sell {asset} {quantity} at {price}")
                self.results.append({'action': 'sell', 'pair': asset, 'amount': quantity, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {asset} {quantity} at {price}")
            except Exception as e:
                print(f"Exception placing sell order: {e}")

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
