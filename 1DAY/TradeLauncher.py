import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

class RealTimeTradeBot:
    def __init__(self):
        self._stacks = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 100.0}  # Set initial funds to $100
        self._initial_stacks = self._stacks.copy()
        self.transaction_log = []
        self.model = None
        self.results = []

    def main_engine(self):
        self.load_model('trading_model.pkl')
        self.simulate_trading('DATA/BTC_USD.csv')
        self.simulate_trading('DATA/ETH_USD.csv')
        self.show_profit()

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def load_and_preprocess_data(self, file_name):
        df = pd.read_csv(file_name, dtype=str)
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'].str.replace(',', ''), errors='coerce')
        df['high'] = pd.to_numeric(df['high'].str.replace(',', ''), errors='coerce')
        df['low'] = pd.to_numeric(df['low'].str.replace(',', ''), errors='coerce')
        df['close'] = pd.to_numeric(df['close'].str.replace(',', ''), errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'].str.replace(',', ''), errors='coerce')

        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('date')
        df['price_change'] = df['close'] - df['open']
        df['high_low_diff'] = df['high'] - df['low']
        return df

    def simulate_trading(self, file_name):
        df = self.load_and_preprocess_data(file_name)
        df.dropna(inplace=True)

        for index, row in df.iterrows():
            features = pd.DataFrame([[row['price_change'], row['high_low_diff'], row['volume']]], columns=['price_change', 'high_low_diff', 'volume'])
            prediction = self.model.predict(features)[0]

            if prediction == 1:  # Buy signal
                self.buy(file_name.split('_')[0], row['open'])
            else:  # Sell signal
                self.sell(file_name.split('_')[0], row['open'])

    def buy(self, pair, price):
        if pair == 'DATA/ETH':
            buy_amount = self._stacks['USDT'] * 0.1
            if buy_amount > 0:
                self._stacks['USDT'] -= buy_amount
                self._stacks['ETH'] += buy_amount / price
                self.transaction_log.append(f"buy {pair} {buy_amount / price} at {price}")
                self.results.append({'action': 'buy', 'pair': pair, 'amount': buy_amount / price, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"buy {pair} {buy_amount / price} at {price}")
        elif pair == 'DATA/BTC':
            buy_amount = self._stacks['USDT'] * 0.1
            if buy_amount > 0:
                self._stacks['USDT'] -= buy_amount
                self._stacks['BTC'] += buy_amount / price
                self.transaction_log.append(f"buy {pair} {buy_amount / price} at {price}")
                self.results.append({'action': 'buy', 'pair': pair, 'amount': buy_amount / price, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"buy {pair} {buy_amount / price} at {price}")

    def sell(self, pair, price):
        if pair == 'DATA/ETH':
            sell_amount = self._stacks['ETH'] * 0.25
            if sell_amount > 0:
                self._stacks['USDT'] += sell_amount * price
                self._stacks['ETH'] -= sell_amount
                self.transaction_log.append(f"sell {pair} {sell_amount} at {price}")
                self.results.append({'action': 'sell', 'pair': pair, 'amount': sell_amount, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {pair} {sell_amount} at {price}")
        elif pair == 'DATA/BTC':
            sell_amount = self._stacks['BTC'] * 0.25
            if sell_amount > 0:
                self._stacks['USDT'] += sell_amount * price
                self._stacks['BTC'] -= sell_amount
                self.transaction_log.append(f"sell {pair} {sell_amount} at {price}")
                self.results.append({'action': 'sell', 'pair': pair, 'amount': sell_amount, 'price': price, 'total_usdt': self._stacks['USDT']})
                print(f"sell {pair} {sell_amount} at {price}")

    def show_profit(self):
        usdt_value = self._stacks['USDT']
        if 'ETH' in self._stacks and self._stacks['ETH'] > 0:
            usdt_value += self._stacks['ETH'] * self.get_last_close_price('DATA/ETH_USD.csv')
        if 'BTC' in self._stacks and self._stacks['BTC'] > 0:
            usdt_value += self._stacks['BTC'] * self.get_last_close_price('DATA/BTC_USD.csv')

        initial_value = self._initial_stacks['USDT']
        profit = usdt_value - initial_value
        print(f"Initial USD: {initial_value}")
        print(f"Current USD: {usdt_value}")
        print(f"Profit: {profit}")
        print("\nTransaction Log:")
        for log in self.transaction_log:
            print(log)
        self.results.append({'initial_usd': initial_value, 'current_usd': usdt_value, 'profit': profit})

    def get_last_close_price(self, file_name):
        df = pd.read_csv(file_name, dtype=str)
        df = df.rename(columns={
            'Date': 'date',
            'Close': 'close'
        })
        df['close'] = pd.to_numeric(df['close'].str.replace(',', ''), errors='coerce')
        df = df.dropna(subset=['close'])
        return df['close'].iloc[-1]

if __name__ == "__main__":
    engine = RealTimeTradeBot()
    engine.main_engine()
    # Save results to a file
    pd.DataFrame(engine.results).to_csv('results.csv', index=False)
