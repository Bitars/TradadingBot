import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

class RealTimeTradeBot:
    def __init__(self):
        self._stacks = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 100.0}  # Initial funds in USDT
        self._initial_stacks = self._stacks.copy()
        self.transaction_log = []
        self.model = None
        self.scaler = None
        self.results = []
        self.n_steps = 10
        self.history = []
        self.profit_target = 0.05  # 5% profit target
        self.stop_loss = 0.02  # 2% stop loss
        self.last_buy_price = {'BTC': None, 'ETH': None}

    def main_engine(self):
        try:
            self.load_model('trading_model.h5', 'scaler.pkl')
            self.simulate_trading('DATA/BTC_USD.csv')
            self.simulate_trading('DATA/ETH_USD.csv')
            self.show_profit()
        except Exception as e:
            print(f"An error occurred: {e}")

    def load_model(self, model_filename, scaler_filename):
        self.model = load_model(model_filename)
        self.scaler = joblib.load(scaler_filename)
        print(f"Model loaded from {model_filename}")
        print(f"Scaler loaded from {scaler_filename}")

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
        df['moving_avg_5'] = df['close'].rolling(window=5).mean()
        df['moving_avg_10'] = df['close'].rolling(window=10).mean()
        df['momentum'] = df['close'] - df['close'].shift(10)
        return df.dropna()

    def simulate_trading(self, file_name):
        df = self.load_and_preprocess_data(file_name)
        df.dropna(inplace=True)

        for index, row in df.iterrows():
            features = self.scaler.transform([[row['price_change'], row['high_low_diff'], row['volume'], row['moving_avg_5'], row['moving_avg_10'], row['momentum']]])
            self.history.append(features[0])
            if len(self.history) > self.n_steps:
                self.history.pop(0)
            if len(self.history) == self.n_steps:
                prediction = self.model.predict(np.array([self.history]))[0][0]
                print(f"Prediction: {prediction}, Price: {row['open']}")  # Print prediction and current price

                if prediction > 0.52 and not self.is_recent_buy(file_name.split('/')[-1].split('_')[0]):
                    print(f"Buy Signal: Prediction {prediction} > 0.52")
                    self.buy(file_name.split('/')[-1].split('_')[0], row['open'])
                elif prediction < 0.45:
                    print(f"Sell Signal: Prediction {prediction} < 0.45")
                    self.sell(file_name.split('/')[-1].split('_')[0], row['open'])
                else:
                    print(f"No action taken for Prediction: {prediction}")

    def buy(self, pair, price):
        buy_amount = self._stacks['USDT'] * 0.1
        if buy_amount > 0:
            self._stacks['USDT'] -= buy_amount
            self._stacks[pair] += buy_amount / price
            self.last_buy_price[pair] = price
            self.transaction_log.append(f"buy {pair} {buy_amount / price} at {price}")
            self.results.append({'action': 'buy', 'pair': pair, 'amount': buy_amount / price, 'price': price, 'total_usdt': self._stacks['USDT']})
            print(f"buy {pair} {buy_amount / price} at {price}")

    def sell(self, pair, price):
        if self._stacks[pair] > 0:
            sell_amount = self._stacks[pair] * 0.25
            self._stacks['USDT'] += sell_amount * price
            self._stacks[pair] -= sell_amount
            self.transaction_log.append(f"sell {pair} {sell_amount} at {price}")
            self.results.append({'action': 'sell', 'pair': pair, 'amount': sell_amount, 'price': price, 'total_usdt': self._stacks['USDT']})
            print(f"sell {pair} {sell_amount} at {price}")

    def is_recent_buy(self, pair):
        if self.last_buy_price[pair] is None:
            return False
        current_price = self.history[-1][0] * self.scaler.scale_[0] + self.scaler.min_[0]
        if (current_price - self.last_buy_price[pair]) / self.last_buy_price[pair] > self.profit_target:
            return False
        if (current_price - self.last_buy_price[pair]) / self.last_buy_price[pair] < -self.stop_loss:
            return False
        return True

    def show_profit(self):
        usdt_value = self._stacks['USDT']
        if self._stacks['ETH'] > 0:
            usdt_value += self._stacks['ETH'] * self.get_last_close_price('DATA/ETH_USD.csv')
        if self._stacks['BTC'] > 0:
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
