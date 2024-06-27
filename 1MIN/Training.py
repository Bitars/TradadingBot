import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib

def load_and_preprocess_data(file_name):
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
    df.dropna(inplace=True)

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    print(df.head())  # Debugging: Print the first few rows to verify the 'target' column
    return df.dropna()

def create_sequences(data, n_steps):
    sequences = []
    targets = data['target'].values
    data = data.drop(columns=['target'])
    for i in range(len(data) - n_steps):
        sequence = data.iloc[i:i + n_steps].values
        target = targets[i + n_steps]
        sequences.append((sequence, target))
    return sequences

def train_model(file_name, model_filename, scaler_filename):
    df = load_and_preprocess_data(file_name)
    features = df[['price_change', 'high_low_diff', 'volume', 'moving_avg_5', 'moving_avg_10', 'momentum']]
    target = df['target']

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Ensure 'target' column is retained for sequence creation
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    features_scaled_df['target'] = target.values

    sequences = create_sequences(features_scaled_df, 10)

    X, y = zip(*sequences)
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)

    print(f"Model saved to {model_filename}")
    print(f"Scaler saved to {scaler_filename}")

if __name__ == "__main__":
    train_model('ETH_USD.csv', 'trading_model.h5', 'scaler.pkl')
