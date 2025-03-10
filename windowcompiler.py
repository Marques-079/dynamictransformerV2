import pandas as pd

#Loads in the data
df = pd.read_csv('AAPL_1min_data.csv')

df['date_time'] = pd.to_datetime(df['date_time'])
df = df.set_index('date_time').sort_index()
#print(df.describe())

#windowcompiler-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

feature_cols = ['volume', 'volume_weighted', 'open_price', 'close', 'high', 'low',
       'number_trades', 'log_return', 'high_low_spread', 'close_open_change', 'sma_5',
       'sma_15', 'ema_5', 'ema_15', 'volatility_5', 'volatility_15',
       'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'atr_14',
       'volume_change_pct', 'vwap', 'obv', 'rsi_14', 'momentum_5',
       'momentum_15', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'roc_10',
       'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi']

label_col = "price_change_pct"  # or something else

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_rolling_windows(df, feature_cols, label_col, window_size=10):

    # Extract the feature matrix
    feature_data = df[feature_cols].values
    
    # Extract the label array
    label_data = df[label_col].values

    # Fit a MinMax scaler on the entire feature set
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Uncomment the line below to actually scale the data:
    # feature_data_scaled = scaler.fit_transform(feature_data)
    feature_data_scaled = feature_data  # Currently no scaling

    # Create a 0-1 time index for one window
    time_index = np.linspace(0, 1, window_size).reshape(-1, 1)

    X, y = [], []
    n = len(df)
    
    for i in range(n - window_size):
        # Extract the window of features
        window_features = feature_data_scaled[i : i + window_size]
        
        # Append the 0-1 time index as an extra column
        window_with_time = np.hstack([window_features, time_index])
        
        X.append(window_with_time)
        
        # Label is the value at the next time step (i+window_size)
        y.append(label_data[i + window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler


# Example usage
window_size = 10  # 10 rows -> 10 minutes
X, y, scaler = create_rolling_windows(df, feature_cols, label_col, window_size=window_size)

# Now X.shape = (num_samples, 10, len(feature_cols))
#     y.shape = (num_samples,)

# Suppose we do an 80/20 train/test split:
split_idx = int(len(X)*0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Feed (X_train, y_train) to your model ...


