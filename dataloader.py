import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the API key
api_key = os.getenv("API_KEY_AV")
ticker = 'NVDA'
start_date = '2024-08-01'
end_date = '2025-01-01'

print(f"Your API Key: {api_key}")  # It should print the API key


# API URL
url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"


# Fetch the data from the API
response = requests.get(url)


# Check if request was successful
if response.status_code == 200:
   data = response.json()


   # Extract the results
   if "results" in data:
       df = pd.DataFrame(data["results"])


       # Convert timestamp to readable datetime
       df["datetime"] = pd.to_datetime(df["t"], unit="ms")


       # Rename columns for clarity
       df.rename(columns={"t": "timestamp"}, inplace=True)


       # Display first few rows
       #print(df.head())


       
       #df.to_csv("AAPL_1min_data.csv", index=False)
       #print("Data saved to AAPL_1min_data.csv")
   else:
       print("No results found in the API response.")
else:
   print(f"Failed to retrieve data: {response.status_code}, {response.text}")




df['volume'] = df['v']
df['volume_weighted'] = df['vw']
df['open_price'] = df['o']
df['close'] = df['c']
df['high'] = df['h']
df['low'] = df['l']
df['number_trades'] = df['n']
df['time_stamp'] = df['timestamp']
df['date_time'] = df['datetime']


df = df.drop(df.columns[:9], axis=1)

#v2 Creating, more metrics derived from oeiginal datagframe# ----------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

# -------------------------------
# 1. Ensure datetime format
# -------------------------------
# Convert timestamp from milliseconds to datetime.
df['date_time'] = pd.to_datetime(df['time_stamp'], unit='ms')

# -------------------------------
# 2. Price Momentum Features
# -------------------------------
# Log Return: natural log of today's close over yesterday's close.
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Price Change Percentage: percentage change between consecutive closing prices.
df['price_change_pct'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100

# High-Low Spread: percentage difference between high and low for each period.
df['high_low_spread'] = (df['high'] - df['low']) / df['low'] * 100

# Close-Open Change: percentage change from open to close.
df['close_open_change'] = (df['close'] - df['open_price']) / df['open_price'] * 100

# -------------------------------
# 3. Moving Average Features
# -------------------------------
# Simple Moving Averages (SMA)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_15'] = df['close'].rolling(window=15).mean()

# Exponential Moving Averages (EMA)
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()

# -------------------------------
# 4. Volatility Indicators
# -------------------------------
# Rolling Standard Deviation as a measure of volatility.
df['volatility_5'] = df['close'].rolling(window=5).std()
df['volatility_15'] = df['close'].rolling(window=15).std()

# Bollinger Bands (using the 15-period SMA and volatility)
df['bollinger_upper'] = df['sma_15'] + 2 * df['volatility_15']
df['bollinger_lower'] = df['sma_15'] - 2 * df['volatility_15']
df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_15']

# Average True Range (ATR) over a 14-period window.
# First compute the True Range (TR) for each row.
df['prev_close'] = df['close'].shift(1)
df['true_range'] = df[['high', 'low', 'prev_close']].apply(
    lambda x: max(x['high'] - x['low'], abs(x['high'] - x['prev_close']), abs(x['low'] - x['prev_close'])),
    axis=1
)
df['atr_14'] = df['true_range'].rolling(window=14).mean()

# -------------------------------
# 5. Volume-Based Features
# -------------------------------
# Volume Change Percentage: percent change in volume relative to the previous period.
df['volume_change_pct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100

# Volume Weighted Average Price (VWAP): cumulative calculation using base price * volume.
df['vwap'] = (df['volume_weighted'] * df['volume']).cumsum() / df['volume'].cumsum()

# On-Balance Volume (OBV): cumulative measure of buying/selling pressure.
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

# -------------------------------
# 6. Momentum Indicators
# -------------------------------
# Relative Strength Index (RSI) with a 14-period window.
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# Price Momentum: difference in close price over 5 and 15 periods.
df['momentum_5'] = df['close'] - df['close'].shift(5)
df['momentum_15'] = df['close'] - df['close'].shift(15)

# MACD Indicator: Moving Average Convergence Divergence
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()


df.drop(['prev_close', 'true_range'], axis=1, inplace=True)

df.dropna(inplace=True)


#v3 Second data frame creation, this time with more metrics derived from the original dataframe# ---------------------------------------------------------------------------------------------

import numpy as np

# 1. Rate of Change (ROC)
# Calculates the percentage change from 'n' periods ago (here n = 10)
n = 10
df['roc_10'] = df['close'].pct_change(periods=n) * 100

# 2. Stochastic Oscillator (%K and %D)
# Uses a 14-period window to calculate the oscillator.
low_min = df['low'].rolling(window=14).min()
high_max = df['high'].rolling(window=14).max()
df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

# 3. Williams %R
# An oscillator that ranges from -100 (oversold) to 0 (overbought)
df['williams_r'] = ((high_max - df['close']) / (high_max - low_min)) * -100

# 4. Commodity Channel Index (CCI)
# Measures deviation of typical price from its 20-period moving average.
typical_price = (df['high'] + df['low'] + df['close']) / 3
sma_typical = typical_price.rolling(window=20).mean()
mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
df['cci'] = (typical_price - sma_typical) / (0.015 * mean_deviation)

# 5. Money Flow Index (MFI)
# Combines price and volume to gauge buying and selling pressure over a 14-period window.
# Re-calculate typical price for MFI.
typical_price = (df['high'] + df['low'] + df['close']) / 3
raw_money_flow = typical_price * df['volume']
tp_change = typical_price.diff()

# Determine positive and negative money flow.
positive_flow = raw_money_flow.where(tp_change > 0, 0)
negative_flow = raw_money_flow.where(tp_change < 0, 0)

# Sum over a 14-period window.
sum_positive_flow = positive_flow.rolling(window=14).sum()
sum_negative_flow = negative_flow.abs().rolling(window=14).sum()

# Money Flow Ratio and MFI calculation.
money_flow_ratio = sum_positive_flow / sum_negative_flow
df['mfi'] = 100 - (100 / (1 + money_flow_ratio))

# Optionally, remove rows with NaN values introduced by the rolling calculations.

df.dropna(inplace=True)

df.to_csv("AAPL_1min_data.csv", index=False)
print(df.head())




##TESTING FOR THE DATAFRAME# -----------------------------------------------------------------------------------------------------------------------------------------------------------


print(df.info())
print(df.describe())
