import requests
import pandas as pd


API_KEY = ''#KEY HERE#
# API URL
url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/minute/2024-08-01/2025-01-01?adjusted=true&sort=asc&limit=50000&apiKey=API_KEY"


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
       print(df.head())


       # Save to CSV (optional)
       df.to_csv("AAPL_1min_data.csv", index=False)
       print("Data saved to AAPL_1min_data.csv")
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
