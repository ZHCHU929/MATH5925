import yfinance as yf
import pandas as pd
import numpy as np

# Define the ticker symbol for gold futures
ticker_symbol = 'NVDA'

# Define the start and end dates
start_date = '2018-06-01'
end_date = '2024-06-01'

# Download the data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Reset the index to make 'Date' a column
data.reset_index(inplace=True)

# Drop any rows with missing values
data.dropna(inplace=True)

# Extract the necessary columns
data = data[['Date', 'Close','Date','Volume']]

# Optionally, save the data to a CSV file
data.to_csv('nvda-6.csv', index=False)


data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# 计算日波动率（标准差）
daily_volatility = data['Log Returns'].std()

# 年化波动率
annualized_volatility = daily_volatility * np.sqrt(252)

print(f'Annualized Volatility: {annualized_volatility:.2%}')

# Display the first few rows of the dataframe
print(data.tail())