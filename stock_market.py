import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock

# Calculate technical indicators
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])
    return df

# Compute RSI (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute MACD (Moving Average Convergence Divergence)
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Get stock data
ticker = 'AAPL'  # Example: Apple stock
start_date = '2020-01-01'
end_date = '2024-01-01'
data = get_stock_data(ticker, start_date, end_date)

# Add technical indicators
data = add_technical_indicators(data)

# Display data
print(data.tail())

# Plot closing price and indicators
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA_20'], label='SMA 20', color='red')
plt.plot(data['SMA_50'], label='SMA 50', color='green')
plt.title(f'{ticker} Stock Price with Technical Indicators')
plt.legend()
plt.show()
