import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model("/Users/ranjith/Desktop/Stock Market Prediction/Stock Predictions Model.keras")

# Streamlit UI
st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-01-01'
end = '2025-01-01'

# Download stock data
data = yf.download(stock, start, end)

# Handle invalid stock symbols
if data.empty:
    st.error("Invalid stock symbol. Please enter a valid stock ticker.")
    st.stop()

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Split into train and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.8)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)  # Fit only on training data

# Prepare test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# Moving Averages
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
st.pyplot(fig1)  # Removed plt.show()

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10, 8))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(ma_100_days, 'b', label="MA100")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
st.pyplot(fig2)  # Removed plt.show()

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="MA100")
plt.plot(ma_200_days, 'b', label="MA200")
plt.plot(data.Close, 'g', label="Stock Price")
plt.legend()
st.pyplot(fig3)  # Removed plt.show()

# Prepare Data for Prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

# Convert to NumPy arrays
x, y = np.array(x), np.array(y)

# Make Predictions
predict = model.predict(x)

# Rescale Predictions
scale = 1 / scaler.scale_[0]  # Fixed scaling issue
predict = predict.reshape(-1) * scale  # Flatten
y = y.reshape(-1) * scale  # Flatten

# Plot Predictions
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')  # Fixed label
plt.plot(y, 'g', label='Original Price')  # Fixed label
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)  # Removed plt.show()
