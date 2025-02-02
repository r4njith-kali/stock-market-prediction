# 📈 Stock Market Prediction App

This project is a **Stock Market Prediction Web App** that uses **LSTM (Long Short-Term Memory) networks** to forecast stock prices. The model was trained in **Jupyter Notebook** and deployed as a **Streamlit web application** using **VS Code**.

## 🚀 Features
- 📊 **Download Stock Data** from Yahoo Finance (`yfinance` API).
- 📈 **Calculate Moving Averages** (50-day, 100-day, and 200-day MAs).
- 🔮 **Predict Future Stock Prices** using a trained LSTM model.
- 🎨 **Interactive UI** with dynamic stock selection.

---

## 📁 Project Structure
```
Stock-Market-Prediction/
│── stock_prediction.ipynb   # Jupyter Notebook (Model Training)
│── app.py                   # Streamlit Web App
│── Stock Predictions Model.keras  # Trained LSTM Model
│── requirements.txt         # Required Libraries
│── README.md                # Project Documentation
```

---

## 📦 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Stock-Market-Prediction.git
cd Stock-Market-Prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web App
```bash
streamlit run app.py
```

---

## 📚 Model Training in Jupyter Notebook
The **LSTM model** was trained in `stock_prediction.ipynb` using the following steps:

1️⃣ **Import Dependencies**
```python
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
```

2️⃣ **Download & Preprocess Data**
```python
stock = 'GOOG'
start, end = '2014-01-01', '2025-01-01'
data = yf.download(stock, start, end)
data_train = data['Close'][0:int(len(data)*0.8)]
data_test = data['Close'][int(len(data)*0.8):]
```

3️⃣ **Feature Scaling & Data Preparation**
```python
scaler = MinMaxScaler(feature_range=(0,1))
data_train_scaled = scaler.fit_transform(np.array(data_train).reshape(-1,1))

x_train, y_train = [], []
for i in range(100, len(data_train_scaled)):
    x_train.append(data_train_scaled[i-100:i])
    y_train.append(data_train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
```

4️⃣ **Build & Train LSTM Model**
```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)),
    Dropout(0.2),
    LSTM(60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(120, activation='relu', return_sequences=False),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)
```

5️⃣ **Save the Model**
```python
model.save("Stock Predictions Model.keras")
```

---

## 🌐 Web App Deployment in VS Code
The trained LSTM model is loaded in `app.py` and used in a **Streamlit web app**.

### 🛠 Key Features in `app.py`
1. **Load the trained model**
```python
from keras.models import load_model
model = load_model("Stock Predictions Model.keras")
```

2. **Download Stock Data**
```python
import yfinance as yf
stock = 'GOOG'
data = yf.download(stock, '2014-01-01', '2025-01-01')
```

3. **Calculate Moving Averages**
```python
ma_50_days = data['Close'].rolling(50).mean()
ma_100_days = data['Close'].rolling(100).mean()
ma_200_days = data['Close'].rolling(200).mean()
```

4. **Make Predictions & Plot Results**
```python
predict = model.predict(x_test)
predict = predict * (1 / scaler.scale_[0])
```

5. **Streamlit UI**
```python
import streamlit as st
st.header('Stock Market Predictor')
st.subheader('Stock Data')
st.write(data)
st.pyplot(fig1)
```

🖼️ Screenshots

📉 Stock Price vs MA50 vs MA100

![4F0DB8CC-3726-4E55-A834-D8DAF143CFC1](https://github.com/user-attachments/assets/8e3d96ab-b908-4ac2-a12e-86e8d879877b)


📊 Stock Price vs MA100 vs MA200

![08CFE7D4-39C6-4981-8CA5-275E06D71CA5](https://github.com/user-attachments/assets/3b452340-ffd0-46f9-a861-12e90d477d23)


🔮 Stock Price Prediction

![6F1EDC7D-BF53-41D9-9132-A5B0CEA82EBB](https://github.com/user-attachments/assets/74f24013-ff12-4300-b08b-85bebf89e71e)


📋 10-Year Stock Data Used

![7F9A3052-AAC6-4750-9FB6-6D41DA4A75FB](https://github.com/user-attachments/assets/8adb98e0-adc9-4fa8-8893-c3b171f0dde3)


## 📌 Future Improvements
- ✅ **Enhance Model Accuracy** with additional features (Volume, Open/High/Low Prices).
- ✅ **Deploy the Web App Online** using Streamlit Sharing or AWS.
- ✅ **Add Real-Time Data Updates** using an API.

---

