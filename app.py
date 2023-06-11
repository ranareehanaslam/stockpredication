import yfinance as yf
import streamlit as st
import pandas as pd
import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.write("""
# Simple Stock Price App
Shown are the stock **closing price** and **volume**.
""")

def user_input_features() :
    stock_symbol = st.sidebar.selectbox('Symbol',('NATF', 'COLG'))
    date_start = st.sidebar.date_input("Start Date", datetime.date(2005, 5, 31))
    date_end = st.sidebar.date_input("End Date", datetime.date.today())

    tickerData = yf.Ticker(stock_symbol+'.KA')
    tickerDf = tickerData.history(period='1d', start=date_start, end=date_end)
    return tickerDf, stock_symbol

input_df, stock_symbol = user_input_features()

st.line_chart(input_df.Close)
st.line_chart(input_df.Volume)

st.write("""
# Stock Price Prediction By AbuBakkar Bhai Kamran Best Friend 🤣😀
Shown are the stock prediction for next 20 days.
""")

n_steps = 100
n_features = 1

model=Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['mean_squared_error'])

model.load_weights(stock_symbol +".KA"+ ".h5")
df = input_df.dropna(axis=0, how='any', subset=None, inplace=False)
df = df[df.Volume > 0]

close = df['Close'][-n_steps:].to_list()
min_in = min(close)
max_in = max(close)
in_seq = []
for i in close :
  in_seq.append((i - min_in) / (max_in - min_in))

for i in range(20) :
  x_input = np.array(in_seq[-100:])
  x_input = x_input.reshape((1, n_steps, n_features))
  yhat = model.predict(x_input, verbose=0)
  in_seq.append(yhat[0][0])

norm_res = in_seq[-20:]
res = []
for i in norm_res :
  res.append(i * (max_in - min_in) + min_in)

closepred = close[-80:]
for x in res :
  closepred.append(x)

plt.figure(figsize = (20,10))
plt.plot(closepred, label="Prediction")
plt.plot(close[-80:], label="Previous")
plt.ylabel('Price (Rp)', fontsize = 15 )
plt.xlabel('Days', fontsize = 15 )
plt.title(stock_symbol + " Stock Prediction", fontsize = 20)
plt.legend()
plt.grid()

st.pyplot(plt)


import numpy as np
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(close).reshape(-1,1))
x_input = df1[len(df1)-100:].reshape(1, -1)
x_input = x_input.reshape((1, 100, 1))

predictions = []

for _ in range(7):
    y_predict = model.predict(x_input, verbose=0)
    next_day_value = scaler.inverse_transform(y_predict)
    predictions.append(next_day_value[0, 0])  # Assuming you want the first value in the prediction
    
    # Update x_input for the next iteration
    x_input = np.concatenate((x_input[:, 1:, :], y_predict.reshape(1, 1, 1)), axis=1)
plt.figure(figsize = (20,10))
plt.plot(predictions, label="Prediction")
plt.plot(close[-80:], label="Previous")
plt.ylabel('Price (Rp)', fontsize = 15 )
plt.xlabel('Days', fontsize = 15 )
plt.title(stock_symbol + " Stock Prediction", fontsize = 20)
plt.legend()
plt.grid()
# Print the predictions for the next 7 days
for i, prediction in enumerate(predictions):
    print(f"Day {i+1}: {prediction}")
st.pyplot(plt)
