import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import yfinance
from sklearn.preprocessing import MinMaxScaler

#ML model
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

start="2015-01-01"
today=date.today().strftime("%Y-%m-%d")

stocks=["GOOG","AAPL","MSFT","GME",'IBM','AMZN',"TSLA","SBIN.NS"]

df = yfinance.download("AAPL",start,today)
df=df.reset_index()

#split data into train and test
data_traning = pd.DataFrame(df["Close"][0:int(len(df)*.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_traning_array = scaler.fit_transform(data_traning)

x_train = []
y_train = []

for i in range(100,data_traning_array.shape[0]):
    x_train.append(data_traning_array[i-100:i])
    y_train.append(data_traning_array[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(units=50,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation="relu",return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation="relu",return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(units=1))

print(model.summary())

model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(x_train,y_train,epochs=50)

model.save(r"D:\MACHINE LEARNING\FINAL PROJECT\Models\finance\stock_market.h5")

