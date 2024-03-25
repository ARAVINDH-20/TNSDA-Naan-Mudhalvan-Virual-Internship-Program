import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import yfinance
from sklearn.preprocessing import MinMaxScaler

#ML model
from keras.models import load_model

start="2015-01-01"
today=date.today().strftime("%Y-%m-%d")

stocks=["GOOG","AAPL","MSFT","GME",'IBM','AMZN',"TSLA","SBIN.NS"]

df = yfinance.download("AAPL",start,today)
df=df.reset_index()

#split data into train and test
data_traning = pd.DataFrame(df["Close"][0:int(len(df)*.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*.70)-100:])

scaler = MinMaxScaler(feature_range=(0,1))
data_traning_array = scaler.fit_transform(data_traning)


model = load_model(r"D:\MACHINE LEARNING\FINAL PROJECT\Models\finance\stock_market.h5")

input_Data = scaler.fit_transform(data_testing)


x_test = []
y_test = []

for i in range(100,input_Data.shape[0]):
    x_test.append(input_Data[i-100:i])
    y_test.append(input_Data[i,0])
    
x_test,y_test = np.array(x_test),np.array(y_test)


#making predictions
y_predicted = model.predict(x_test)

scale_factor = 1/0.01008726
y_predicted = y_predicted*scale_factor
y_test = y_test *scale_factor


plt.figure(figsize=(15,5))
plt.plot(y_test,"b",label="Original Price")
plt.plot(y_predicted,"r",label="Predicated Price")
plt.ylabel("price")
plt.xlabel("Time")
plt.show()
