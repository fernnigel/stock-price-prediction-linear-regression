import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading dataset
df = pd.read_csv('NIFTY50.csv')

#dropping rows with na value
df.dropna(inplace = True)

#dropping duplicates
df.drop_duplicates()

#converting date to numerical value for calculations
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.datetime.toordinal)

#getting the x and y variables
y = np.array(df['Open']).reshape(-1,1)
x= np.array(df['Date']).reshape(-1,1)

#spilting data into test and train 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

#Linear Regression Class from sklearn
regr = LinearRegression()

#fitting the model
regr.fit(X_train, y_train)

#predict the model on test data
y_pred = regr.predict(X_test)

#calculationg mae,mse and rmse
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
  
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#plotting the model
plt.title("NIFTY 50 Trend")
plt.xlabel("Date")
plt.ylabel("Open")

plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
  
plt.show()