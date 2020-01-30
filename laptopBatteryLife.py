# https: // www.hackerrank.com/challenges/battery/problem
from sklearn import linear_model
import numpy as np
import pandas as pd
from matplotlib import pyplot 

names = ['charge', 'life']
data = pd.read_csv("battery_data.txt", names=names, delimiter=',')
data = data.replace(8.00, np.NaN)
data.dropna(inplace=True)


x_train = data['charge'].values
y_train = data['life'].values
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

model.predict(x_train)

