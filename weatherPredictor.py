from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np


#yearmonthday, mm, mm, C, C, deg, m/s, C
names = ['date','precipitation','snowfall','temperature-max','temperature-min','wind-direction','wind-speed','temperature-avg']
dataset = read_csv('toronto_weather_2016-2020.csv', names=names, usecols=['date', 'temperature-avg'])
# impute our data, missing values -> mean
dataset = dataset.replace(-99999, np.NaN)
dataset.fillna(dataset.mean(), inplace=True)
# print(dataset.describe())

# store our features and labels
data_array = dataset.values
x = data_array[:, 0]
y = data_array[:, 1]
x = x.reshape((x.shape[0], 1))
y = y.reshape((y.shape[0], 1))

# 60% used as training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

model = DecisionTreeRegressor()
model.fit(x_train, y_train)
# predictions = model.predict(x_test)
print(model.score(x_test, y_test))
print(model.predict([[20190724]]))
