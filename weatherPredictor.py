from datetime import date, datetime
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# yearmonthday, mm, mm, C, C, 10s of deg, km/h, C
# https://climate.weather.gc.ca/climate_data/daily_data_e.html?StationID=51459
names = ['date','temperature-avg', 'temperature-max', 'temperature-min','wind-direction','wind-speed', 'precipitation', 'snowfall','predicted-temp']
dataset = read_csv('toronto_weather_2016-2020.csv', names=names, skiprows=1)
# impute our data, missing values -> mean
dataset = dataset.replace(-99999, np.NaN)
dataset.fillna(dataset.mean(), inplace=True)
# print(dataset.describe())

# store our features and labels
data_array = dataset.values
x = data_array[:, 0:(len(names)-1)]
y = data_array[:, len(names)-1]
x = x.reshape(-1, len(names)-1)

# 50% used as training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

model = DecisionTreeRegressor()
model.fit(x_train, y_train)
# predictions = model.predict(x_test)
print(model.score(x_test, y_test))

arr = []
today = date.today()
print('The date is: ' + str(today))
arr.append(int(str(today.year)+str(today.month)+str(today.day))+1)
arr.append(input('Enter the average temperature today (deg C): '))
arr.append(input('Enter the max temperature today (deg C): '))
arr.append(input('Enter the min temperature today (deg C): '))
arr.append(input('Enter the peak gust wind direction today (10s of deg): '))
arr.append(input('Enter the peak guest wind speed today (km/h): '))
arr.append(input('Enter the amount of rain today (mm): '))
arr.append(input('Enter the amount of snowfall today (mm): '))

print('The temperature tomorrow will be ' + str(model.predict([arr])[0]))
