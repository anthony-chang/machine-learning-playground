from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


#yearmonthday, mm, mm, C, C, deg, m/s, C
names = ['date','precipitation','snowfall','temperature-max','temperature-min','wind-direction','wind-speed','temperature-avg']
dataset = read_csv('toronto_weather_2016-2020.csv', names=names)
data_array = dataset.values
# scatter plot matrix
print(data_array[0:5, :])
