# https: // www.hackerrank.com/challenges/time-series-prediction/problem
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np


N = int(input())
x_train = []
y_train = []
for i in range(N):
    x_train.append(i)
    y_train.append(int(input()))


x_train = np.array(x_train)
x_train = x_train.reshape(-1, 1)
y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
# print(y_train.shape)

scl_x = StandardScaler()
scl_y = StandardScaler()

x_train = scl_x.fit_transform(x_train)
y_train = scl_y.fit_transform(y_train)

model = SVR()
model.fit(x_train, y_train.ravel())

for i in range(N, N+31):
    print(scl_y.inverse_transform(
        (model.predict(scl_x.transform(np.array([[i]])))))[0])
