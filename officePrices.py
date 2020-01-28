# https: // www.hackerrank.com/challenges/predicting-office-space-price/problem
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


features, N = (int(n) for n in input().split())
x_train = []
y_train = []
x_test = []

x_train = [0 for i in range(N)]
for i in range(N):
    x_train[i] = list(map(float, input().split()))

x_train = np.array(x_train)
y_train = x_train[:, features]
x_train = x_train[:, 0:features]

poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
model = linear_model.LinearRegression(fit_intercept=True)
model.fit(x_train_poly, y_train)

T = int(input())
x_test = [0 for i in range(T)]
for i in range(T):
    x_test[i] = list(map(float, input().split()))

x_test_poly = poly.fit_transform(x_test)

for i in range(T):
    print(model.predict([x_test_poly[i]])[0])
