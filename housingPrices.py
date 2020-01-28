# Enter your code here. Read input from STDIN. Print output to STDOUT
from sklearn import linear_model
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

model = linear_model.LinearRegression()
model.fit(x_train, y_train)

T = int(input())
x_test = [0 for i in range(T)]
for i in range(T):
    x_test[i] = list(map(float, input().split()))

for i in range(T):
    print(model.predict([x_test[i]])[0])
