import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Reading data from CSV file
data = pd.read_csv("./data/advertising.csv")
data = data.values
X = np.array([data[:, 2]]).T
y = np.array([data[:, 4]]).T

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 50, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')     # data
plt.plot(x0, y0)
plt.show()