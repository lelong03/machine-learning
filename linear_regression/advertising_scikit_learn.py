from sklearn import datasets, linear_model
import pandas as pd
import numpy as np

# Reading data from CSV file
data = pd.read_csv("./data/advertising.csv")
data = data.values
X = np.array([data[:, 2]]).T
y = np.array([data[:, 4]]).T

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

print('w = ', regr.coef_)
