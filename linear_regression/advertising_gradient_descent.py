import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("./data/advertising.csv")
data = data.values
X = data[:, 2]
y = data[:, 4]


def predict(x, weight, bias):
    return weight * x + bias


def cost_calculate(X, y, weight, bias):
    n = len(X)
    sum = 0
    for i in range(n):
        sum += (y[i] - predict(X[i], weight, bias)) ** 2
    return sum / n


def update_weight_bias(X, y, weight, bias, learning_rate):
    n = len(X)
    sum_weight = 0.0
    sum_bias = 0.0
    for i in range(n):
        sum_weight += -2 * X[i] * (y[i] - (weight * X[i] + bias))
        sum_bias += -2 * (y[i] - (weight * X[i] + bias))
    weight -= (sum_weight / n) * learning_rate
    bias -= (sum_bias / n) * learning_rate
    return weight, bias


def train(X, y, weight, bias, learning_rate, times):
    cost_arr = []
    for i in range(times):
        weight, bias = update_weight_bias(X, y, weight, bias, learning_rate)
        cost = cost_calculate(X, y, weight, bias)
        cost_arr.append(cost)
    return weight, bias, cost_arr


# print(data)
weight, bias, cost_result = train(X, y, 0.03, 0.0014, 0.001, 60)
print(weight, bias)
print(cost_result)
print(predict(19, weight, bias))

# iter_range = [i for i in range(100)]
# plt.plot(iter_range, cost_result)
# plt.show()

plt.plot(X, y, 'go')
plt.show()
