#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : logistic_regression_with_nonlinear_boundary.py
# Time       ：2024/7/3 23:02
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from logistic_regression import LogisticRegression

data = pd.read_csv('data/microchips-tests.csv')

validities = [0, 1]
x_axis = 'param_1'
y_axis = 'param_2'
# for validity in validities:
#     plt.scatter(
#         data[x_axis][data['validity'] == validity],
#         data[y_axis][data['validity'] == validity],
#         label=validity
#     )
#
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.title('Microchips Tests')
# plt.legend()
# plt.show()

train_data = data.sample(frac=0.7)
test_data = data.drop(train_data.index)
print(f'train {train_data.shape[0]}, test {test_data.shape[0]}')

x_train = train_data[[x_axis, y_axis]].values.reshape((train_data.shape[0], 2))
y_train = train_data['validity'].values.reshape((train_data.shape[0], 1))

x_test = test_data[[x_axis, y_axis]].values.reshape((test_data.shape[0], 2))
y_test = test_data['validity'].values.reshape((test_data.shape[0], 1))

max_iterations = 100000
regularization_param = 0
polynomial_degree = 5
sinusoid_degree = 0
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree, normalize_data=False)

thetas, costs = logistic_regression.train(max_iterations)
columns = []
for theta_index in range(0, thetas.shape[1]):
    columns.append('Theta ' + str(theta_index))

# labels = logistic_regression.unique_labels
# plt.plot(range(len(costs[0])), costs[0], label=labels[0])
# plt.plot(range(len(costs[1])), costs[1], label=labels[1])
#
# plt.xlabel('Gradient Steps')
# plt.ylabel('Cost')
# plt.legend()
# plt.show()

predictions = logistic_regression.predict(x_test)
precision = np.sum(predictions == y_test) / y_test.shape[0]
print(f'precision: {precision}')

samples = 150
x_min = np.min(x_test[:, 0])
x_max = np.max(x_test[:, 0])
y_min = np.min(x_test[:, 1])
y_max = np.max(x_test[:, 1])
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
Z = np.zeros((samples, samples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        Z[x_index][y_index] = logistic_regression.predict(data)[0][0]

positives = (y_test == 1).flatten()
negatives = (y_test == 0).flatten()

plt.scatter(x_test[negatives, 0], x_test[negatives, 1], label='0')
plt.scatter(x_test[positives, 0], x_test[positives, 1], label='1')

plt.contour(X, Y, Z)

plt.xlabel('param_1')
plt.ylabel('param_2')
plt.title('Microchips Tests')
plt.legend()

plt.show()
