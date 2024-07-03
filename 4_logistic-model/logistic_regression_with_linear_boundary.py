#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : logistic_regression_with_linear_boundary.py
# Time       ：2024/7/3 21:29
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
x_axis = 'petal_length'
y_axis = 'petal_width'

# for iris_type in iris_types:
#     plt.scatter(data[x_axis][data['class'] == iris_type],
#                 data[y_axis][data['class'] == iris_type],
#                 label=iris_type)
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.show()

train_data = data.sample(frac=0.7)
test_data = data.drop(train_data.index)
print(f'train {train_data.shape[0]}, test {test_data.shape[0]}')

x_train = train_data[[x_axis, y_axis]].values.reshape((train_data.shape[0], 2))
y_train = train_data['class'].values.reshape((train_data.shape[0], 1))

x_test = test_data[[x_axis, y_axis]].values.reshape((test_data.shape[0], 2))
y_test = test_data['class'].values.reshape((test_data.shape[0], 1))

classifier = LogisticRegression(x_train, y_train, normalize_data=False)
theta, cost_histories = classifier.train()
labels = classifier.unique_labels

# plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
# plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
# plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
# plt.show()

predictions = classifier.predict(x_test)
precision = np.sum(y_test == predictions) / y_test.shape[0]
# print(f'y test: {y_test}\ny_predict: {predictions}')
print(f'precision: {precision}')

x_min = np.min(x_test[:, 0])
x_max = np.max(x_test[:, 0])
y_min = np.min(x_test[:, 1])
y_max = np.max(x_test[:, 1])
samples = 150
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

Z_SETOSA = np.zeros((samples, samples))
Z_VERSICOLOR = np.zeros((samples, samples))
Z_VIRGINICA = np.zeros((samples, samples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        prediction = classifier.predict(data)[0][0]
        if prediction == 'SETOSA':
            Z_SETOSA[x_index][y_index] = 1
        elif prediction == 'VERSICOLOR':
            Z_VERSICOLOR[x_index][y_index] = 1
        elif prediction == 'VIRGINICA':
            Z_VIRGINICA[x_index][y_index] = 1

for iris_type in iris_types:
    plt.scatter(
        x_test[(y_test == iris_type).flatten(), 0],
        x_test[(y_test == iris_type).flatten(), 1],
        label=iris_type
    )

plt.contour(X, Y, Z_SETOSA)
plt.contour(X, Y, Z_VERSICOLOR)
plt.contour(X, Y, Z_VIRGINICA)
plt.show()
