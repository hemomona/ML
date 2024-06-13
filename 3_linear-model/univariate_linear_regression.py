#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : univariate_linear_regression.py
# Time       ：2024/6/12 20:41
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('data/world-happiness-report-2017.csv')

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# make sure that data is a column vector rather than a list
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# plt.scatter(x_train, y_train, label='Train Data')
# plt.scatter(x_test, y_test, label='Test Data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.legend()
# plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
theta, cost_history = linear_regression.train(learning_rate, num_iterations)

print('initial loss: ', cost_history[0])
print('last loss: ', cost_history[-1])

# plt.plot(range(num_iterations), cost_history)
# plt.xlabel('Iter')
# plt.ylabel('Cost')
# plt.title('GD')
# plt.show()

prediction_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), prediction_num).reshape(prediction_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train Data')
plt.scatter(x_test, y_test, label='Test Data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.legend()
plt.show()
