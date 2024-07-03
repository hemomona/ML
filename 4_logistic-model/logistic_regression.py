#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : logistic_regression.py
# Time       ：2024/7/2 14:06
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid


class LogisticRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        data_processed, features_mean, features_deviation \
            = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        num_unique_labels = self.unique_labels.shape[0]
        self.theta = np.zeros((num_unique_labels, num_features))  # multi classification

    def train(self, num_iterations=1000):
        cost_histories = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            # theta = (num_unique_labels, num_features) current_initial_theta = (num_features, 1)
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            current_labels = (self.labels == unique_label).astype(float)
            current_theta, cost_history = LogisticRegression.gradient_descent(self.data, current_labels, current_initial_theta, num_iterations)
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)
        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, current_labels, current_initial_theta, num_iterations):
        cost_history = []
        num_features = data.shape[1]
        result = minimize(lambda current_theta: LogisticRegression.cost_function(data, current_labels, current_theta.reshape((num_features, 1))),
                          current_initial_theta.flatten(),  # !IMPORTANT
                          method='CG',
                          jac=lambda current_theta: LogisticRegression.gradient_step(data, current_labels, current_theta.reshape((num_features, 1))),
                          callback=lambda current_theta: cost_history.append(LogisticRegression.cost_function(data, current_labels, current_theta.reshape((num_features, 1)))),
                          options={'maxiter': num_iterations})
        if not result.success:
            raise ArithmeticError('can not minimize cost function ', result.message)
        optimized_theta = result.x
        return optimized_theta, cost_history

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))
        return predictions

    @staticmethod
    def cost_function(data, labels, theta):
        num_examples = data.shape[0]
        # # data = (num_examples, num_features), theta here = (num_features, 1)
        predictions = LogisticRegression.hypothesis(data, theta)  # (num_examples, 1)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels == 1]))
        y_isnot_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels == 0]))
        cost = (-1 / num_examples) * (y_is_set_cost + y_isnot_set_cost)
        return cost

    @staticmethod
    def gradient_step(data, labels, theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1 / num_examples) * np.dot(data.T, label_diff)  # (num_features, 1)
        return gradients.flatten()

    def predict(self, data):
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)  # (num_examples, num_features)
        max_prob_index = np.argmax(prob, axis=1)  # return index of max prob in rows
        class_prediction = np.empty(max_prob_index.shape, dtype=object)  # default is float64
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index == index] = label
        return class_prediction.reshape((data.shape[0], 1))
