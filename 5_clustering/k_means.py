#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : k_means.py
# Time       ：2024/7/13 9:50
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        # initialize random centroids
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # find the closest centroid id of every example
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)
            # update centroids
            centroids = KMeans.centroids_update(self.data, self.num_clusters, closest_centroids_ids)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters], :]
        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        for example_index in range(num_examples):
            # temporary variable to store the distance between an example and centroids
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    @staticmethod
    def centroids_update(data, num_clusters, closest_centroids_ids):
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroid_id in range(num_clusters):
            # assign all ids of example whose closest centroid is this centroid to closest_ids
            example_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[example_ids.flatten(), :], axis=0)  # axis=0 return 1*n matrix
        return centroids


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../3_linear-model/data/iris.csv")
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
x_axis = 'petal_length'
y_axis = 'petal_width'

# plt.subplot(1, 2, 1)
# for iris_type in iris_types:
#     plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.scatter(data[x_axis], data[y_axis])
# plt.show()

num_clusters = 3
max_iterations = 50
data = data[[x_axis, y_axis]].values.reshape((-1, 2))

k_means = KMeans(data, num_clusters)
centroids, closest_centroids_ids = k_means.train(max_iterations)

plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(daa[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type], label=iris_type)
plt.legend()

plt.subplot(1, 2, 2)
for centroid_id, centroid in enumerate(centroids):
    example_ids = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][example_ids], data[y_axis][example_ids], label=centroid_id)
    plt.scatter(centroid[0], centroid[1], marker='o')
plt.legend()
plt.show()
