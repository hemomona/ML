#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : naive_bayes.py
# Time       ：2024/7/20 21:43
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
"""
import math
import os
import re
import string
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

DIR = "enron"
target_names = ['ham', 'spam']  # normal, spam
stopwords = set(open('stopwords.txt', 'r').read().splitlines())


def get_data(folder: str) -> tuple[list[str], list[int]]:  # data, target
    subfolders = ['enron%d' % i for i in range(1, 7)]
    data, target = [], []  # data store emails, target 1 for spam, 0 for ham

    for subfolder in subfolders:
        spam_folder = os.path.join(folder, subfolder, 'spam')
        spam_file_paths = [os.path.join(spam_folder, name) for name in os.listdir(spam_folder)]
        for spam_file_path in spam_file_paths:
            with open(spam_file_path, encoding="latin-1") as f:
                data.append(f.read())
                target.append(1)  # 1 for spam

        ham_folder = os.path.join(folder, subfolder, 'ham')
        ham_file_paths = [os.path.join(ham_folder, name) for name in os.listdir(ham_folder)]
        for ham_file_path in ham_file_paths:
            with open(ham_file_path, encoding="latin-1") as f:
                data.append(f.read())
                target.append(0)  # 0 for ham
    return data, target


def preprocess(text: str) -> list[str]:
    t = text.lower()  # lower
    t = re.sub(f"[{string.punctuation}]", ' ', t)  # punctuation -> blank
    t = [word for word in t.split() if word not in stopwords]  # delete words in stopwords, which are acceptable words?
    return t


class NaiveBayesClassifier():
    def __init__(self):
        self.vocabulary = set()  # unordered, no repeated elements
        # defaultdict(int) returns 0 when the key is not existed
        self.class_total = defaultdict(int)  # the number of emails in spam or ham
        self.word_total = defaultdict(int)  # the number of words in spam or ham
        self.word_given_class = defaultdict(lambda: defaultdict(int))  # the number of a specified word in spam or ham

    def fit(self, X: list[str], y: list[int]):
        for text, label in zip(X, y):
            words = preprocess(text)
            self.class_total[label] += 1
            for word in words:
                self.vocabulary.add(word)
                self.word_total[label] += 1
                self.word_given_class[label][word] += 1

    def predict(self, X):
        log_priors = {}  # store the log of the probability of a class, avoid underflow
        for c in self.class_total.keys():
            log_priors[c] = math.log(self.class_total[c] / sum(self.class_total.values()))  # log(class prior prob)

        predictions = []
        for text in X:  # for each email
            words = preprocess(text)
            log_probs = {}
            for c in self.class_total.keys():  # for each class
                log_probs[c] = log_priors[c]  # log(class prior prob)
                for word in words:  # for each word in the email
                    if word in self.vocabulary:
                        # accuracy: 0.92
                        # log_probs[c] += np.log(self.word_given_class[c][word] / self.word_total[c])
                        # math will break out if numerator is 0, so we add all words into a class once to handle this!!!
                        # accuracy: 0.99
                        log_probs[c] += math.log(
                            (self.word_given_class[c][word] + 1) / (self.word_total[c] + len(self.vocabulary)))
            predictions.append(max(log_probs, key=log_probs.get))

        return predictions


X, y = get_data(DIR)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = NaiveBayesClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(np.array(predictions) == np.array(y_test)) / len(y_test)
print(f'Accuracy: {accuracy:.2f}')
