#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : decisiontree.py
# Time       ：2024/7/15 19:40
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# Description：
silly list in python, = will refer to the original list
"""
from math import log
from matplotlib import pyplot as plt


def create_dataset():
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataset, labels


def majority_count(label_list):  # input list of labels
    class_count = {}
    for y in label_list:
        if y not in class_count.keys():
            class_count[y] = 0
        class_count[y] += 1
    class_count = sorted(class_count.items(), key=lambda d: d[0], reverse=True)
    return class_count[0][0]  # return the value of the most label


def cal_entropy(dataset):
    num_examples = len(dataset)
    label_count = {}
    for example in dataset:
        y = example[-1]  # even pseudo split dataset for measuring info gain will generate n*1 2D array
        if y not in label_count.keys():
            label_count[y] = 0
        label_count[y] += 1

    entropy = 0
    for key in label_count:
        p = float(label_count[key]) / num_examples
        entropy -= p * log(p, 2)  # -
    return entropy


def choose_best_feature(dataset):  # input dataset
    num_features = len(dataset[0]) - 1  # - column of labels
    base_entropy = cal_entropy(dataset)
    best_info_gain = 0
    best_feature_index = -1
    for i in range(num_features):
        feature_values = [example[i] for example in dataset]
        unique_values = set(feature_values)
        feature_entropy = 0  # record the entropy after being classified by this feature
        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)  # then len(sub_dataset[0]) might == 1
            weight = float(len(sub_dataset)) / len(dataset)
            feature_entropy += weight * cal_entropy(sub_dataset)
        info_gain = base_entropy - feature_entropy
        if info_gain > best_info_gain:  # record the feature index which makes the biggest information gain
            best_info_gain = info_gain
            best_feature_index = i
    return best_feature_index


def split_dataset(dataset, feature_index, value):
    rest_dataset = []
    for example in dataset:
        if example[feature_index] == value:
            reduced_example = [e for e in example]
            reduced_example.pop(feature_index)
            rest_dataset.append(reduced_example)
    return rest_dataset  # return the dataset with feature_index == value and deleting this feature column


def create_tree(dataset, features, featurelabels):  # featurelabels record the feature name which has been used
    label_list = [example[-1] for example in dataset]
    if label_list.count(label_list[0]) == len(label_list):  # if it has been pure
        return label_list[0]  # just return one value

    if len(dataset[0]) == 1:  # if it has no feature
        return majority_count(label_list)  # return the value of the most label

    best_feature_index = choose_best_feature(dataset)  # column index
    best_feature_label = features[best_feature_index]  # column index -> label name
    featurelabels.append(best_feature_label)
    tree = {best_feature_label: {}}
    subfeatures = [f for f in features]  # python will directly change the original list if = was used !!!
    subfeatures.pop(best_feature_index)  # pop will delete the value at the index but return the value !!!

    feature_values = [example[best_feature_index] for example in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_dataset = split_dataset(dataset, best_feature_index, value)
        tree[best_feature_label][value] = create_tree(sub_dataset, subfeatures, featurelabels)  # recursive
    return tree


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, features = create_dataset()
    feature_labels = []
    tree = create_tree(dataset, features, feature_labels)
    createPlot(tree)
