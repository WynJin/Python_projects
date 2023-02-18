# -*- codeding = utf-8 -*-
# @Time : 2022/5/31 22:37
# @Author : 怀德
# @File : KNN-MNIST.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import operator
import numpy as np
import matplotlib.pyplot as plt
from numpy import *

import warnings

warnings.filterwarnings('ignore')


#
def KNN_classify_getClassList(train, train_label, test, k, dis):
    assert dis == 'E' or dis == 'M', 'dis must E or M，E代表欧拉距离，M代表曼哈顿距离'
    num_test = test.shape[0]  # 测试样本的数量
    labellist = []
    '''
    使用欧拉公式作为距离度量
    '''
    if (dis == 'E'):
        for i in range(num_test):  # train: 训练测试的特征数据,x_train: 训练测试的标签数据,Y_test: 测试的特征数据
            # 实现欧拉距离公式
            dist = np.power((np.power(train[:] - test[i], 2)).sum(axis=1), 0.5)  # 欧氏距离
            # distances = np.sqrt(np.sum(((train - np.tile(Y_test[i], (train.shape[0], 1))) ** 2), axis=1))
            nearest_k = np.argsort(dist)  # 距离由小到大进行排序，并返回index值
            topK = nearest_k[:k]  # 选取前k个距离
            votes = {}
            for m in topK:
                votes.setdefault(train_label[m], 0)
                votes[train_label[m]] += 1
            labellist.append(max(votes, key=votes.get))
        return np.array(labellist)
    # 使用曼哈顿公式作为距离度量
    if (dis == 'M'):
        for i in range(num_test):
            # 按照列的方向相减
            dist = np.sum(np.abs(train[:, :-1] - test[i, :-1]), axis=1)
            nearest_k = np.argsort(dist)
            topK = nearest_k[:k]
            votes = {}
            for m in topK:
                votes.setdefault(train_label[m], 0)
                votes[train_label[m]] += 1
            labellist.append(max(votes, key=votes.get))
        return np.array(labellist)


def Accuracy_rate_vs_k(train, train_label, test, dis):
    x = []
    y = []
    for i in range(1, 30):  # 测试1到30邻接点的影响
        x.append(i)
        test_pred = KNN_classify_getClassList(train, train_label, test, i, dis)
        num_correct = np.sum(test_pred == test)
        acc = float(num_correct) / num_test
        y.append(acc)
    plt.title('Acr VS K ', fontsize=20, fontname='Times New Roman')
    plt.xlabel("K", fontsize="large")
    plt.ylabel("Accurate rate", fontsize="large")
    plt.plot(x, y)
    plt.show()


# 载入数据集
train_dataset = dsets.MNIST(root='./data',  # 根目录
                            train=True,  # 选择训练集
                            transform=None,  # 不考虑使用任何数据预处理
                            download=False)
# 载入验证集
test_dataset = dsets.MNIST(root='./data',
                           train=False,  # 选择测试集
                           transform=None,  # 无数据预处理
                           download=False)

# 分批次（batch）训练
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 将数据打乱
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
# 预处理
train = train_loader.dataset.train_data.numpy()  # 获取训练集图像，转为numpy数组
train = train.reshape(train.shape[0], 28 * 28)
# 需要reshape之后才能放入knn分类器,图像转成一个28*28的维度train.shape[0]个样本的矩阵
train_label = train_loader.dataset.train_labels.numpy()  # 训练集获取标签

# 验证集
test = test_loader.dataset.test_data[:500].numpy()
test = test.reshape(test.shape[0], 28 * 28)
test_label = test_loader.dataset.test_labels[:500].numpy()
num_test = test_label.shape[0]

test_label_pred = KNN_classify_getClassList(train, train_label, test, 1, 'M', )
num_correct = np.sum(test_label_pred == test_label)
acc = float(num_correct) / num_test
print('accuracy rate: %f' % acc)

Accuracy_rate_vs_k(train, train_label, test, 'M')

# k: 选取的最近k个目标
# dis: 'E'
# 欧拉;
# 'M'
# 曼哈顿
