# -*- codeding = utf-8 -*-
# @Time : 2022/5/21 12:13
# @Author : 怀德
# @File : KNN-projects.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from numpy import *


def create_LabelArg(labels):
    """ 读取数据元素排序前的下标号，结果放入列表"""
    arglist = []
    for i in range(len(labels)):
        path = "../dataanalysis/label/" + labels[i]  # 文件路径拼接
        datalabel = np.loadtxt(path, dtype=np.float64, delimiter="\n")
        labelargs = np.argsort(abs(datalabel))[::-1]  # 取绝对值， argsort记录排序前的下标号
        arglist.append(labelargs)
    return arglist


def To_Sub(train, arglist, d):
    """根据argList选择 dim个维度降维"""
    return np.array([train[i][arglist[j]] for i in range(train.shape[0]) for j in range(d)]).reshape(train.shape[0], d)


def gaussian(dist, sigma=10):
    """输入距离返回其权重"""
    weight = sqrt(-(dist - b) ** 2 / (2 * sigma ** 2))
    return weight


def KNN_classify_getErr(train, test, k, dis):  # di为决策距离
    """ 根据k个近邻点进行分类，返回预测错误率 """
    assert dis == 'E' or dis == 'M', '距离决策方式必须为 E 或 M，E代表欧氏距离，M代表曼哈顿距离'
    test_num = test.shape[0]
    train_num = train.shape[0]
    score = 0
    if dis == 'E':  # 欧氏距离
        for i in range(test_num):
            dist = np.power((np.power(train[:] - test[i], 2)).sum(axis=1), 0.5)
            topK = np.argsort(dist)[:k]  # 选距离最接近的k个样本
            votes = {}
            for m in topK:
                label = classify(m, train_num)
                votes.setdefault(label, 0)
                votes[label] += 1
            if classify(i, test_num) == max(votes, key=votes.get):
                score += 1
    elif dis == 'M':  # 曼哈顿距离
        for i in range(test_num):
            dist = np.sum(np.abs(train[:, :-1] - test[i, :-1]), axis=1)  # 曼哈顿距离
            topK = np.argsort(dist)[:k]  # 选距离最接近的k个样本
            votes = {}
            for m in topK:
                label = classify(m, train_num)
                votes.setdefault(label, 0)
                votes[label] += 1
            if classify(i, test_num) == max(votes, key=votes.get):
                score += 1
    err = 1 - score / test_num
    return err


def classify(index, num):  # 只有两个分类
    return index < num / 2


def Errorrate_vs_k(trainsub, testsub, dis):
    """ 比较错误率和K的关系"""
    x = []
    y = []
    for i in range(1, 30):  # 一般K值取1-20，此处测试1到30邻接点
        x.append(i)
        err = KNN_classify_getErr(trainsub, testsub, i, dis)
        y.append(err)
    plt.title('Err VS K when dimension={} dis = {}'.format(testsub.shape[1], dis), fontsize=20,
              fontname='Times New Roman')
    plt.xlabel("K", fontsize="large")
    plt.ylabel("error rate", fontsize="large")
    plt.plot(x, y)
    plt.show()


def Errorrate_VS_dim(train, test, argList, k, dis):
    dim = [i for i in range(200, 2000, 100)]
    errlist = []
    for d in dim:
        TrainSub = To_Sub(train, argList, d)
        TestSub = To_Sub(test, argList, d)
        err = KNN_classify_getErr(TrainSub, TestSub, k, dis)
        errlist.append(err)
    plt.plot(dim, [i for i in errlist], color='m')
    plt.title('Err VS Dim when k={} dis={}'.format(k, dis), fontsize=20, fontname='Times New Roman')
    plt.xlabel("Dimension", fontsize="large")
    plt.ylabel("Errorrate", fontsize="large")
    plt.show()


if __name__ == '__main__':
    label = ['MTL_Male.dat', 'CMTL_Male.dat', 'CEMTL_Male.dat']
    argList = create_LabelArg(label)  # 第一题排序记录
    # ---------------------------------------------第二第三题
    TrainSample = np.loadtxt("../dataanalysis/train/MTL_Male_train.dat", dtype=int, delimiter=",")
    TestSample = np.loadtxt("../dataanalysis/test/MTL_Male_test.dat", dtype=int, delimiter=",")
    # 行列数
    dim = 400
    MTL_TrainSub = To_Sub(TrainSample, argList[2], dim)
    MTL_TestSub = To_Sub(TestSample, argList[2], dim)
    print("TrainSample 降维后矩阵行列数:", MTL_TrainSub.shape)
    # ---------------------------------------------第四题
    k = 5
    dis_statitic = 'E'  # 决策距离方式
    print(KNN_classify_getErr(MTL_TrainSub, MTL_TestSub, k, dis_statitic))  #
    Errorrate_vs_k(MTL_TrainSub, MTL_TestSub, dis_statitic)#
    Errorrate_VS_dim(TrainSample, TestSample, argList[2], k, dis_statitic)
