# -*- codeding = utf-8 -*-
# @Time : 2023/1/29 1:14
# @Author : 怀德
# @File : svm0.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib

# 1.读取训练数据集
# data = pd.read_csv('./result11.csv')
# data = data[data['v_Vel'] != 0]
# data = data[data['v_Class'] == 2]
# x = data[['v_Vel', "v1", "v2", 'v3', 'v4', 'y1', 'y2', 'y3', 'y4']]
# y = data["label"]
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.6, random_state=0)
# print(type(y[0]))
# print(y)

#通过SVM预测，划分决策面时利用了等高线contour函数
def plot_svc_decision_function(data, label, feature1, feature2, ax=None):
    fea1 = data[:, feature1]
    fea2 = data[:, feature2]
    # 获取当前的子图，如果不存在，则创建新的子图
    if ax == None:
        ax = plt.gca()
    #绘图准备
    #获取平面上两条坐标轴的最大值和最小值
    xlim = [int(np.max(fea1)),int(np.min(fea1))]
    ylim = [int(np.max(fea2)),int(np.min(fea2))]
    # 要画决策边界，必须要有网格
    axisx = np.linspace(xlim[0], xlim[1], 50)
    axisy = np.linspace(ylim[0], ylim[1], 50)
    axisx, axisy = np.meshgrid(axisx, axisy)#使用meshgrid函数将两个一维向量转换为特征矩阵,将使用这里形成的特征矩阵二维数组作为contour函数中的X和Y
    xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
    # 将两个特征向量广播，获取x.shape*y.shape这么多点的横坐标和纵坐标
    # ravel()降维函数，把每一行都放在第一行，vstack能够将多个结构一致的一维数组按行堆叠起来
    # xy就是形成的网络，遍及在画布上密集的点
    # plt.scatter(xy[:,0],xy[:,1],s=1,cmap="rainbow")#理解函数meshgrid和vstack的作用
    #转为0 1标签
    np_label = np.ones(len(data)).astype('int32')
    for i in range(len(label)):
        if label[i] == '女':
                np_label[i] = 0
    np_label = np_label.astype('int32')#下方SVC(kernel = "linear").fit(X,y) 必须为int32数据类型

    #，通过fit计算出对应的决策边界
    X = list(zip(fea1,fea2))
    y = np_label
    clf = SVC(kernel = "linear").fit(X,y)
    Z = clf.decision_function(xy).reshape(axisx.shape) #预测结果
    print(Z.shape)
    #重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离。
    #然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
    plt.scatter(fea1,fea2,c=y,s=30,cmap=plt.cm.Spectral)
    ax = plt.gca() #获取当前的子图，如果不存在，则创建新的子图
    #画决策边界和平行于决策边界的超平面
    ax.contour(axisx,axisy,Z
               ,colors="k"
               ,levels=[-0.3,0,0.3] #画三条等高线，分别是Z为-1，Z为0和Z为1的三条线,可用于界定误差大小
               ,alpha=0.5
               ,linestyles=["--","-","--"])#画等高线
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# plot_svc_decision_function(data, label, 0, 1)

data = pd.read_csv(r'/data/NGSIM_Data/i-80/big_car_10min_plus_acc_useful_new.csv')

data1 = data
print(data)
# x = data[['v_Vel', "v1", "v2", 'v3', 'v4', 'y1', 'y2', 'y3', 'y4']]
# x = data[['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y','v_length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','O_Zone','D_Zone','Int_ID','Section_ID','Direction','Movement','Preceding','Following','Space_Headway','Time_Headway','Location','x_v','x_a','y_v','y_a']]
x = data[['Vehicle_ID',"v1","v2",'v3','v4','y1','y2','y3','y4']]
y = data["label"]
# # d3=data[data['label1']==2]
# # d3=d3.sample(n=338,random_state=132,axis=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# #特征工程
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
X = x_train
Y = y_train
# 3.初始化参数
W = 0.5  # 惯性因子
c1 = 0.2  # 学习因子
c2 = 0.5  # 学习因子
n_iterations = 10  # 迭代次数
n_particles = 50  # 种群规模


# 4.设置适应度值 输出分类精度得分，返回比较分类结果和实际测得值，可以把分类结果的精度显示在一个混淆矩阵里面
def fitness_function(position):  # 输出
    # 全局极值   svm分类器  核函数gamma  惩罚参数c
    svclassifier = SVC(kernel='rbf', gamma=position[0], C=position[1])
    # 参数gamma和惩罚参数c以实数向量的形式进行编码作为PSO的粒子的位置
    svclassifier.fit(X, Y)
    score = cross_val_score(svclassifier, X, Y, cv=10).mean()  # 交叉验证
    print('分类精度', score)  # 分类精度
    Y_pred = cross_val_predict(svclassifier, X, Y, cv=10)  # 获取预测值

    # 返回混淆函数，分类误差矩阵，分别是训练中的 测试中的 下面输出错误分类结果
    return confusion_matrix(Y, Y_pred)[0][1] + confusion_matrix(Y, Y_pred)[1][0], confusion_matrix(Y, Y_pred)[0][1] + \
           confusion_matrix(Y, Y_pred)[1][0]


# 5.粒子图
def plot(position, iteration):
    x = []
    y = []
    for i in range(0, len(particle_position_vector)):
        x.append(particle_position_vector[i][0])
        y.append(particle_position_vector[i][1])
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    plt.figure(dpi=600)
    plt.scatter(x, y)
    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
    plt.xlabel('gamma')  # 核函数
    plt.ylabel('C')  # 惩罚函数

    plt.title(f'第{iteration}代粒子群')

    plt.axis([0, 100, 0, 100], )
    plt.gca().set_aspect('equal', adjustable='box')  # #设置横纵坐标缩放比例相同，默认的是y轴被压缩了。
    # plt.savefig(f"./image/第{iteration}代.jpg")
    return plt.show()


# 6.初始化粒子位置，进行迭代
# 粒子位置向量
particle_position_vector = np.array(
    [np.array([random.random() * 100, random.random() * 100]) for _ in range(n_particles)])
pbest_position = particle_position_vector  # 个体极值等于最初位置
pbest_fitness_value = np.array([float('inf') for _ in range(n_particles)])  # 个体极值的适应度值
gbest_fitness_value = np.array([float('inf'), float('inf')])  # 全局极值的适应度值
gbest_position = np.array([float('inf'), float('inf')])
velocity_vector = ([np.array([0, 0]) for _ in range(n_particles)])  # 粒子速度
# 迭代更新
iteration = 0
while iteration < n_iterations:
    plot(particle_position_vector, iteration)  # 粒子具体位置
    for i in range(n_particles):  # 对每个粒子进行循环
        fitness_cadidate = fitness_function(particle_position_vector[i])  # 每个粒子的适应度值=适应度函数（每个粒子的具体位置）
        # print("粒子误差", i, "is (training, test)", fitness_cadidate, " At (gamma, c): ",
        # particle_position_vector[i])

        if (pbest_fitness_value[i] > fitness_cadidate[
            1]):  # 每个粒子的适应度值与其个体极值的适应度值(pbest_fitness_value)作比较，如果更优的话，则更新个体极值，
            pbest_fitness_value[i] = fitness_cadidate[1]
            pbest_position[i] = particle_position_vector[i]

        if (gbest_fitness_value[1] > fitness_cadidate[1]):  # 更新后的每个粒子的个体极值与全局极值(gbest_fitness_value)比较，如果更优的话，则更新全局极值
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

        elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position_vector[i]

    for i in range(n_particles):  # 更新速度和位置，更新新的粒子的具体位置
        new_velocity = (W * velocity_vector[i]) + (c1 * random.random()) * (
                pbest_position[i] - particle_position_vector[i]) + (c2 * random.random()) * (
                               gbest_position - particle_position_vector[i])
        new_position = new_velocity + particle_position_vector[i]
        particle_position_vector[i] = new_position

    iteration = iteration + 1

# 7.输出最终结果
print("全局最优点的位置是 ", gbest_position, "在第", iteration, "步迭代中（训练集，测试集）错误个数:",
      fitness_function(gbest_position))
