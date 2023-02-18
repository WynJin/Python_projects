# -*- codeding = utf-8 -*-
# @Time : 2023/2/17 17:32
# @Author : 怀德
# @File : FCMTEST.py.py
# @Software: PyCharm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from fcmeans import FCM

def clean_unnormal(f,v_max,a_max):
    for col in ['x_v', 'y_v']:
        f[col] = f[col].replace("none",0).replace('nan',0).astype(np.float64).apply(lambda x: 0 if abs(x)>v_max else x)
    for col in ['x_a', 'y_a']:
        f[col] = f[col].replace("none",0).replace('nan',0).astype(np.float64).apply(lambda x: 0 if abs(x)>a_max else x)


import random

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
from sklearn import datasets
import pandas as pd
m=2



class FCM_self:
    def __init__(self, data, clust_num,iter_num=10):
        self.data = data
        self.cnum = clust_num
        self.sample_num=data.shape[0]
        self.dim = data.shape[-1]  # 数据最后一维度数
        Jlist=[]   # 存储目标函数计算值的矩阵

        U = self.Initial_U(self.sample_num, self.cnum)
        for i in range(0, iter_num): # 迭代次数默认为10
            C = self.Cen_Iter(self.data, U, self.cnum)
            U = self.U_Iter(U, C)
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心",C)
            J = self.J_calcu(self.data, U, C)  # 计算目标函数
            Jlist = np.append(Jlist, J)
        self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
        self.Clast = C    # 最后的类中心矩阵
        self.Jlist = Jlist  # 存储目标函数计算值的矩阵

    # 初始化隶属度矩阵U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)  # sample_num为样本个数, cluster_n为分类数
        row_sum = np.sum(U, axis=1)  # 按行求和 row_sum: sample_num*1
        row_sum = 1 / row_sum    # 该矩阵每个数取倒数
        U = np.multiply(U.T, row_sum)  # 确保U的每列和为1 (cluster_n*sample_num).*(sample_num*1)
        return U   # cluster_n*sample_num

    # 计算类中心
    def Cen_Iter(self, data, U, cluster_n):
        c_new = np.empty(shape=[0, self.dim])  # self.dim为样本矩阵的最后一维度
        for i in range(0, cluster_n):          # 如散点的dim为2，图片像素值的dim为1
            u_ij_m = U[i, :] ** m  # (sample_num,)
            sum_u = np.sum(u_ij_m)
            ux = np.dot(u_ij_m, data)  # (dim,)
            ux = np.reshape(ux, (1, self.dim))  # (1,dim)
            c_new = np.append(c_new, ux / sum_u, axis=0)   # 按列的方向添加类中心到类中心矩阵
        return c_new  # cluster_num*dim

    # 隶属度矩阵迭代
    def U_Iter(self, U, c):
        for i in range(0, self.cnum):
            for j in range(0, self.sample_num):
                sum = 0
                for k in range(0, self.cnum):
                    temp = (np.linalg.norm(self.data[j, :] - c[i, :]) /
                            np.linalg.norm(self.data[j, :] - c[k, :])) ** (
                                2 / (m - 1))
                    sum = temp + sum
                U[i, j] = 1 / sum

        return U

    # 计算目标函数值
    def J_calcu(self, data, U, c):
        temp1 = np.zeros(U.shape)
        for i in range(0, U.shape[0]):
            for j in range(0, U.shape[1]):
                temp1[i, j] = (np.linalg.norm(data[j, :] - c[i, :])) ** 2 * U[i, j] ** m

        J = np.sum(np.sum(temp1))
        print("目标函数值:%.2f" %J)
        return J


    # 打印聚类结果图
    def plot(self):

        mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 聚类点的颜色及形状

        if self.dim == 2:
            #第一张图
            plt.subplot(221)
            plt.plot(self.data[:, 0], self.data[:, 1],'ob',markersize=2)
            plt.title('未聚类前散点图')

            #第二张图
            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], self.data[j:j + 1, 1], mark[i],
                         markersize=2)
                j += 1

            plt.plot(self.Clast[:, 0], self.Clast[:, 1], 'k*', markersize=7)
            plt.title("聚类后结果")

            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图",)

            plt.show()
        elif self.dim==1:

            plt.subplot(221)
            plt.title("聚类前散点图")
            for j in range(0, self.data.shape[0]):
                plt.plot(self.data[j, 0], 'ob',markersize=3)  # 打印散点图

            plt.subplot(222)
            j = 0
            for i in self.label:
                plt.plot(self.data[j:j + 1, 0], mark[i], markersize=3)
                j += 1

            plt.plot([0]*self.Clast.shape[0],self.Clast[:, 0], 'k*',label='聚类中心',zorder=2)
            plt.title("聚类后结果图")
            plt.legend()
            # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()

        elif self.dim==3:
            # 第一张图
            fig = plt.figure()
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.scatter(self.data[:, 0], self.data[:, 1],self.data[:,2], "b")
            ax1.set_xlabel("X 轴")
            ax1.set_ylabel("Y 轴")
            ax1.set_zlabel("Z 轴")
            plt.title("未聚类前的图")

            # 第二张图
            ax2 = fig.add_subplot(222, projection='3d')

            j = 0

            for i in self.label:
                ax2.plot(self.data[j:j+1, 0], self.data[j:j+1, 1],self.data[j:j+1,2], mark[i],markersize=5)
                j += 1
            ax2.plot(self.Clast[:, 0], self.Clast[:, 1], self.Clast[:, 2], 'k*', label='聚类中心', markersize=8)

            plt.legend()


            ax2.set_xlabel("X 轴")
            ax2.set_ylabel("Y 轴")
            ax2.set_zlabel("Z 轴")
            plt.title("聚类后结果")
            # # 第三张图
            plt.subplot(212)
            plt.plot(self.Jlist, 'g-', )
            plt.title("目标函数变化图", )
            plt.show()

def plot_3D(data,label,Clast,x,y,z):
    mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 聚类点的颜色及形状
    # 第一张图
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], "b")
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    ax1.set_zlabel(z)
    plt.title("未聚类前的图")

    # 第二张图
    ax2 = fig.add_subplot(222, projection='3d')

    j = 0

    for i in label:
        ax2.plot(data[j:j + 1, 0], data[j:j + 1, 1], data[j:j + 1, 2], mark[i], markersize=5)
        j += 1
    ax2.plot(Clast[:, 0], Clast[:, 1], Clast[:, 2], 'k*', label='聚类中心', markersize=8)

    plt.legend()

    ax2.set_xlabel(x)
    ax2.set_ylabel(y)
    ax2.set_zlabel(z)
    plt.title("聚类后结果")
    # # 第三张图
    # plt.subplot(212)
    # plt.plot(Jlist, 'g-', )
    # plt.title("目标函数变化图", )
    plt.show()

def plot_2D(data,label,Clast,x,y):
    fea = {2: '平均速度', 3: '平均加速度', 4: '平均减速度', 5: '横向加速度', 6: '平均加加速度', 7: '加速度标准差', 8: '速度标准差', 9: '加加速度标准差'}
    data[:,0] = data[:,x]
    data[:,1] = data[:,y]
    mark = ['or', 'ob', 'og', 'om', 'oy', 'oc']  # 聚类点的颜色及形状
    # 第一张图
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(data[:, 0], data[:, 1], 'ob', markersize=2)
    ax1.set_xlabel(fea[x])
    ax1.set_ylabel(fea[y])
    plt.title("未聚类前的图")

    # 第二张图
    ax2 = fig.add_subplot(222)

    j = 0
    for i in label:
        ax2.plot(data[j:j + 1, 0], data[j:j + 1, 1], mark[i],
                 markersize=2)
        j += 1

    ax2.plot(Clast[:, 0], Clast[:, 1], 'k*', markersize=7)
    ax2.legend()
    ax2.set_xlabel(fea[x])
    ax2.set_ylabel(fea[y])
    plt.title("聚类后结果")
    # # 第三张图
    # plt.subplot(212)
    # plt.plot(Jlist, 'g-', )
    # plt.title("目标函数变化图", )
    plt.show()

def es():
    f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc.csv")
    print(len(f))
    # f = clean_unnormal(f, 120, 30)
    print(f)
    df = f[['x_a', 'y_v', 'y_a']]
    data = df.values
    a = FCM_self(data, 3,5)  # 将数据分为三类
    a.plot()  # 打印结果图
def cal_score(f,labels):
    ss = metrics.silhouette_score(f, labels, metric='euclidean', sample_size=None, random_state=None)
    ch = metrics.calinski_harabasz_score(f, labels)
    print("轮廓系数为" + str(ss))
    print("轮廓系数为" + str(ch))


def test1():
    # f1=pd.read_csv('./fact3.csv').to_numpy()
    ''' 2-9列特征为
    平均速度,平均加速度,平均减速度,横向加速度,平均加加速度,加速度标准差,速度标准差,加加速度标准差
    '''
    fea = {2:'平均速度',3:'平均加速度',4:'平均减速度',5:'横向加速度',6:'平均加加速度',7:'加速度标准差',8:'速度标准差',9:'加加速度标准差'}
    f2 = pd.read_csv(r'D:\Scientific_Research\NGSIM_prediction\验收\us101.csv', header=None).to_numpy()
    #
    # X = f2[:, 2:6]
    # f2=pd.read_csv('./us101_with_label3.csv',header=None).to_numpy()
    # labels=f2[:,-1]

    # gmm=GaussianMixture(n_components=3,covariance_type='full', random_state=0)
    # gmm.fit(X)

    # labels=KMeans(n_clusters=3,init=np.array([[43.00935855,3.58866541,-3.26825081, 8.15652813],
    #                               [49.48522346, 3.8706381,  -3.37473873,  9.75866983],
    #                               [36.60611517,  3.35846602, -3.30020614,  7.72648037]]),max_iter=500).fit_predict(X)
    X1 = f2[:, 6:10]
    # print("手搓FCM------")
    # # data = X.values
    # a = FCM_self(X1, 3, 20)  # 将数据分为三类
    # a.plot()  # 打印结果图
    # labels = a.label  # 最后的类中心矩阵
    # cal_score(f2[:, 2:5], labels)

    print ("调库FCM------")
    # 调库
    fcm = FCM(n_clusters=3)
    fcm.fit(X1)

    # labels = gmm.predict(X)
    labels = fcm.predict(X1)
    center = fcm.centers
    print(center)

    cal_score( X1, labels)
    # plot_3D( X1, labels,center,'平均加速度','平均减速度','横向加速度')# '横向加速度','平均加加速度','加速度标准差',8:'速度标准差',9:'加加速度标准差'
    plot_3D( X1, labels,center,'平均加加速度','加速度标准差','速度标准差')# '横向加速度','平均加加速度','加速度标准差',8:'速度标准差',9:'加加速度标准差'
    plot_2D(f2, labels,center,4,5)
    # plot_2D(f2, labels,center,2,3)
    # plot_2D(f2, labels,center,3,5)
    plot_2D(f2, labels,center,6,7)
    plot_2D(f2, labels,center,8,9)
    plot_2D(f2, labels,center,6,9)
    # 储存分类
    # f3 = np.column_stack((f2[:, 2:], labels))
    # np.savetxt("./us101_with_label_fcm.csv", f3, fmt="%f", delimiter=",")
    #


if __name__ == '__main__':

    test1()

