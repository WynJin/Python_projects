# -*- codeding = utf-8 -*-
# @Time : 2023/2/7 18:00
# @Author : 怀德
# @File : dbscantest.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.offline as pyo
# # pyo.init_notebook_mode()
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# from warnings import filterwarnings
# filterwarnings('ignore')

'''
BSCAN聚类的簇的尺寸问题

关于DBSCAN聚类的尺寸问题是本人在工作中了解到的相关信息。大致算法流程是：

前提条件是DBSCAN将一帧的原始数据（经过距离、速度和角度FFT之后得到的数据）经过处理后已经聚成了一个或多个簇。

1）根据原始数据中保存的一帧数据，先计算簇中x方向和y方向距离的据平均值，记为xcenter,ycenter，作为聚类尺寸的中心点。

2）然后初始化尺寸大小为xsize=0,ysize=0。

3）选择簇中的第一个点作为核心点，计算该点到中心点（xcenter,ycenter）的距离，记为temp_x,temp_y。然后将其与xsize,ysize进行比较。取最大值存为新的xsize,ysize。

4）之后依次比较簇中其他点到中心点（xcenter,ycenter）的距离，取其中最大的值作为新的xsize,ysize。从而作为DBSCAN聚类的簇的尺寸。

5）另外如果得到的xsize,ysize有一个是为0的，那么会将xsize,ysize等于一个固定的值。
'''
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import dbscan  # 今天使用的新算法包
import numpy as np
import sklearn.preprocessing as pre
from sklearn import metrics
from collections import Counter
from sklearn.cluster import DBSCAN

def dbscan_test():
    X, _ = datasets.make_moons(500, noise=0.1, random_state=1)  # 单单用x=。。。的话最后面还会有一个类别的数组
    df = pd.DataFrame(X, columns=['x', 'y'])
    df.plot.scatter('x', 'y', s=200, alpha=0.5, c='green')

    # 接下来我们就用dbscan算法来进行聚类计算
    # eps为邻域半径， min_samples为最少样本量
    core_samples, cluster_ids = dbscan(X, eps=0.2, min_samples=20)
    # cluster_ids中 -1表示对应的点为噪声
    df = pd.DataFrame(np.c_[X, cluster_ids], columns=['x', 'y', 'cluster_id'])
    # np.c 中的c 是 column(列)的缩写，就是按列叠加两个矩阵，就是把两个矩阵左右组合，要求行数相等。
    df['cluster_id'] = df['cluster_id'].astype('i2')  # 变整数
    print(df)

    df.plot.scatter('x', 'y', s=200, c=list(df['cluster_id']), cmap='Reds', colorbar=False, alpha=0.6, title='DBSCAN')
    plt.show()


def clean_unnormal(f,v_max,a_max):
    for col in ['x_v', 'y_v']:
        f[col] = f[col].replace("none",0).replace('nan',0).astype(np.float64).apply(lambda x: 0 if abs(x)>v_max else x)
    for col in ['x_a', 'y_a']:
        f[col] = f[col].replace("none",0).replace('nan',0).astype(np.float64).apply(lambda x: 0 if abs(x)>a_max else x)

    return f

def plot2D(f,feature_x,feature_y):
    plt.figure()
    s1 = plt.scatter(f[feature_x],f[feature_y], c='b', marker='o', label='car')
    # s2 = plt.scatter(female_fea1, female_fea2, c='b', marker='x', label='女')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(loc='best')
    plt.show()

def plot_seaboren(DB_df):
    # plt.xticks(np.arange(-30, 30, 2))
    # plt.yticks(np.arange(-40, 120, 5))
    # seaborn整体风格设置
    sns.set_style("darkgrid")
    sns.set_theme(style="ticks")  # 设置刻度
    sns.despine()  # 去掉上面和右边的轴
    sns.set_theme(style="white")
    sns.set_context("notebook", font_scale=0.7, rc={"lines.linewidth": 1.2})  # 设置label字体大小、线条粗细等
    # 调色板
    palette = sns.hls_palette(2, l=0.7, s=0.8)  # 个数、亮度、饱和度
    # palette = sns.color_palette("Set2", n_colors=2, desat=1)
    # palette = sns.xkcd_palette(["sky blue", "periwinkle"])palette = sns.xkcd_palette(["sky blue", "periwinkle"])

    # x轴的刻度值设置
    fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
    # ax.set_xticks([i + 1 for i in range(100)])
    ax.set_xticks(range(-30, 30, 2))
    # seaborn 设置刻度
    # ax = sns.heatmap(df, vmin=0., vmax=1, linewidths=.05, cbar_kws={'label': 'x_a'})

    # ax1.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))

    # ax = plt.gca() # 坐标轴的移动
    # from matplotlib.ticker import FixedLocator, MaxNLocator
    #
    # ax.xaxis.set_major_locator(MaxNLocator(steps=[2]))
    # ax.xaxis.set_minor_locator(FixedLocator(range(-30, 31)))
    # ax.yaxis.set_major_locator(MaxNLocator(steps=[2]))
    # ax.yaxis.set_minor_locator(FixedLocator(range(-40, 120)))

    # 绘制

    sns.set_palette("dark", 5)
    sns.scatterplot(x='x_a', y='y_a', hue='driving_style', data=DB_df, palette="Set1")
    plt.show()

import itertools
def DBSCAN_process(f,par_list):
    # 3.标准化

    DB_df = f
    DB_df_standardize = pre.StandardScaler().fit_transform(DB_df)
    DB_df_standardize = pd.DataFrame(data=DB_df_standardize, columns=list(DB_df.columns))  # 训练dbscan模型

    df_list = []
    for par in par_list:
        DB_df = f
        DB = DBSCAN(eps=par[0],min_samples = par[1])
        # 若类别数太多，需要增大eps领域大小，默认是0.5,似乎是关键，或者减少 min_samples 默认为5
        DB.fit(DB_df_standardize)
        # 将新标签组合到原来的数据框里面
        DB_label = pd.DataFrame(DB.labels_, columns=['driving_style'])
        DB_df = pd.concat([DB_df, DB_label], axis=1)
        df_list.append(DB_df)

    return df_list

def classify_style(f,df):
    cars = df['Vehicle_ID'].unique()
    # aggressive = f[df['driving_style'].isin([0])]
    # medium = f[df['driving_style'].isin([1])]
    # conservative = f[df['driving_style'].isin([-1])]

    # df1 = df.groupby['Vehicle_ID'].count().reset_index()
    # print(df1)

    styles = {-1: 'aggressive', 0: 'medium', 1: 'conservative'}

    for car in cars:
        car_inf = df[df['Vehicle_ID'] == car]
        car_c3 = Counter(car_inf['driving_style']).most_common(1)
        print(car_c3)
        print("Vehicle_ID为: "+ str(car)+ "的车辆的种类为: " +str(styles[car_c3[0][0]]))

        # # 转化为字典类型
        # print("字典类型: ", dict(car_inf.value_counts()))
        # for key, value in car_inf.value_counts().items():
        #     print("Key: ", key, " value: ", value)

def ss(df): #确定合适的epsilon级别
    plt.figure(figsize=(10, 5))
    nn = NearestNeighbors(n_neighbors=5).fit(df)
    distances, idx = nn.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()

def estimate_coffecient(df):
    labels = df["driving_style"]
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, labels))

def pca_with_eandm(pca_df,pca_eps_values,pca_min_samples):
    DB_df = pca_df
    DB_df_standardize = pre.StandardScaler().fit_transform(DB_df)
    pca_df = pd.DataFrame(data=DB_df_standardize, columns=list(DB_df.columns))  # 训练dbscan模型

    # df_list = DBSCAN_process(df, par_list)
    # for DB_df in df_list:
    #     print("-------")
    #     estimate_coffecient(DB_df)
    #     plot_seaboren(DB_df)
    # pca_eps_values = np.arange(0.8,1.5,0.1)
    # pca_min_samples = np.arange(2,15)
    pca_dbscan_params = list(itertools.product(pca_eps_values, pca_min_samples))
    pca_no_of_clusters = []
    pca_sil_score = []
    pca_epsvalues = []
    pca_min_samp = []
    pca_eps_min_df = []
    for p in pca_dbscan_params:
        pca_dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(pca_df)
        pca_epsvalues.append(p[0])
        pca_min_samp.append(p[1])
        pca_no_of_clusters.append(len(np.unique(pca_dbscan_cluster.labels_)))
        pca_sil_score.append(silhouette_score(pca_df, pca_dbscan_cluster.labels_))#计算轮廓系数
        pca_eps_min = list(zip(pca_no_of_clusters, pca_sil_score, pca_epsvalues, pca_min_samp))
        pca_eps_min_df = pd.DataFrame(pca_eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
        print(pca_eps_min_df)
    print("--------")
    print(pca_eps_min_df[pca_eps_min_df['no_of_clusters'] == 3])
    pca_eps_min_df = pca_eps_min_df[pca_eps_min_df['no_of_clusters'] == 3]

    print(pca_eps_min_df)
    best = max(pca_eps_min_df["silhouette_score"].unique())
    best_inf = pca_eps_min_df.iloc[pca_eps_min_df["silhouette_score"].argmax()]
    print(best)
    print(best_inf)


def Narrow_time_range(f,frame_len,start):
    if start == 0:
        start = f['Global_Time'].min()
    end = start+frame_len*100# 一帧100ms
    print(end)
    return f[(f['Global_Time']>= start) & (f['Global_Time']<= end)]


if __name__ == '__main__':
    # dbscan_test()
    # f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min_plus_acc.csv")#,skiprows=(1,2)
    # f2 = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min_plus_acc.csv",skiprows= 0,nrows=1000,dtype= np.float64)#,skiprows=(1,2)
    # f = pd.concat([f1,f2], ignore_index=True)
    # 数据清洗
    # f = Narrow_time_range(f, 10 * 60 * 10, 0)  # 截取30min内数据
    # f.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc.csv",
    #              index=False)
    f = pd.read_csv(r"/data/NGSIM_Data/i-80/big_car_10min_plus_acc.csv")


    print(len(f))
    f = clean_unnormal(f, 120, 30)
    f.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc.csv",index=False)

    df = f[['Vehicle_ID','x_a', 'y_v', 'y_a']]
    # plot2D(df, 'x_a', 'y_a')
    # ss(df)
    pca_eps_values = np.arange(1.0, 1.5, 0.1)
    pca_min_samples = np.arange(2, 15)
    # pca_with_eandm(df,pca_eps_values,pca_min_samples) # 确定合适的epsilon级别


    # 4.建立DBSCAN模型并可视化
    # par_list = [[1.1,2],[1.3,13],[1.4, 5], [0.5, 7], [0.8, 8],[0.6,6]] # 1000 [0.8, 8]最优选 5000 [1.3,13]
    par_list = [[1.0,8],[1.0,12],[1.0, 13], [1.0, 14], [1.1, 9],[1.4,5]] # 36338 [0.8, 8]最优选 5000 [1.3,13]
    # par_list = [[1,13],[0.8,13],[0.9, 9],[0.8,11]] # 1000 [0.8, 8]最优选 5000 [1.3,13]
    # 若类别数太多，需要增大eps领域大小，默认是0.5,似乎是关键，或者减少 min_samples 默认为5
    df_list = DBSCAN_process(df,par_list)
    # 绘图
    for DB_df in df_list:
        print("-------")
        # estimate_coffecient(DB_df )
        plot_seaboren(DB_df)

        # counts = pd.value_counts(DB_df, sort=True)
        # print(Counter(DB_df['driving_style']))
        # print(counts)
        # print(DB_df)



    #  分类
















