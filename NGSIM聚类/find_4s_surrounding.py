# -*- codeding = utf-8 -*-
# @Time : 2023/1/29 13:22
# @Author : 怀德
# @File : find_4s_surrounding.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import random
f=pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min_plus_acc.csv")
# df=pd.read_csv("/home/mt/learn/NGSIM数据处理/csv/80/4:00-4:15/CL_train41.csv")
ff=pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min.csv")

f=f.rename(columns={'Preceeding':'Preceding'})
num1=int(len(f)/41)
id_max=max(ff.Vehicle_ID.values)
time_point=0
point_list=[]
for i in range(num1):
    point_list.append(time_point+41*i) # 每40秒取样，去检测

print(point_list)


f1=f.iloc[point_list]
f2=f1[['Vehicle_ID','Preceding','Following']].values
index=[]
for i in range(num1):
    if f2[i][1]>id_max or f2[i][2]>id_max:
        index.append(i)
print("1--------")
print(index)
index_row=[]
for i in index:
    index_row.extend(a for a in range(i*41,i*41+41))
f=f.drop(index_row) # 丢弃前后车不出现在当前数据中的行
f=f.reset_index(drop=True)
f.Vehicle_ID=f.Vehicle_ID.astype('object')
f.Preceding=f.Preceding.astype('object')
f.Following=f.Following.astype('object')
f=f.astype('object')
print("2--------")
print(f)

num2=int(len(f)/41)
point_list2=[]
for i in range(num2):
    point_list2.append(time_point+41*i)
f3=f.iloc[point_list2]
f4=f3[['Vehicle_ID','Frame_ID','Preceding','Following']].values
print("f4--------")
print(f4) # 取timepoint上的数据
print(len(f4))

def add_aim_lane(f):
    row41=[]
    for i in range(num2):# 取样每40帧的数据
        row41.append(41*i+40)
    df=f.drop(row41)
    df=df.reset_index(drop=True)
    # print(df)

    pd41=f.loc[row41]
    print(pd41) # 第40帧的数据

    aim_lane=pd41.Lane_ID.values # 取出40帧的车道线数据
    print("aim_lane--------")
    # print(aim_lane)

    aim_lane_list=[]
    for i in aim_lane:
        for a in range(40):
            aim_lane_list.append(i)
    # print(aim_lane_list)
    aim_lane_data=pd.DataFrame(data=aim_lane_list,columns={'aim_lane'})
    print("aim_lane_data--------")
    print(aim_lane_data)
    df_new=pd.concat([df,aim_lane_data],axis=1)
    print("加入目标车道后-------")
    print(df_new)
    print(len(df_new))
    return df_new

f = add_aim_lane(f)

def choose_frame(arr):
    Pre = pd.DataFrame(columns=['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y','v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceding','Following','Space_Headway','Time_Headway'])
    Fol = pd.DataFrame(columns=['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y','v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceding','Following','Space_Headway','Time_Headway'])
    shape=np.shape(arr)
    for i in range(shape[0]):
        a=ff[ff.Vehicle_ID.isin([arr[i][2]])]
        b=a[a.Frame_ID.isin([arr[i][1]])]
        b2=b.copy()
        b2['Vehicle_ID1']=arr[i][0]
        c=ff[ff.Vehicle_ID.isin([arr[i][3]])]
        d=c[c.Frame_ID.isin([arr[i][1]])]
        d2 = d.copy()
        d2['Vehicle_ID1']=arr[i][0]
        Pre=pd.concat([Pre,b2], ignore_index=True)
        Fol=pd.concat([Fol,d2], ignore_index=True)
    return Pre,Fol

print("f2------")
print(type(f4[2][0]))
print(type(f4[3][0]))
f_1=f[['Vehicle_ID','Frame_ID','Preceding','Following']].values
Pre1,Fol1=choose_frame(f_1)
print("Pre1---------")
print(Pre1)
print("Fol1---------")
print(Fol1)

Pre1=Pre1[['Vehicle_ID','Frame_ID','Local_X','Local_Y','v_Vel','Vehicle_ID1']]
Fol1=Fol1[['Vehicle_ID','Frame_ID','Local_X','Local_Y','v_Vel','Vehicle_ID1']]
Pre1.columns=['Preceding','Frame_ID','Pre1_Local_X','Pre1_Local_Y','Pre1_v_Vel','Vehicle_ID']
Fol1.columns=['Following','Frame_ID','Fol1_Local_X','Fol1_Local_Y','Fol1_v_Vel','Vehicle_ID']

f.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_New.csv",index = False)
Pre1.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_Pre.csv",index = False)
Fol1.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_Fol.csv",index = False)

Pre1 = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_Pre.csv")
f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_New.csv")
Fol1 = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_Fol.csv")


def find_same(df,col1,col2):
    df_new = df[df.duplicated(subset=[col1, col2], keep=False)]
    return df_new

# 数据合并
# pd1=pd.merge(f,Pre1, how='left', on=['Frame_ID','Preceding','Vehicle_ID'])
# df = f[['Vehicle_ID','Frame_ID','Preceding','Following']]
df = f[['Vehicle_ID','Frame_ID','Local_X','Local_Y','v_Vel','Preceding','Following','Lane_ID','aim_lane']]

pd1=pd.merge(df,Pre1,how='left',on=['Vehicle_ID','Frame_ID','Preceding'])
# pd1=pd1.dropna(axis=0,how='any')
print("pd1-----")
print(pd1)
pd2=pd.merge(pd1,Fol1,how='left',on=['Vehicle_ID','Frame_ID','Following'])
print("pd2-----")
print(pd2)
pd3=pd2.dropna(axis=0,how='any')
print("pd3-----")
print(pd3)

pd3.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_10min_plus_acc_useful.csv",index = False)

