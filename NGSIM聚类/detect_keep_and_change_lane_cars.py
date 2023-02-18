# -*- codeding = utf-8 -*-
# @Time : 2023/1/18 22:55
# @Author : 怀德
# @File : detect_keep_and_change_lane_cars.py
# @Software: PyCharm

# !/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
import random

def detect_change_lane(f):
    '''
    检索所有转弯车辆的索引位置，及左右转向
    '''
    ID = f[['Vehicle_ID', 'Local_X', 'Local_Y', 'Lane_ID',  'x_a', 'y_v']]
    # print(ID)
    change_vehicle_id = []
    change_vehicle_row = []
    # change_vehicle_row40=[]
    TR_row = []
    TL_row = []
    TR_vehicle_id = []
    TL_vehicle_id = []
    for i in range(len(f)):
        if i == 0:
            continue
        else:
            if ID['Vehicle_ID'][i] == ID['Vehicle_ID'][i - 1] and ID['Lane_ID'][i] != ID['Lane_ID'][i - 1]:
                if ID['Vehicle_ID'][i] not in change_vehicle_id:
                    change_vehicle_id.append(ID['Vehicle_ID'][i])
                #             change_vehicle_row40.extend(a for a in range(i-39:i-1))
                change_vehicle_row.append(i - 1)
                change_vehicle_row.append(i)
            if ID['Vehicle_ID'][i] == ID['Vehicle_ID'][i - 1] and ID['Lane_ID'][i] == ID['Lane_ID'][i - 1] + 1:
                if ID['Vehicle_ID'][i] not in TR_vehicle_id:
                    TR_vehicle_id.append(ID['Vehicle_ID'][i])
                TR_row.append(i - 1)
                TR_row.append(i)
            if ID['Vehicle_ID'][i] == ID['Vehicle_ID'][i - 1] and ID['Lane_ID'][i] == ID['Lane_ID'][i - 1] - 1:
                if ID['Vehicle_ID'][i] not in TL_vehicle_id:
                    TL_vehicle_id.append(ID['Vehicle_ID'][i])
                TL_row.append(i - 1)
                TL_row.append(i)
    # print(change_vehicle_row)  # 第某行数据为换道数据
    print("换道车辆统计共: "+ str(len(change_vehicle_id)))
    print("向左换道车辆: " + str(TL_row))
    print("向右换道车辆: " + str(TR_row))
    return change_vehicle_id,TL_row,TR_row


'''
通过索引，选出所有帧数大于40直行车辆
'''

def carID_set(f):
    Vehicle_ID = f['Vehicle_ID']
    Vehicle_ID_LIST = []
    for i in range(len(f)):
        if Vehicle_ID[i] not in Vehicle_ID_LIST:
            Vehicle_ID_LIST.append(Vehicle_ID[i])
    # print(Vehicle_ID_LIST)
    print("数据车辆总数： " + str(len(Vehicle_ID_LIST)))
    return Vehicle_ID

def detect_keep_lane(f,Vehicle_ID_LIST,change_vehicle_id): # 划分道
    '''
    def findadd_change_lane():
    '''
    LK_list = [] #直行车辆ID列表
    for i in Vehicle_ID_LIST:
        if i not in change_vehicle_id:# 排除变道就是直行
            LK_list.append(i)


    LK_ID40 = f[f.Vehicle_ID.isin(LK_list)]
    LK_ID40 = LK_ID40[LK_ID40.Lane_ID.isin([1, 2, 3, 4, 5])]
    LK_ID40['label'] = 0 # 变那根道
    LK_ID40 = LK_ID40.reset_index(drop=True)

    # print(LK_ID40)
    keep_lane_row = [] # 保持40帧以上的一个片段
    for i in LK_list:
        one_ve_info = LK_ID40[LK_ID40.Vehicle_ID.isin([i])]
        total_frames = len(one_ve_info)
        if total_frames >= 40: #
            # start_frame = random.randint(0, total_frames) - 40
            start_frame = random.randint(0, total_frames) - 40
            end_frame = start_frame + 40

            # one_ve_index = one_ve_info.iloc[start_frame:end_frame, :].index # 单车数据 截取4s以内
            one_ve_index = one_ve_info.iloc[start_frame:end_frame].index # 单车数据 截取4s以内
            keep_lane_row.extend(one_ve_index)
    # print(keep_lane_row) #

    print("数量： " + str(len(keep_lane_row) /40)) #车辆

    KL_list = list(set(list(LK_ID40.Vehicle_ID)))  # 更新后
    print("直行车辆数" + str(len(KL_list)))
    lk_40 = []
    # Vehicle_ID40 = []
    for i in range(len(LK_ID40)):
        if i <= 40:
            continue
        else:
            if LK_ID40['Vehicle_ID'][i] != LK_ID40['Vehicle_ID'][i - 1] and LK_ID40['Vehicle_ID'][i - 1] == \
                    LK_ID40['Vehicle_ID'][i - 40]:
                lk_40.extend(a for a in range(i - 40, i))
    # print("直行车辆帧数" + str(lk_40))
    # print("保持n帧以上的直行车辆帧数：" + str(len(lk_40))) #没啥用

    return LK_ID40,keep_lane_row,lk_40 # 返回直线车辆数据index 和 id列表

if __name__ == '__main__':
    # f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\new_NGSIM_Data\1_sample.csv")
    f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min_plus_acc.csv")  # ,nrows = 20000
    carID_set = carID_set(f)
    change_id,TL_row,TR_row = detect_change_lane(f)

    LK_ID40,keep_lane_row,lk_40 = detect_keep_lane(f,carID_set,change_id)
    # 保存数据
    # LK_ID40是40帧以上得车辆数据 KL_train40_csvs是将40帧数以上得都集合在一起 随机的数据集
    print(LK_ID40)

    KL_train40_csv = LK_ID40.loc[keep_lane_row]

    print(KL_train40_csv)
    # KL_40car = f[f['Vehicle_ID'].isin(lk_40)]
    # print(KL_train40_csv)
    # print(KL_40car)
    LK_ID40.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_kl40.csv", index=False)
    KL_train40_csv.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_kl40.csv", index=False)

    '''
    保存信息
    '''
    #
    # lane_local = pd.DataFrame(columns=["Vehicle_ID", "Lane_ID", "Local_X"])
    # change_lane_id = ID[ID.Vehicle_ID.isin(change_vehicle_id)]
    # TR_ID = ID[ID.Vehicle_ID.isin(TR_vehicle_id)]
    # TL_ID = ID[ID.Vehicle_ID.isin(TL_vehicle_id)]
    # change_lane_id.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_change_lane.csv", index=False)
    # print("换道车辆" + str(change_lane_id))
    #




