# -*- codeding = utf-8 -*-
# @Time : 2023/1/29 1:02
# @Author : 怀德
# @File : data_preprocess.py
# @Software: PyCharm
import csv
import pandas as pd
import numpy as np

# 1.先对按车ID排序 后按Frame排序
def sortcars(f_origin):
    # 返回值是一个带有multiindex的dataframe数据，其中level=0为groupby的by列，而level=1为原index
    # new_f = f_origin.groupby("Vehicle_ID").apply(lambda x: x.sort_values('Frame_ID', ascending=True))
    # 通过设置group_keys参数对multiindex进行优化
    return f_origin.groupby("Vehicle_ID", group_keys=False).apply(lambda x: x.sort_values('Frame_ID', ascending=True))

def caladd_Lateral_acc(f,local_x):
    '''
    对数据添加特征：前后帧localx,localy之差，并除以0.1，得到车辆的侧向加速度
    '''
    x=[]
    y=[]
    x_v=[]
    y_v=[]
    x_a = []
    y_a = []
    # x.append('none')
    x_v.append('none')
    x_a.append('none')
    # y.append('none')
    y_v.append('none')
    y_a.append('none')


    for i in range(1,len(local_x)):
        if local_x.Vehicle_ID[i]==local_x.Vehicle_ID[i-1]:
            # x.append(local_x.Local_X[i]-local_x.Local_X[i-1])
            x_dis = local_x.Local_X[i]-local_x.Local_X[i-1]
            y_dis = local_x.Local_Y[i]-local_x.Local_Y[i-1]
            x_v.append(x_dis/0.1)
            y_v.append(y_dis/0.1)
            # x_last_dis =  x_dis# 上一个辆车距离
            # y_last_dis =  y_dis
        else:
            x.append('none')
            x_v.append('none')
            y.append('none')
            y_v.append('none')

    for i in range(1,len(local_x)):
        if local_x.Vehicle_ID[i] == local_x.Vehicle_ID[i - 1]:
            if  x_v[i-1] != "none" and x_v[i] != "none" and y_v[i-1] != "none" and y_v[i] != "none":
                x_a.append(x_v[i-1] - x_v[i] / 0.1)
                y_a.append(y_v[i-1] - y_v[i] / 0.1)
            else:
                x_a.append('none')
                y_a.append('none')
        else:
            x_a.append('none')
            y_a.append('none')

    print(len(x_v))
    print(len(x_a))
    print(len(y_v))
    print(len(y_v))

    # dic={'x':x,'x_v':x_v,'y':y,'y_v':y_v}
    dic={'x_v':x_v,'x_a':x_a,'y_v':y_v,'y_a':y_a}
    df = pd.DataFrame(data=dic)
    # print(df)

    new_f=pd.concat([f,df],axis=1)
    # print(new_f)

    # new_f=new_f[new_f["x_v"]!='none']
    # new_f=new_f.rename(columns={'x_v':'x_a'})
    # print(new_f)

    return new_f

def Narrow_time_range(f,frame_len,start):
    if start == 0:
        start = f['Global_Time'].min()
    end = start+frame_len*100# 一帧100ms
    print(end)
    return f[(f['Global_Time']>= start) & (f['Global_Time']<= end)]

# df2 = df.copy()
# df2['Highest'] = df.idxmax(axis=1)
# df2['Lowest'] = df.idxmin(axis=1)

# 2.数据筛选-筛选出换道以及直行车辆

def inch_to_second(f):
    f['v_Acc'] = f['v_Acc'].map(lambda x: x* 0.3048)
    f['v_Vel'] = f['v_Vel'].map(lambda x: x* 0.3048)
    return f
    # meter = (foot + (inch / 12)) * 0.3048

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

    # f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv")

    # 1.挑选路段

    # datas_i_80_m = f[f["Location"] == 'i-80']
    # datas_i_80_m = datas_i_80_m[datas_i_80_m["v_Class"] == 3] #
    # datas_i_80_m.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_origin.csv",index=False)

    # 2.截取特定事件
    # f = pd.read_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_origin.csv")
    # # f = Narrow_time_range(f,10*60*10,0)#截取10 min内数据
    # f = Narrow_time_range(f,60*60*10,0)#截取1 h内数据
    # datas_i_80_m = sortcars(f)#排序
    # datas_i_80_m.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min.csv", index=False)
    #
    # print(datas_i_80_m)

    # 3.计算前后帧车辆xy方向速度
    f = pd.read_csv(r"/data/NGSIM_Data/i-80/big_car_30min.csv", nrows=10000)

    local_x = f[["Vehicle_ID","Frame_ID","Local_X", "Local_Y"]]
    acc = f[["Vehicle_ID","Frame_ID","Local_X", "Local_Y"]]

    # print(type(float(local_x.Local_Y[3])))
    new_f = caladd_Lateral_acc(f,local_x)
    # new_f = sortcars(new_f) # 排序
    print(new_f[["Vehicle_ID","Frame_ID","Local_X",'v_Vel','y_v','v_Acc','y_a']])
    new_f.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_30min_plus_acc.csv",
                 index=False)

    # 4.保存位训练数据
    # new_f.to_csv(r"D:\Scientific_Research\NGSIM_prediction\data\NGSIM_Data\i-80\big_car_origin_inorder.csv", index=False)
    # train_data = new_f[['Vehicle_ID', 'Local_X', 'Local_Y', 'v_Class', 'v_Vel', 'v_Acc', 'x_a', 'Lane_ID']]
    # print(train_data)
