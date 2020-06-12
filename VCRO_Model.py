from math import *
import csv
import numpy as np
import matplotlib.pyplot as plt

def getDistance(lat1, lon1, lat2, lon2):
    """
    计算两船之间距离
    :param lat1: 1号点纬度
    :param lon1: 1号点经度
    :param lat2: 2号点纬度
    :param lon2: 2号点经度
    :return: 两船之间的距离
    """
    lat1 = lat1 / 180 * pi
    lon1 = lon1 / 180 * pi
    lat2 = lat2 / 180 * pi
    lon2 = lon2 / 180 * pi
    # lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    a = lat2 - lat1
    b = lon2 - lon1
    R = 6371   #结果中的距离与R单位一致，km或者m
    x = 2 * asin(sqrt(sin(a/2) ** 2 + cos(lat1) * cos(lat2) * sin(b/2) ** 2)) * R
    return x / 1.852

def getRelativeSpeed(v1, v2, c1, c2):
    """
    计算两船之间相对速度
    :param v1: 1号船速度
    :param v2: 2号船速度
    :param c1: 1号船航向
    :param c2: 2号船航向
    :return: 两船之间的相对速度
    """
    r = getPhase_data(c1, c2)
    c = sqrt(v1 ** 2 + v2 ** 2 - 2*v1*v2*cos(r*pi/180))
    return c

def getPhase(lat1, lon1, lat2, lon2, c1, c2, lat1_past, lon1_past, lat2_past, lon2_past):
    """
    计算两船之间相位值
    :param lat1: 1号船当前纬度
    :param lon1: 1号船当前经度
    :param lat2: 2号船当前纬度
    :param lon2: 2号船当前经度
    :param c1: 1号船当前航向
    :param c2: 2号船当前航向
    :param lat1_past: 1号船前一位置纬度
    :param lon1_past: 1号船前一位置经度
    :param lat2_past: 2号船前一位置纬度
    :param lon2_past: 2号船前一位置经度
    :return: 两船之间相位值
    """
    #获取两船相位值正负
    flag = getPhase_plus_minus(lat1, lon1, lat2, lon2, c1, c2, lat1_past, lon1_past, lat2_past, lon2_past)
    r = getPhase_data(c1, c2)
    if flag == True:
        r = abs(r)
    else:
        r = -abs(r)
    return r

def getPhase_data(c1, c2):
    """
    获取两船之间相位数值或者航向夹角
    :param c1: 1号船航向
    :param c2: 2号船航向
    :return: 两船之间相位数值
    """
    temp = abs(c1 - c2)
    r = 0.0
    if temp < 180:
        r = temp
    else:
        r = 360 - temp
    return r

def getPhase_plus_minus(lat1, lon1, lat2, lon2, c1, c2, lat1_past, lon1_past, lat2_past, lon2_past):
    """
    计算两船相位正负，用于判断两船舶是靠近还是远离，是否有碰撞风险
    :param lat1: 1号船当前纬度
    :param lon1: 1号船当前经度
    :param lat2: 2号船当前纬度
    :param lon2: 2号船当前经度
    :param c1: 1号船当前航向
    :param c2: 2号船当前航向
    :param lat1_past: 1号船前一位置纬度
    :param lon1_past: 1号船前一位置经度
    :param lat2_past: 2号船前一位置纬度
    :param lon2_past: 2号船前一位置经度
    :return: 两船相位正负值（True：有碰撞风险，False：无碰撞风险）
    """
    #计算两船当前航向斜率
    k1 = getK(c1)
    k2 = getK(c2)
    #计算两船当前法线斜率
    k1_normal = 0.0
    k2_normal = 0.0
    if k1 == 0:
        k1_normal = np.inf
    elif k1 == np.inf:
        k1_normal = 0
    else:
        k1_normal = -1 / k1
    if k2 == 0:
        k2_normal == np.inf
    elif k2 == np.inf:
        k2_normal = 0
    else:
        k2_normal = -1 / k2
    #计算两船直线方程交点
    b1 = lat1 - k1 * lon1
    b2 = lat2 - k2 * lon2
    x_cross = (b2 - b1) / (k1 - k2)
    y_cross = k1 * x_cross + b1
    # y_cross = k2 * x_cross + b2  两种方法均能得到交点坐标
    #计算两船当前法线方程
    b1_normal = lat1 - k1_normal * lon1
    b2_normal = lat2 - k2_normal * lon2
    #判断是否有碰撞风险
    #两船前一位置点与交点均需位于各自法线两侧则有风险，否则无风险
    flag1 = flag2 = False
    if k1_normal == np.inf:
        if (lon1_past < lon1 and x_cross > lon1) or (lon1_past > lon1 and x_cross < lon1):
            flag1 = True
    elif k1_normal == 0:
        if (lat1_past < lat1 and y_cross > lat1) or (lat1_past > lat1 and y_cross < lat1):
            flag1 = True
    else:
        if (k1_normal * x_cross + b1_normal < y_cross and k1_normal * lon1_past + b1_normal > lat1_past) or (k1_normal * x_cross + b1_normal > y_cross and k1_normal * lon1_past + b1_normal < lat1_past):
            flag1 = True
    if k2_normal == np.inf:
        if (lon2_past < lon2 and x_cross > lon2) or (lon2_past > lon2 and x_cross < lon2):
            flag2 = True
    elif k2_normal == 0:
        if (lat2_past < lat2 and y_cross > lat2) or (lat2_past > lat2 and y_cross < lat2):
            flag2 = True
    else:
        if (k2_normal * x_cross + b2_normal < y_cross and k2_normal * lon2_past + b2_normal > lat2_past) or (k2_normal * x_cross + b2_normal > y_cross and k2_normal * lon2_past + b2_normal < lat2_past):
            flag2 = True
    return flag1 and flag2

def getK(c):
    """
    计算航向对应斜率
    :param c: 航向
    :return: 对应斜率
    """
    k = 0.0
    if 0 <= c < 90:
        k = tan((90 - c)*pi/180)
    elif 90 <= c < 180:
        k = -tan((c - 90)*pi/180)
    elif 180 <= c < 270:
        k = tan((270 - c)*pi/180)
    elif 270 <= c < 360:
        k = -tan((c - 270)*pi/180)
    elif c == 0 or c == 180:
        k = np.inf
    return k

def getVCRO(distance, relative_speed, phase):
    vcro = 3.87 * (distance**(-1)) * relative_speed * (sin(phase*pi/180) + 0.386 * sin(2*phase*pi/180))
    return vcro

def calVCRO(file1, file2):
    """
    计算两条轨迹的VCRO
    :param file1: 轨迹1的文件路径
    :param file2: 轨迹2的文件路径
    :return: 距离，相对速度，相位，VCRO结果列表
    """
    with open(file1) as f:
        reader = csv.reader(f)
        data1 = [row for row in reader]    #文件1数据
        f.close()
    with open(file2) as f:
        reader = csv.reader(f)
        data2 = [row for row in reader]    #文件2数据
        f.close()
    #创建结果集存储列表变量
    distance_list = [0] * len(data1)
    relative_speed_list = [0] * len(data1)
    phase_list = [0] * len(data1)
    vcro_list = [0] * len(data1)
    #分别遍历两条轨迹数据
    for i in range(len(data1)):
        distance_list[i] = getDistance(float(data1[i][2]), float(data1[i][3]), float(data2[i][2]), float(data2[i][3]))
        relative_speed_list[i] = getRelativeSpeed(float(data1[i][4]), float(data2[i][4]), float(data1[i][5]), float(data2[i][5]))
        if i == 0:     #跳过第一条数据（因为没有前一时刻坐标等信息）
            continue
        else:
            phase_list[i] = getPhase(float(data1[i][2]), float(data1[i][3]), float(data2[i][2]), float(data2[i][3]), float(data1[i][5]), float(data2[i][5]), float(data1[i-1][2]), float(data1[i-1][3]), float(data2[i-1][2]), float(data2[i-1][3]))
            vcro_list[i] = getVCRO(distance_list[i], relative_speed_list[i], phase_list[i])
    latitude_list1 = [0] * len(data1)
    longitude_list1 = [0] * len(data1)
    latitude_list2 = [0] * len(data1)
    longitude_list2 = [0] * len(data1)
    for i in range(len(data1)):
        latitude_list1[i] = float(data1[i][2])
        longitude_list1[i] = float(data1[i][3])
        latitude_list2[i] = float(data2[i][2])
        longitude_list2[i] = float(data2[i][3])
    plt.figure()
    plt.plot(longitude_list1, latitude_list1, 'o', label="第1条轨迹")
    plt.plot(longitude_list2, latitude_list2, 'o', label="第2条轨迹")
    plt.show()

    return distance_list, relative_speed_list, phase_list, vcro_list


if __name__ == '__main__':
    file1 = "C:\\Users\Administrator\Desktop\Box\时空风险建模与计算\代码\船舶冲突数据\encounter_3_crossing\\ship_01.csv"
    file2 = "C:\\Users\Administrator\Desktop\Box\时空风险建模与计算\代码\船舶冲突数据\encounter_3_crossing\\ship_02.csv"
    distance_list, relative_speed_list, phase_list, vcro_list = calVCRO(file1, file2)
    #作图
    plt.figure(figsize=(15, 7))

    plt.subplot(221)
    t = np.arange(0, 10, 10/len(distance_list))
    plt.plot(t, distance_list)
    plt.xlabel('Time(min)')
    plt.ylabel('Distance(nm)')

    plt.subplot(222)
    t = np.arange(0, 10, 10 / len(relative_speed_list))
    plt.plot(t, relative_speed_list)
    plt.xlabel('Time(min)')
    plt.ylabel('Relative_speed(kn)')

    plt.subplot(223)
    t = np.arange(0, 10, 10 / len(phase_list))
    plt.plot(t, phase_list)
    plt.xlabel('Time(min)')
    plt.ylabel('Phase(degree)')

    plt.subplot(224)
    t = np.arange(0, 10, 10 / len(vcro_list))
    plt.plot(t, vcro_list)
    plt.xlabel('Time(min)')
    plt.ylabel('VCRO')

    plt.show()
