import pandas as pd
import os
import sys
import math
import VCRO_Model as vm
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from KDEpy import FFTKDE

LAT_MAX = 37.8
LON_MAX = 123.375
LAT_MIN = 37.1
LON_MIN = 122.5

def getPreviousRecord(currentTime, mmsi, date):
    """
    获取指定MMSI船舶的前一时刻记录
    :param currentTime: 当前时刻时间
    :param mmsi: 船舶MMSI
    :param date: 当前日期
    :return: 获取的记录(Series)
    """
    previousTime = str(int(currentTime) - 300)     #前一时刻时间
    file_path = "E:\成山头数据\\result\\" + date + "\\" + previousTime + ".csv"
    rankings_colname = ['Mmsi', 'Latitude', 'Longitude', 'Sog', 'Cog']
    try:
        df = pd.read_csv(file_path, header=None, names=rankings_colname)
    except Exception:
        return None
    #遍历每条船舶
    rowsNum_df = df.shape[0]
    for i in range(rowsNum_df):
        if df.iloc[i, 0] == mmsi:
            return df.iloc[i, :]
    #没有找到则递归找再前一个时刻文件
    if previousTime != '0':
        getPreviousRecord(previousTime, mmsi, date)
    else:
        return None

def calVCRO_each(df, time, resultList, date):
    """
    计算每条船舶的VCRO值
    :param df: 轨迹数据(DataFrame)
    :param time: 当前时刻
    :param resultList: 存储各条船舶VCRO值的List([[mmsi, [VCRO1，VCRO2，....]], ........])
    :param date: 日期
    :return:
    """
    rowsNum_df = df.shape[0]
    # 遍历每条船舶
    for i in range(rowsNum_df - 1):
        for j in range(i + 1, rowsNum_df):
            distance = vm.getDistance(df.iloc[i, 1], df.iloc[i, 2], df.iloc[j, 1], df.iloc[j, 2])
            # 判断两船之间距离是否小于6nm,若是则进行下一步判断，若不是直接跳过
            if distance < 6:
                # 判断两船是否有前一时刻记录，若无则直接跳过，若有则继续下一步判断
                previousRecord_1 = getPreviousRecord(time, df.iloc[i, 0], date)
                previousRecord_2 = getPreviousRecord(time, df.iloc[j, 0], date)
                if previousRecord_1 is None or previousRecord_2 is None:
                    continue
                else:
                    if len(previousRecord_1) != 0 and len(previousRecord_2) != 0:
                        # 判断两船之间相位的正负，若为负(False)则无风险，直接跳过，若为正(True)则计算VCRO值
                        phase_plus_minus = vm.getPhase_plus_minus(df.iloc[i, 1], df.iloc[i, 2], df.iloc[j, 1],
                                                                  df.iloc[j, 2], df.iloc[i, 4], df.iloc[j, 4],
                                                                  previousRecord_1.iloc[0], previousRecord_1.iloc[1],
                                                                  previousRecord_2.iloc[0], previousRecord_2.iloc[1])
                        # phase_plus_minus = True
                        if phase_plus_minus == True:
                            relative_speed = vm.getRelativeSpeed(df.iloc[i, 3], df.iloc[j, 3], df.iloc[i, 4], df.iloc[j, 4])
                            phase = vm.getPhase(df.iloc[i, 1], df.iloc[i, 2], df.iloc[j, 1], df.iloc[j, 2], df.iloc[i, 4],
                                                df.iloc[j, 4], previousRecord_1.iloc[0], previousRecord_1.iloc[1],
                                                previousRecord_2.iloc[0], previousRecord_2.iloc[1])
                            vcro = vm.getVCRO(distance, relative_speed, phase)
                            # vcro = 1 - math.exp(-vcro / 100)
                            vcro = 1 - math.exp(-vcro)         #此处的参数是动态获取的，不同的数据集需要重新敲定
                            resultList[i][1].append(vcro)
                            resultList[j][1].append(vcro)

def calRisk_each(resultList, riskList):
    """
    计算各条船舶的风险值
    :param resultList: 各条船舶VCRO值的List
    :param riskList: 各条船舶风险值的List
    :return:
    """
    for i in range(len(resultList)):
        vcroList = resultList[i][1]
        if len(vcroList) != 0:
            vcroList = sorted(vcroList, reverse=True)
            risk = 0.0
            temp = vcroList[0]
            for j in range(len(vcroList)):
                if j == 0:
                    risk = temp
                    temp = 1 - temp
                else:
                    risk += temp * vcroList[j]
                    temp -= temp * vcroList[j]
            riskList[i].append(risk)

def getRisk(date, time):
    """
    获取指定时间的全部船舶风险值
    :param date: 指定的日期(str)
    :param time: 指定的时间(int)
    :return: 该时刻全部船舶风险值(List)
    """
    file_path = "E:\成山头数据\\result\\" + date + "\\" + str(time) + ".csv"
    if not os.path.exists(file_path):      #若文件不存在直接返回空列表
        return []
    rankings_colname = ['Mmsi', 'Latitude', 'Longitude', 'Sog', 'Cog']
    df = pd.read_csv(file_path, header=None, names=rankings_colname)
    rowsNum_df = df.shape[0]
    resultList = []     #创建存储各条船舶VCRO值的List([[mmsi, [VCRO1，VCRO2，....]], ........])
    for i in range(rowsNum_df):
        tempList = [df.iloc[i, 0], []]
        resultList.append(tempList)
    riskList = []      #创建存储各条船舶风险值的List([[mmsi, 风险值], ........])
    for i in range(rowsNum_df):
        tempList = [df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]]
        riskList.append(tempList)
    #风险值计算
    calVCRO_each(df, time, resultList, date)      #计算各船VCRO值，并存入resultList
    calRisk_each(resultList, riskList)
    return riskList

def writeRisk(date):
    """
    将风险值写入CSV文件
    :param date: 待写入数据的日期
    :return:
    """
    # 创建目录
    dir_result_path = "E:\成山头数据\\risk\\" + date  # 结果存储目录路径
    if not os.path.exists(dir_result_path):
        os.mkdir(dir_result_path)
    for i in range(900, 86400, 3600):
        for j in range(i, i+3600, 900):
            riskList = getRisk(date, j)      #得到的风险结果
            file_path = dir_result_path + "\\" + str(j) + ".csv"
            with open(file_path, "a", newline="") as filewriter:
                writer = csv.writer(filewriter)
                for risk in riskList:
                    writer.writerow(risk)
                filewriter.close()
            print("{}:{}-{}风险写入完毕".format(date, j, j+900))
        print("---------------------------")

def gridPatition():
    """
    网格划分
    :return: 网格边界（经纬度的最大和最小共四个值），网格划分行列数量
    """
    root_path = "E:\\成山头数据\\data"
    latMax = 0
    lonMax = 0
    latMin = 10000
    lonMin = 10000
    dirs = os.listdir(root_path)
    for dir in dirs:
        dir_path = root_path + "\\" + str(dir)
        files = os.listdir(dir_path)
        for file in files:
            file_path = dir_path + "\\" + str(file)
            rankings_colname = ['Date_Time', 'Timestamp', 'Latitude', 'Longitude', 'Sog', 'Cog']
            df = pd.read_csv(file_path, header=None, names=rankings_colname)
            latMaxTemp = df.iloc[:, 2].max()
            latMinTemp = df.iloc[:, 2].min()
            lonMaxTemp = df.iloc[:, 3].max()
            lonMinTemp = df.iloc[:, 3].min()
            if latMaxTemp > latMax:
                latMax = latMaxTemp
            if latMin > latMinTemp:
                latMin = latMinTemp
            if lonMaxTemp > lonMax:
                lonMax = lonMaxTemp
            if lonMin > lonMinTemp:
                lonMin = lonMinTemp
    return latMax, lonMax, latMin, lonMin

def constructGridMatrix():
    """
    构建水域网格化矩阵
    :return: 网格张量
    """
    length = vm.getDistance(LAT_MAX, LON_MAX, LAT_MAX, LON_MIN)  # 网格区域长度
    width = vm.getDistance(LAT_MAX, LON_MAX, LAT_MIN, LON_MAX)  # 网格区域宽度
    # row_num = int(width / 2)  # 网格行数
    # column_num = int(length / 2)+1  # 网格列数，此行还需要修改，+1带有随意性
    row_num = 21  # 网格行数
    column_num = 21  # 网格列数

    #遍历风险文件
    root_path = "E:\成山头数据\\risk"
    dirs = os.listdir(root_path)
    # 创建张量（全部日期）(7*7*dayx24)
    gridTensor = np.zeros((row_num, column_num, len(dirs)*24))
    dayTemp = 1  # 表征日数临时变量，用于决定gridTensor_day写入gridTensor第几日
    for dir in dirs:
        gridTensor_day = np.zeros((row_num, column_num, 24))    #创建日张量(7*7*24)
        tensor_index = 0    #日张量slice索引
        dir_path = root_path + "\\" + str(dir)
        #遍历一日所有时刻的文件
        gridArray = np.zeros((row_num, column_num))   # 创建水域网格化矩阵（初始值均为0）
        for file in range(900, 87300, 900):
            file_path = dir_path + "\\" + str(file) + ".csv"
            rankings_colname = ['Mmsi', 'Latitude', 'Longitude', 'Risk']
            df = pd.read_csv(file_path, header=None, names=rankings_colname)
            #遍历某时刻所有风险记录，计算对应的风险矩阵（一个时刻一个风险矩阵）
            for i in range(df.shape[0]):
                if not np.isnan(df.iloc[i, 3]):
                    row, column = generalID(df.iloc[i, 2], df.iloc[i, 1], column_num, row_num)
                    if row == -1 and column == -1:
                        continue
                    gridArray[row][column] += df.iloc[i, 3]
            if file % 3600 == 0:
                if file != 86400:
                    gridArray /= 4
                    gridArray = get_kde(gridArray)     #KDE，使风险矩阵更为平滑
                    gridArray = np.exp(gridArray)      #对风险矩阵做exp处理，防止稀疏
                    gridTensor_day[:, :, tensor_index] = gridArray   #将gridArray加入张量中
                    tensor_index += 1
                    gridArray = np.zeros((row_num, column_num))  #重新创建水域网格化矩阵（初始值均为0）
                else:
                    gridArray /= 3
                    gridArray = get_kde(gridArray)     #KDE，使风险矩阵更为平滑
                    gridArray = np.exp(gridArray)      #对风险矩阵做exp处理，防止稀疏
                    gridTensor_day[:, :, tensor_index] = gridArray   #将gridArray加入张量中

            print("{}风险矩阵构造完毕".format(file_path))

        gridTensor[:, :, (dayTemp-1)*24 : dayTemp*24] = gridTensor_day      #将gridTensor_day写入gridTensor
        dayTemp += 1
    #计算稀疏率
    # print("稀疏率：{}".format(len(np.where(gridTensor == 1)[0])/21/21/48))
    return gridTensor

def get_kde(gridArray):
    """
    对风险矩阵做KDE，降低稀疏率
    :param gridArray: 待处理风险矩阵
    :return: KDE后的风险矩阵
    """
    #判断风险矩阵是否为零矩阵，若是则返回原矩阵
    if np.where(gridArray != 0)[0].shape[0] == 0:
        return gridArray
    #找到矩阵中的非零值
    tempArray = np.nonzero(gridArray)
    data = np.zeros((tempArray[0].T.shape[0], 2))     #数据矩阵
    data[:, 0] = tempArray[0].T
    data[:, 1] = tempArray[1].T
    rows = data.shape[0]
    weights = []
    for i in range(rows):
        weights.append(gridArray[int(data[i, 0]), int(data[i, 1])])

    # fig = plt.figure(figsize=(2,1))
    # ax = fig.add_subplot(1,2,1)
    # bx = fig.add_subplot(1,2,2)
    # datai = np.zeros((gridArray.shape[0]*gridArray.shape[1], 2))
    # dataj = []
    # temp = 0
    # for i in range(len(gridArray)):
    #     for j in range(len(gridArray)):
    #         datai[temp, :] = np.array([i, j])
    #         dataj.append(gridArray[i, j])
    #         temp += 1

    grid_points = gridArray.shape[0]       #矩阵行（列）数
    kde = FFTKDE(kernel='gaussian', norm=2, bw=0.5)
    grid, points = kde.fit(data, weights=weights).evaluate(grid_points)
    # x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    resultArray = points.reshape(grid_points, grid_points).T

    # ax.contour(x,y,resultArray, 16, linewidths=0.8, colors='k')
    # ax.contourf(x,y,resultArray, 16, cmap='RdBu_r')
    # ax.plot(data[:, 0], data[:, 1], 'ok', ms=3)

    #原矩阵构图
    # grid, points = kde.fit(datai, weights=dataj).evaluate(grid_points)
    # x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    # z = points.reshape(grid_points, grid_points).T
    # z = RotateMatrix(z)     #z旋转90度
    # bx.contour(x, y, z, 16, linewidths=0.8, colors='k')
    # bx.contourf(x, y, z, 16, cmap='RdBu_r')

    # plt.tight_layout()
    # plt.show()

    # print(gridArray)
    # print(resultArray)
    return resultArray

def RotateMatrix(arr):
    tr = tc = 0
    dr = len(arr)-1
    dc = len(arr[0])-1
    while tr<dr:
        #因为等于的时候就一个数所以不用旋转了
        for i in range(dc-tc):
            tmp = arr[tr][tc+i]
            arr[tr][tc+i] = arr[dr-i][tc]
            arr[dr-i][tc] = arr[dr][dc-i]
            arr[dr][dc-i] = arr[tr+i][dc]
            arr[tr+i][dc] = tmp
        tr= tc =tr+1
        dc =dr = dc-1
    return arr

def generalID(lon, lat, column_num, row_num):
    """
    根据经纬度获取所在网格的位置
    :param lon: 经度
    :param lat: 纬度
    :param column_num: 网格列数
    :param row_num: 网格行数
    :return: 所在网格位置（行列号）
    """
    #若在范围外的点，返回-1
    if lon <= LON_MIN or lon >= LON_MAX or lat <= LAT_MIN or lat >= LAT_MAX:
        return -1, -1
    #把经度范围根据列数等分切割
    column = 2
    #把纬度范围根据行数等分切割
    row = 2
    #计算行列号
    column = int(vm.getDistance(lat, lon, lat, LON_MIN) / column)
    row = int(vm.getDistance(lat, lon, LAT_MAX, lon) / row)
    return row, column

if __name__ == '__main__':
    # pass
    # root_path = "E:\成山头数据\\result\\"
    # dirs = os.listdir(root_path)
    # for dir in dirs:
    #     if str(dir) <= "2018-01-07":
    #         continue
    #     writeRisk(str(dir))
    #网格化，此处获取网格边界，然后人为划分，不同的数据集需要重新敲定
    # gridPatition()
    # 构建网格化水域张量
    gridTensor = constructGridMatrix()
    #保存张量到mat文件
    data_path = "E:\成山头数据\\data.mat"
    scio.savemat(data_path, {"tensor":gridTensor})

    # X = np.array([[1,1], [1,2], [1,3], [2,1], [2,3], [2,4]])
    # Y = np.array([[1,3], [2,3], [3,4]])
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    # log_dens = kde.score_samples(Y)
    # print(np.exp(log_dens))






