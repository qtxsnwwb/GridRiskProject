import pandas as pd
import scipy.interpolate as spi
import numpy as np
import os
import csv
import sys

def medianFilter(tempSeries):
    """
    对待处理的数据列进行中值滤波数据去噪
    :param tempSeries: 待处理数据列(Series)
    :return: 处理后的数据列(Series)
    """
    for index in range(len(tempSeries)):
        tempList = []       #存放每轮循环待中值滤波处理的数据
        #先存放当前循环待处理的值
        if not np.isnan(tempSeries.iloc[index]):
            tempList.append(tempSeries.iloc[index])
        #依次取出当前循环待处理值的前两个和后两个值存入tempList
        if index + 1 <= len(tempSeries)-1:
            if not np.isnan(tempSeries.iloc[index + 1]):
                tempList.append(tempSeries.iloc[index + 1])
        if index - 1 >= 0:
            if not np.isnan(tempSeries.iloc[index - 1]):
                tempList.append(tempSeries.iloc[index - 1])
        if index + 2 <= len(tempSeries)-1:
            if not np.isnan(tempSeries.iloc[index + 2]):
                tempList.append(tempSeries.iloc[index + 2])
        if index - 2 >= 0:
            if not np.isnan(tempSeries.iloc[index - 2]):
                tempList.append(tempSeries.iloc[index - 2])
        tempList.sort()     #从小到大排序
        #判断中位数数量
        if len(tempList) % 2 != 0:
            tempSeries.iloc[index] = tempList[int(len(tempList) / 2)]
        else:
            tempSeries.iloc[index] = (tempList[int(len(tempList) / 2)] + tempList[int(len(tempList) / 2) - 1]) / 2
    return tempSeries

def cutData_longInterval(df, list):
    """
    对缺失大量数据（数据发送间隔>0.5h）的情况进行数据切割
    :param df: 待处理数据(DataFrame)
    :param list: 存储切分轨迹的list
    :return: 是否进行了数据切分标识(True表示切分了，False表示没切分)
    """
    flag = False
    rowsNum = df.shape[0]
    for index in range(rowsNum - 1):
        if df.iloc[index + 1, 1] - df.iloc[index, 1] > 1800:
            flag = True
            list.append(df.iloc[:index+1, :])
            cutData_longInterval(df.iloc[index+1:, :], list)
    return flag

def cutData_anomaly(trajDf, trajList):
    """
    对持续航速<1，时间>0.5h的异常轨迹进行数据切割
    :param trajDf: 待处理轨迹数据段
    :param trajList: 存储切分轨迹的list
    :return: 是否进行了数据切分标识(True表示切分了，False表示没切分)
    """
    flag = False
    # 找寻航速<1的起始点
    rowsNum = trajDf.shape[0]
    for i in range(rowsNum-2):
        if trajDf.iloc[i, 4] < 1:
            #找寻航速<1的终止点
            end = 0    #终止点索引
            for j in range(i+1, rowsNum):
                if trajDf.iloc[j, 4] > 1:
                    end = j - 1
                    break
            #时间间隔>0.5h(1800s)
            if trajDf.iloc[end, 1] - trajDf.iloc[i, 1] > 1800:
                flag = True
                trajList.append(trajDf.iloc[:i+1, :])
                cutData_anomaly(trajDf.iloc[end:, :], trajList)
    return flag

def getTimeRange(time_start, time_end):
    """
    获取插值时间范围
    :param time_start: 数据起始时间
    :param time_end: 数据终止时间
    :return: 插值时间范围(numpy.ndarray)
    """
    if time_start % 300 != 0:
        time_start = (int(time_start / 300) + 1) * 300
    if time_end % 300 != 0:
        time_end = (int(time_end / 300) + 1) * 300
    t = np.arange(time_start, time_end, 300)  # 插值时间范围
    return t

def cubic_spline_latAndLon(X, Y, t):
    """
    三次样条插值（经纬度）
    :param X: 时间
    :param Y: 经度/纬度
    :param t: 插值时间
    :return: 插值后的经度/纬度(numpy.ndarray)
    """
    ipo3 = spi.splrep(X, Y, k=3)     #源数据点导入，生成参数
    iy3 = spi.splev(t, ipo3)         #根据观测点和样条参数，生成插值
    return iy3

def cubic_spline_cog(X, Y, t):
    """
    三次样条插值（航向）
    :param X: 时间
    :param Y: 航向
    :param t: 插值时间
    :return: 插值后的航向(numpy:ndarray)
    """
    #对待插值航向做初步处理
    for index in range(Y.size-1):
        if abs(Y.iloc[index + 1] - Y.iloc[index]) >= 180:
            if Y.iloc[index + 1] > Y.iloc[index]:     #此处存疑
                Y.iloc[index] = Y.iloc[index] + 360
            else:
                Y.iloc[index + 1] = Y.iloc[index + 1] + 360
    #插值
    ipo3 = spi.splrep(X, Y.values, k=3)      #源数据点导入，生成参数
    iy3 = spi.splev(t, ipo3)                 #根据观测点和样条参数，生成插值
    #对插值结果进行处理
    for i in range(len(iy3)):
        if iy3[i] / 360 > 1:
            iy3[i] = iy3[i] - 360
    return iy3

def preprocessing_each(df):
    """
    数据预处理执行函数（针对每条船舶），处理的记录为一条船舶一天的记录
    :param df: 待处理的船舶一天的记录(DataFrame)
    :return: 该船舶可进行风险计算的预处理后数据(DataFrame)
    """
    rowsNum = df.shape[0]     #数据行数
    #数据清洗
    #1.删除重复发送数据
    df = removeDuplicates(df)
    #2.对航速和航向、经纬度进行数据去噪
    df.iloc[:, 2] = medianFilter(df.iloc[:, 2].copy())
    df.iloc[:, 3] = medianFilter(df.iloc[:, 3].copy())
    df.iloc[:, 4] = medianFilter(df.iloc[:, 4].copy())
    df.iloc[:, 5] = medianFilter(df.iloc[:, 5].copy())
    #3.寻找缺失值并插值（航速、航向、经纬度）

    #4.缺失大量数据，发送时间间隔超过0.5h，轨迹一分为二
    trajList_temp = []
    flag = cutData_longInterval(df, trajList_temp)
    if flag == False:
        trajList_temp.append(df)

    #5.切割航速<1，时间>0.5h的轨迹
    trajList = []      #存储切分后的最终轨迹数据([DataFrame, .....])，每一项都是一个轨迹段
    #遍历所有轨迹段
    for trajDf in trajList_temp:
        flag = cutData_anomaly(trajDf, trajList)
    if flag == False:
        trajList = trajList_temp

    resultDf = pd.DataFrame(columns=('Timestamp', 'Latitude', 'Longitude', 'Sog', 'Cog'))      #创建存储插值结果的DataFrame
    for trajDf in trajList:
        rowsNum_temp = trajDf.shape[0]
        if rowsNum_temp <= 3:     #轨迹数据量过少直接跳过
            continue
        # 6.异常数据剔除：航速非常高、航向>360的数据
        time_lat_df = trajDf.iloc[:, [1, 2]].copy().reset_index()
        time_lon_df = trajDf.iloc[:, [1, 3]].copy().reset_index()
        time_sog_df = trajDf.iloc[:, [1, 4]].copy().reset_index()
        time_cog_df = trajDf.iloc[:, [1, 5]].copy().reset_index()
        temp_drop_list_1 = []
        temp_drop_list_2 = []
        for i in range(rowsNum_temp):
            if time_sog_df.iloc[i][2] > 25:
                temp_drop_list_1.append(i)
            if time_cog_df.iloc[i][2] > 360:
                temp_drop_list_2.append(i)
        time_sog_df = time_sog_df.drop(index=temp_drop_list_1)
        time_cog_df = time_cog_df.drop(index=temp_drop_list_2)

        #数据插值
        time_start = trajDf.iloc[0, 1]
        time_end = trajDf.iloc[rowsNum_temp-1, 1]
        t = getTimeRange(time_start, time_end)
        X = time_lat_df.iloc[:, 1].values
        Y = time_lat_df.iloc[:, 2].values
        latArray = cubic_spline_latAndLon(X, Y, t)     #纬度插值结果
        X = time_lon_df.iloc[:, 1].values
        Y = time_lon_df.iloc[:, 2].values
        lonArray = cubic_spline_latAndLon(X, Y, t)     #经度插值结果
        X = time_sog_df.iloc[:, 1].values
        Y = time_sog_df.iloc[:, 2].values
        sogArray = cubic_spline_latAndLon(X, Y, t)     #航速插值结果
        cogArray = cubic_spline_cog(time_cog_df.iloc[:, 1], time_cog_df.iloc[:, 2], t)      #航向插值结果

        #组合插值结果为DataFrame
        for idx in range(len(t)):
            resultDf = resultDf.append(pd.DataFrame({'Timestamp':[t[idx]], 'Latitude':[latArray[idx]], 'Longitude':[lonArray[idx]], 'Sog':[sogArray[idx]], 'Cog':[cogArray[idx]]}), ignore_index=True)
    return resultDf

def preprocessing(date):
    """
    数据预处理启动函数，结果写入csv文件
    :param date: 待读取文件的日期
    :return:
    """
    # rankings_colname = ['Date_Time', 'Timestamp', 'Latitude', 'Longitude', 'Sog', 'Cog']
    # df = pd.read_csv("E:\成山头数据\\2018-01-01\\413501110.csv", header=None, names=rankings_colname)
    # tempDf = preprocessing_each(df)
    # print(tempDf)


    #创建目录
    dir_result_path = "E:\成山头数据\\result\\" + date    #结果存储目录路径
    os.mkdir(dir_result_path)
    #遍历日期文件夹
    dir_path = "E:\成山头数据\\" + date     #待读取文件目录路径
    files = os.listdir(dir_path)
    # temp = 0
    for file in files:
        # temp += 1
        file_path = dir_path + "\\" + str(file)
        rankings_colname = ['Date_Time', 'Timestamp', 'Latitude', 'Longitude', 'Sog', 'Cog']
        df = pd.read_csv(file_path, header=None, names=rankings_colname)
        tempDf = preprocessing_each(df)
        if len(tempDf) == 0:     #轨迹记录数过少，导致插值结果为空
            continue

        #按时间存储预处理，写入文件
        rowsNum_tempDf = tempDf.shape[0]
        for tempDf_index in range(rowsNum_tempDf):
            file_name = dir_result_path + "\\" + str(tempDf.iloc[tempDf_index, 0]) + ".csv"
            with open(file_name, 'a', newline='') as file_writer:
                writer = csv.writer(file_writer)
                mmsi = str(file).split(".")[0]
                data = [mmsi, tempDf.iloc[tempDf_index, 1], tempDf.iloc[tempDf_index, 2], tempDf.iloc[tempDf_index, 3], tempDf.iloc[tempDf_index, 4]]   #[mmsi, latitude, longitude, sog, cog]
                writer.writerow(data)
                file_writer.close()

        print("{}预处理完毕".format(file_path))
        # if temp == 50:
        #     break

def removeDuplicates(df):
    """
    去除轨迹数据中的重复值
    :param df: 待处理轨迹(DataFrame)
    :return: 去除重复值后的轨迹(DataFrame)
    """
    list = []
    rowsNum = df.shape[0]
    for i in range(rowsNum-1):
        if df.iloc[i, 1] == df.iloc[i+1, 1]:
            list.append(i)
    df = df.drop(index=list)
    return df

if __name__ == '__main__':
    preprocessing("2018-01-01")
