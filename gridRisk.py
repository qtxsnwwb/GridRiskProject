import pandas as pd
import os
import sys
import math
import VCRO_Model as vm

def getPreviousRecord(currentTime, mmsi):
    """
    获取指定MMSI船舶的前一时刻记录
    :param currentTime: 当前时刻时间
    :param mmsi: 船舶MMSI
    :return: 获取的记录(Series)
    """
    previousTime = str(currentTime - 300)     #前一时刻时间
    file_path = "E:\成山头数据\\result\\2018-01-01" + "\\" + previousTime + ".csv"
    rankings_colname = ['Mmsi', 'Latitude', 'Longitude', 'Sog', 'Cog']
    df = pd.read_csv(file_path, header=None, names=rankings_colname)
    #遍历每条船舶
    rowsNum_df = df.shape[0]
    for i in range(rowsNum_df):
        if df.iloc[i, 0] == mmsi:
            return df.iloc[i, :]
    #没有找到则递归找再前一个时刻文件
    if previousTime != '0':
        getPreviousRecord(previousTime, mmsi)
    else:
        return pd.Series()

def calVCRO_each(df, time, resultList):
    """
    计算每条船舶的VCRO值
    :param df: 轨迹数据(DataFrame)
    :param time: 当前时刻
    :param resultList: 存储各条船舶VCRO值的List([[mmsi, [VCRO1，VCRO2，....]], ........])
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
                previousRecord_1 = getPreviousRecord(time, df.iloc[i, 0])
                previousRecord_2 = getPreviousRecord(time, df.iloc[j, 0])
                if not previousRecord_1.empty and not previousRecord_2.empty:
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
                        vcro = 1 - math.exp(-vcro / 100)
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

def getRisk(time):
    """
    获取指定时间的全部船舶风险值
    :param time: 指定的时间(int)
    :return: 该时刻全部船舶风险值(List)
    """
    file_path = "E:\成山头数据\\result\\2018-01-01\\" + str(time) + ".csv"
    rankings_colname = ['Mmsi', 'Latitude', 'Longitude', 'Sog', 'Cog']
    df = pd.read_csv(file_path, header=None, names=rankings_colname)
    rowsNum_df = df.shape[0]
    resultList = []     #创建存储各条船舶VCRO值的List([[mmsi, [VCRO1，VCRO2，....]], ........])
    for i in range(rowsNum_df):
        tempList = [df.iloc[i, 0], []]
        resultList.append(tempList)
    riskList = []      #创建存储各条船舶风险值的List([[mmsi, 风险值], ........])
    for i in range(rowsNum_df):
        tempList = [df.iloc[i, 0]]
        riskList.append(tempList)
    #风险值计算
    calVCRO_each(df, time, resultList)      #计算各船VCRO值，并存入resultList
    calRisk_each(resultList, riskList)
    return riskList

if __name__ == '__main__':
    #-----------------------------
    #编程时日期是固定的，后期需修改
    #-----------------------------
    for i in range(900, 86400, 3600):
        for j in range(i, i+3600, 900):
            print(getRisk(j))
        print("------------")
