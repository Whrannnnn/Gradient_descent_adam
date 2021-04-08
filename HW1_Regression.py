"""
author: Andrew_Wang
date:(start): 2021/4/6 20:26
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import Tese_set
import csv
import torch

# 任务：由前9个小时的18个特征，预测第10个小时的PM2.5（PM2.5是第10个特征）
# 训练数据：train.csv是12个月，每个月取20天，每天24小时的数据，每个小时又18个特征  行: 12*20*18 列：24
# 测试数据：有18个特征的前9个小时的数据
# 参考链接：https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=dcOrC4Fi-n3i


def load_data():
    # # 加载训练数据
    # data = pd.read_csv('train.csv', encoding='big5')
    # data = data.iloc[:, 3:]
    # data[data == 'NR'] = 0
    # raw_data = data.to_numpy()  # 4320*24
    #
    # # 提取特征，将原始的4320*18，按照每个月分组成为12个18(特征)*480(小时)
    # month_data = {}
    # for month in range(12):
    #     sample = np.empty([18, 480])  # 每月份有：20*24
    #     for day in range(20):
    #         sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    #     month_data[month] = sample
    #
    # # 提取特征，每个月有480个小时，每9个小时的数据成一组，每个月一共有471组数据；所以行为471*12；列是9*18的特征(一个小时的18个特征*9个小时)
    # x = np.empty([12 * 471, 18 * 9], dtype=float)  # input
    # y = np.empty([12 * 471, 1], dtype=float)  # output
    # for month in range(12):
    #     for day in range(20):
    #         for hour in range(24):
    #             if day == 19 and hour > 14:
    #                 continue
    #             # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9) ，reshape(1,-1)转换成一行，转换成一列reshape(-1,1)
    #             x[month * 471 + day * 24 + hour, :] = month_data[month][:,
    #                                                   day * 24 + hour: day * 24 + hour + 9].reshape(1,
    #                                                                                                 -1)
    #             y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
    #
    # # 归一化处理 axis=0:压缩行，对列求均值
    # mean_x = np.mean(x, axis=0)  # 18 * 9
    # std_x = np.std(x, axis=0)  # 18 * 9
    # for i in range(len(x)):  # 12 * 471
    #     for j in range(len(x[0])):  # 18 * 9
    #         if std_x[j] != 0:
    #             x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    # return x, y, mean_x, std_x
    # 读取数据
    data = pd.read_csv('train.csv', encoding='big5')
    # 不取前三列
    data = data.iloc[:, 3:]
    # 清洗数据，data == 'NR的赋值为0
    data[data == 'NR'] = 0
    # 将DataFrame转换为NumPy数组,才能进行相应运算
    raw_data = data.to_numpy()

    # 提取特征，将原始的4320*18，按照每个月分组成为12个18(特征)*480(小时)
    month_data = {}
    for month in range(12):
        # 初始化每月：18X480
        sample = np.empty([18, 480])
        for day in range(20):
            sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
        # 将12组18*480数组以放入一个字典中
        month_data[month] = sample

    # 提取特征，每个月有480个小时，每9个小时的数据成一组，每个月一共有471组数据；所以行为471*12；列是9*18的特征(一个小时的18个特征*9个小时)
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if hour > 14 and day == 19:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 20 + hour:day * 20 + hour + 9].reshape(
                    1, -1)
                y[month * 471 + day * 24 + hour, :] = month_data[month][9, day * 20 + hour + 9]

    # 归一化处理 axis=0:压缩行，对列求均值
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(len(x)):  # 12 * 471
        for j in range(len(x[0])):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    # print(x)
    return x, y, mean_x, std_x


def train_model(x, y, learning_rate=0.01, iter_time=1000, eps=0.0000000001):
    # ***********Training_Adagrad***********
    # dim = 18 * 9 + 1
    # w = np.zeros([dim, 1])
    # x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # np.concatenate追加列
    #
    # adagrad = np.zeros([dim, 1])
    # # 每次迭代loss更新，所以与迭代次数相同
    # loss = np.zeros(([iter_time, 1]))
    # for t in range(iter_time):
    #     loss[t] = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
    #     if t % 100 == 0:
    #         print(str(t) + ":" + str(loss[t]))
    #     gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1 x.transpose()表示转置
    #     adagrad += gradient ** 2
    #     w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    # np.save('weight.npy', w)
    # plt.plot(loss)
    # plt.show()
    # 初始化

    # ***********Training_Adam***********
    # m 为 12 * 471， dim为 18 * 9
    dim = 18 * 9 + 1
    theta = np.zeros(dim)  # 参数
    learning_rate = 20  # 学习率
    momentum = 0.1  # 冲量
    iter_time = 20000  # 迭代次数
    threshold = 0.0001  # 停止迭代的错误阈值
    error = 0  # 初始错误为0

    b1 = 0.9  # 算法作者建议的默认值
    b2 = 0.999  # 算法作者建议的默认值
    e = 0.00000001  # 算法作者建议的默认值
    mt = np.zeros([dim, 1])  # 一阶原点矩
    vt = np.zeros([dim, 1])  # 二阶原点矩

    loss = np.zeros(([iter_time, 1]))
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  # np.concatenate追加列

    for i in range(iter_time):
        loss[i] = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1 x.transpose()表示转置
        mt = mt * b1 + (1 - b1) * gradient
        vt = vt * b2 + (1 - b2) * (gradient**2)
        mtt = mt / (1 - (b1**(i + 1)))
        vtt = vt / (1 - (b2**(i + 1)))
        w = w - learning_rate * mtt / (np.sqrt(vtt) + e)
        if i % 100 == 0:
          print(str(i) + ":" + str(loss[i]))



        # if abs(error) <= threshold:
        #     break

    np.save('weight.npy', w)
    plt.plot(loss)
    plt.show()


def pre_test(x, y):
    # ***********Pre-Testing***********
    x_train_set = x[: math.floor(len(x) * 0.8), :]  # 约为4521
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    x_validation = x[math.floor(len(x) * 0.8):, :]  # 约为1131
    y_validation = y[math.floor(len(y) * 0.8):, :]

    w = np.load('weight.npy')  # 提取训练好的参数w
    x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis=1).astype(float)
    ans_y = np.dot(x_validation, w)
    loss = np.sqrt(np.sum(np.power(ans_y - y_validation, 2)) / 1131)
    print("误差为{}".format(loss))  # 输出误差


# ***********Predict**********
def test_model(mean_x, std_x):
    testdata = pd.read_csv('test.csv', header=None, encoding='big5')
    test_data = testdata.iloc[:, 2:].copy()
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18 * 9], dtype=float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)

    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)
    # 数据导入



if __name__ == '__main__':
    x, y, mean_x, std_x = load_data()
    train_model(x, y)
    pre_test(x, y)
    test_model(mean_x, std_x)
