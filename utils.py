#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# 绘图设置
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# 随机数设置
seed = 12
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)

def myplot(data, xlabel=None, ylabel=None, label=None, linewidth=1, color='green'):
    """绘图函数"""
    # 创建图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 设置刻度
    xticks = np.arange(0, len(data), len(data) / 10)
    yticks = np.arange(1.2 * min(data), 1.2 * max(data), (max(data) - min(data)) / 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)    
    # 绘制图形
    ax.plot(data, linewidth=linewidth, color=color, label=label)
    # 设置x轴标签
    if xlabel:
        ax.set_xlabel(xlabel)
    # 设置y轴标签
    if ylabel:
        ax.set_ylabel(ylabel)
    # 添加图例
    if label:
        ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     fig.autofmt_xdate(rotation=90)
    # 返回图形对象
    return fig

def read_water_data():
    """读取数据并且去除缺失值"""
    data = pd.read_csv('./data/water_data.csv', index_col=0, parse_dates=True)
    data = data.iloc[0:-5]
    columns = []
    columns.append('水位高度')
    for i in range(1, 30):
        columns.append('S' + str(i))
    columns.append('闸1')
    columns.append('闸2')
    data.columns = columns
    return data


def del_zero_column(data, ratio=0.8):
    """删除含0值大于ratio的列"""
    zero_counts = data.eq(0).sum()
    del_columns = zero_counts[zero_counts > len(data) * ratio].index
    data = data.drop(columns=del_columns)
    return data


def read_part_water_data(ratio=0.8):
    """获取删除0值大于ratio之后的数据"""
    data = read_water_data()
    return del_zero_column(data, ratio)


"""数据预处理函数"""
def create_datarray(data, n_past, multiple=False):
    """生成时间序列数组"""
    features, labels = [], []
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    for i in range(n_past, len(data)):
        labels.append(data.iloc[i, 0])
        if multiple: 
            features.append(data.iloc[i - n_past:i, 0:].values)
        else: 
            features.append(data.iloc[i - n_past:i, 0].values)
    return np.array(features), np.array(labels)    

def create_dataset(data, n_past, multiple=False):
    """根据数组生成dataset"""
    features, labels = create_datarray(data, n_past, multiple)
    features_tensor, labels_tensor = torch.tensor(features).float(), torch.tensor(labels).float()
    return TensorDataset(features_tensor, labels_tensor)

def create_dataloader(data, n_past, batch_size, shuffle=False, drop_last=True, multiple=False):
    """根据dataset生成dataloader"""
    return DataLoader(
        dataset=create_dataset(data, n_past, multiple),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

def create_modify_datarray(data, lag):
    """根据dataarray生成对应的修正模型数组"""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    modify_features, modify_labels = [], []
    for i in range(lag, len(data)):
        modify_labels.append(data.iloc[i, 0]) # 当天的水位高度
        modify_features.append(data.iloc[i - lag:i + 1, 0:].values) # 历史lag天和当天的水位高度
    modify_features, modify_labels = np.array(modify_features), np.array(modify_labels)
    modify_features = modify_features.reshape(modify_features.shape[0], -1)
    return modify_features, modify_labels

def divide_dataset(data, ratio=0.8):
    """划分测试集和训练集"""
    train_size = int(ratio * len(data))
    data_train, data_test = data.iloc[:train_size], data.iloc[train_size:]
    return data_train, data_test

def sigma_criterion(data):
    """利用3sigma原则找出异常值"""
    mean, std = data.mean(), data.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    out_idx = np.where((data < lower_bound) | (data > upper_bound))[0]
    outliers = data[out_idx]
    return outliers, out_idx

def train_modify_model(features, labels, predictions, 
                           out_idx = None, n_past=7, lag=3, kernel='rbf', eps=0.01):
    """训练修正模型"""
    """
        params:（均归一化过）
        features:修正模型使用的特征，即若干天合并的包括水位高度的所有特征（包括当天）
        labels:预测模型的标签
        predictions:训练集上的预测值
        n_past:训练预测模型时使用的天数
        lag:训练修正模型时往后看的天数
        out_idx:使用criterion从训练集上选取的异常值坐标
        kernel,eps:SVR模型的相关参数
        
        return:
        svr_modify_model:svr修正模型
    """
    init_days = n_past - lag
    # 原训练集上out_idx1和修正模型训练集上out_idx2之间的关系：out_idx2 = out_idx1 + init_days
    # 获取diff_labels差分标签
    if isinstance(out_idx, np.ndarray):
        diff_labels = (labels[init_days:] - predictions)[out_idx]
        new_features = features[out_idx + init_days]
    else:
        diff_labels = labels[init_days:] - predictions
        new_features = features[init_days:]
    
    svr_modify_model = SVR(kernel=kernel, epsilon=eps)
    svr_modify_model.fit(new_features, diff_labels)
    
    return svr_modify_model

def predict_with_bootstrap(X_train, y_train, n_bs=1000):
    svr_bs = SVR()

    # 初始化预测结果列表
    predictions = []

    # 进行Bootstrap抽样和预测
    for i in range(n_bs):
        # 从原始数据集中有放回地抽样，生成Bootstrap样本
        bootstrap_sample_indices = np.random.choice(range(len(X_train)), size=len(X_train), replace=True)
        bootstrap_sample_X = X_train[bootstrap_sample_indices]
        bootstrap_sample_y = y_train[bootstrap_sample_indices]

        # 使用SVR模型进行训练和预测
        svr_bs.fit(bootstrap_sample_X, bootstrap_sample_y)
        y_pred = svr_bs.predict(X_train)

        # 将预测结果添加到列表中
        predictions.append(y_pred)
    return predictions

def bootstrap_sigma_criterion(predictions, y_train, conf_level = 0.95):
    # 计算预测结果的均值和标准差
    predictions_mean = np.mean(predictions, axis=0)
    predictions_std = np.std(predictions, axis=0)

    # 计算置信区间的上下限
    z_score = stats.norm.ppf(conf_level, loc=0, scale=1)  # 对应于置信度p的z值
    lower_bound = predictions_mean - z_score * predictions_std
    upper_bound = predictions_mean + z_score * predictions_std

    out_idx = np.where((y_train < lower_bound) | (y_train > upper_bound))[0]
    outliers = y_train[out_idx]
    return outliers, out_idx, lower_bound, upper_bound

def bootstrap_percent_criterion(predictions, y_train, lp=2.5, up=97.5):
    if not isinstance(predictions, np.ndarray):
        result = np.array(predictions)
    else:
        result = predictions
        
    lower_bound = np.percentile(result, lp, axis=0)
    upper_bound = np.percentile(result, up, axis=0)

    out_idx = np.where((y_train < lower_bound) | (y_train > upper_bound))[0]
    outliers = y_train[out_idx]
    return outliers, out_idx, lower_bound, upper_bound