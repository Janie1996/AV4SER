# -*- coding: UTF-8 -*-
"""
@file:wj_2021_12_26_Scale.py
@author: Wei Jie
@date: 2021/12/26
@description:  数据归一化处理
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

# scale data
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_Mel(X_train):

    #X_train = np.expand_dims(X_train, 1)  # 添加通道
    scaler = StandardScaler()

    b, c, h, w = X_train.shape
    X_train = np.reshape(X_train, newshape=(b, -1))
    X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train, newshape=(b, c, h, w))

    return X_train


if __name__ == "__main__":

    x=np.ones([55,120,3,128])
    y=scale_Mel(x)


    print()