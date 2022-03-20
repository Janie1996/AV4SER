# -*- coding: UTF-8 -*-
"""
@file:wj_2021_12_27_utils.py
@author: Wei Jie
@date: 2021/12/27
@description: 划分训练集、测试集函数，提供了各种不同的划分方式
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import random
import numpy as np

"""
5531 utterances 
randomly chose 80% for training and 20% for testing

70.07%   WA
70.67%   UA

"""


def randomSplit(length,testNum):

    # x= np.random.randint(10000)
    # print(x)
    random.seed(6990)
    # length=100
    # testNum=20
    test_list = random.sample(range(0, length), int(testNum))
    #print(test_list)
    #print(sum(test_list))
    train_list = list(set(np.arange(length)) - set(test_list))
    # np.random.shuffle(train_list)

    return train_list,test_list #,x

def leaveOneSession(name_emotionLabel,session):

    test_list = []
    names = list(name_emotionLabel.keys())
    for i in range(len(names)):
        if session in names[i]:
            test_list.append(i)

    train_list = list(set(np.arange(len(names))) - set(test_list))
    np.random.shuffle(train_list)
    return train_list,test_list



def leaveOneSpeaker(name_emotionLabel,session,man):


    test_list = []
    names = list(name_emotionLabel.keys())
    for i in range(len(names)):
        if session in names[i] and man in names[i]:
            test_list.append(i)

    train_list = list(set(np.arange(len(names))) - set(test_list))
    np.random.shuffle(train_list)
    return train_list, test_list


if __name__ == '__main__':
    randomSplit(5531,6)

    x=np.zeros([3,2])
    x[2][1]=3
    x1=np.ones(3)
    x1[2]=10
    state = np.random.get_state()
    np.random.shuffle(x)
    print(x)

    np.random.set_state(state)
    np.random.shuffle(x1)
    print(x1)

    np.random.shuffle(x)
    print(x)
    randomSplit(5,2)