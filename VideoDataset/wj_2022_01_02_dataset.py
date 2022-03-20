# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_02_dataset.py
@author: Wei Jie
@date: 2022/1/2
@description:继承dataset类，用于数据集读取；目前初始化只保存了数据路径
            dataload在训练过程中读取每个视频内容
            优： 不占用大量内存
            缺： 每一轮训练都要从硬盘读取数据，耗时
"""

import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

try:
    import cPickle as pickle
except:
    import pickle

def get_video(index,allfeatures,name_emotion):


    labels = list(name_emotion.values())
    index_label = labels[index]
    features = allfeatures[index]

    return features,index_label

# video数据集
class VideoDataset(data.Dataset):

    def __init__(self, video, video_list, name_emotin):
        self.video = video
        self.video_list = video_list
        self.name_emotin = name_emotin


    def __getitem__(self, index):
        #print(index)
       # print(self.video_list[index])
        video,label=get_video(self.video_list[index],self.video,self.name_emotin)
        return video, label

    def __len__(self):
        return len(self.video_list)

if __name__=="__main__":

    print(1)
