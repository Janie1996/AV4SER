# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_02_dataload.py
@author: Wei Jie
@date: 2022/1/2
@description:
"""
import sys
import os

from PIL import Image

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
from VideoDataset import wj_2022_01_02_dataset as dataset
import cv2

def readAllVideos(chunk_wavs,path_names, transform, length,channel,feature):

    name_feature = torch.ones([len(path_names),length,channel,224,feature])
    i=0
    for path in path_names:
        file_path = chunk_wavs + path + '.mp4'

        video = get_video(file_path, transform, length,channel,feature)

        name_feature[i]=video
        i+=1
    return name_feature



# 读取视频数据
def get_video(file_path, transform, length,channel,feature):
    # length=5
    # index_name='Ses04M_script03_2_M020'
    vc = cv2.VideoCapture(file_path)  # 读入视频文件
    video_length = int(vc.get(7))
    video = torch.ones(length, channel, 224, feature)

    num = 0
    # 均匀采样视频帧
    # print(index_name)
    if (video_length < length):
        for i in range(video_length):
            rval, frame = vc.read()
            if(channel==1):
                frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=2)  # 灰度化  [640,1200]
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            video[i, :, :, :] = frame
            num += 1
        for j in range(num, length):
            video[j, :, :, :] = frame
    else:
        step = int(video_length / length) - 1
        for i in range(length):
            # print(i)
            rval, frame = vc.read()
            if(channel==1):
              frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),axis = 2)
            # cv2.imshow('OriginalPicture', frame)
            # cv2.waitKey()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            video[i, :, :, :] = frame
            for j in range(step):
                rval, frame = vc.read()
    vc.release()

    # if(index==1):
    #     print(sum(sum(sum(sum(video)))))

    return video


def VideoLoadData(datadir, train_list, test_list, name_emotion,batchsize,length,channel,feature):

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, feature)),
                                    transforms.Normalize((0.5), (0.5))
                                    ])
    # local
    # transform = transforms.Compose([
    #
    #     transforms.Resize((224, 128)),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5), (0.5))
    #
    # ])
    videos=readAllVideos(datadir,name_emotion.keys(),transform,length,channel,feature)

    train_dataset = dataset.VideoDataset(
        video=videos,
        video_list=train_list,
        name_emotin=name_emotion,
    )

    test_dataset = dataset.VideoDataset(
        video=videos,
        video_list=test_list,
        name_emotin=name_emotion,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False)

    return train_loader,test_loader


if __name__ == "__main__":
    import pickle
    from VideoDataset import wj_2022_01_01_utils as utils
    #from VideoDataset i
    name_emotionLabel = pickle.load(open('DATA/name_emotionLable_dict_noblack.pickle', 'rb'))


    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel), testNum=1098)

    train_loader, test_loader = VideoLoadData('E:/Dataset/IEMOCAP_full_release/faceallavi/', train_list, test_list,
                                                           name_emotionLabel, 16,
                                                           length=5)
    print()