# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_11_dataload.py
@author: Wei Jie
@date: 2022/1/11
@description:将分段好的feature加载到内存， 每次训练只需要检索
"""

from __future__ import print_function
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import cv2
import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
from AudioDataset import wj_2021_12_27_dataset as dataset

import pickle

#
# # 只给了audio文件位置， 每次训练特征都是从硬盘读取
# def AudioLoadData(datadir, train_list, test_list, name_emotion, batchsize):
#     train_dataset = dataset.AudioDataset(
#         audio_root=datadir,
#         audio_list=train_list,
#         name_emotin=name_emotion,
#     )
#
#     test_dataset = dataset.AudioDataset(
#         audio_root=datadir,
#         audio_list=test_list,
#         name_emotin=name_emotion,
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batchsize, shuffle=False)
#
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batchsize, shuffle=False)
#
#     return train_loader, test_loader
#
#
# # 将原始特征加载到内存， 每次训练进行分段处理
# def AudioLoadDataRam(datadir, train_list, test_list, name_emotion, batchsize, framelen=10, winlen=224):
#     train_dataset = dataset_ram.AudioDataset(
#         feature_root=datadir,
#         audio_list=train_list,
#         name_emotin=name_emotion,
#         framelen=framelen,
#         winlen=winlen,
#     )
#
#     test_dataset = dataset_ram.AudioDataset(
#         feature_root=datadir,
#         audio_list=test_list,
#         name_emotin=name_emotion,
#         framelen=framelen,
#         winlen=winlen,
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batchsize, shuffle=False)
#
#     test_loader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=batchsize, shuffle=False)
#
#     return train_loader, test_loader


def get_audio_split(allfeatures, framenum, winlen):
    feature_dataset = []

    for index in range(len(allfeatures)):
        # print(index)

        mfcc = allfeatures[index]  # 当前音频的特征： frame,3,filter
        # 特征前后分别加 winlen/2帧 特征
        padding_front = np.zeros((int(winlen / 2), 3, mfcc.shape[2]))
        padding_back = np.zeros((int(winlen / 2), 3, mfcc.shape[2]))
        front = np.vstack((padding_front, mfcc))
        mfcc = np.vstack((front, padding_back))

        # sequence_list=np.zeros((framenum,winlen,3,mfcc.shape[2]))

        sequence_list = np.zeros((framenum, 3, 224, 224))
        sequence_idx = 0  # 特征分段数  10
        winstep = int(mfcc.shape[0] / framenum)
        idx = 0

        while (idx < mfcc.shape[0] - winlen):

            middle = mfcc[idx:idx + winlen, :, :].transpose((1, 0, 2))  # 3,win,filter

            for j in range(3):
                sequence_list[sequence_idx][j] = cv2.resize(middle[j], (224, 224), interpolation=cv2.INTER_LINEAR)

            # sequence_list[sequence_idx]=mfcc[idx:idx+winlen,:,:]
            idx += winstep
            sequence_idx += 1
            if (sequence_idx == framenum):
                break
        features = torch.from_numpy(sequence_list)
        feature_dataset.append(features)  # 所有分段完的特征  5531,N,3,224,224

    return feature_dataset


def get_audio_split_withoutResize(allfeatures, framenum, winlen):
    """

    :param allfeatures:  特征list: audioNum,OriframeLenth,channel,feature_dim
    :param framenum:     分段的段数 length
    :param winlen:       分段后的 feature map 维度, 每段的帧数
    :return:    特征list: audioNum,length,channel,winlen,feature_dim
    """


    feature_dataset = []
    channel = allfeatures[0].shape[1]

    for index in range(len(allfeatures)):
        # print(index)

        mfcc = allfeatures[index]  # 当前音频的特征： frame,3,filter

        if (mfcc.shape[0] < framenum + winlen):
            # 特征前后分别加 winlen/2帧 特征
            padding_front = np.zeros((int(winlen / 2), channel, mfcc.shape[2]))
            padding_back = np.zeros((int(winlen / 2), channel, mfcc.shape[2]))
            front = np.vstack((padding_front, mfcc))
            mfcc = np.vstack((front, padding_back))

        # sequence_list=np.zeros((framenum,winlen,3,mfcc.shape[2]))

        sequence_list = np.zeros((framenum, channel, winlen, mfcc.shape[2]))
        sequence_idx = 0  # 特征分段数  10
        winstep = int(mfcc.shape[0] / framenum)
        idx = 0

        while (idx < mfcc.shape[0] - winlen):

            middle = mfcc[idx:idx + winlen, :, :].transpose((1, 0, 2))  # 3,win,filter
            sequence_list[sequence_idx] = middle
            # for j in range(3):
            #     sequence_list[sequence_idx][j] = cv2.resize(middle[j], (224, 224), interpolation=cv2.INTER_LINEAR)

            # sequence_list[sequence_idx]=mfcc[idx:idx+winlen,:,:]
            idx += winstep
            sequence_idx += 1
            if (sequence_idx == framenum):
                break
        features = torch.from_numpy(sequence_list)
        feature_dataset.append(features)  # 所有分段完的特征  5531,N,3,224,224

    return feature_dataset


# 将分段好的feature加载到内存， 每次训练只需要检索
def AudioLoadData(datadir, train_list, test_list, name_emotion, batchsize, framelen=10, winlen=128):

    """

    :param datadir: 特征文件位置
    :param train_list:  训练数据下标list
    :param test_list:
    :param name_emotion: WAV文件名 + label的存储字典
    :param batchsize:
    :param framelen:  分段数
    :param winlen:   每张feature map包含的帧数
    :return:
    """
    allfeatures = list(pickle.load(open(datadir, 'rb')).values())
    # feature_dataset = get_audio_split(allfeatures, framelen, winlen)
    feature_dataset = get_audio_split_withoutResize(allfeatures, framelen, winlen)

    train_dataset = dataset.AudioDataset(
        feature=feature_dataset,
        audio_list=train_list,
        name_emotin=name_emotion,
    )

    test_dataset = dataset.AudioDataset(
        feature=feature_dataset,
        audio_list=test_list,
        name_emotin=name_emotion,

    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32, shuffle=False)


    return train_loader, test_loader



if __name__ == '__main__':

    import pickle
    import random

    """audio"""

    # datadir='E:/Dataset/IEMOCAP_full_release/mfccfeatures/'
    # name_emotionLabel=pickle.load(open('../Data/name_emotionLabel_dict.pickle','rb'))
    #
    # test_list=random.sample(range(0,len(name_emotionLabel)),int(553*3))
    # train_list=list(set(np.arange(len(name_emotionLabel)))-set(test_list))
    #
    # batchsize=16
    #
    # train_loader,test_loader = AudioLoadData(datadir,train_list, test_list,name_emotionLabel, batchsize)
    #
    #
    # for i, (video, index) in enumerate(test_loader):
    #     audio=video.cuda()  # batch,10,3,224,224
    #     label=index.cuda()
    #     print(i)
    #     print()

    """video"""

    datadir = '../Data/2022_01_01_name_delta_logfbank_dict.pickle'
    name_emotionLabel = pickle.load(open('../Data/name_emotionLabel_dict.pickle', 'rb'))

    test_list = random.sample(range(0, len(name_emotionLabel)), int(553 * 3))
    train_list = list(set(np.arange(len(name_emotionLabel))) - set(test_list))

    batchsize = 16

    train_loader, test_loader,fea_loader = AudioLoadData(datadir, train_list, test_list, name_emotionLabel, batchsize)

    for i, (video, index) in enumerate(test_loader):
        video = video.cuda()  # batch,10,3,224,224
        label = index.cuda()
        print(i)
        print()