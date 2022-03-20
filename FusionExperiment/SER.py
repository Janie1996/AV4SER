# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_29_LSTM.py
@author: Wei Jie
@date: 2022/1/29
@description:
"""

import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import warnings

import pretrainedmodels
warnings.filterwarnings('ignore')

import time
from torch.optim import lr_scheduler
import argparse
import torch
import torch.nn as nn
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2022_01_11_dataload as load_dataset_audio
from VideoDataset import wj_2022_01_02_dataload as load_dataset_video
from DomainModels import wj_2022_01_08_ResNet_Domain as DomainModel
# from DomainModels import wj_2022_01_17_proposed_Domain as DomainModel
from AudioModels import wj_2022_01_08_DSCASA_Transformer as AudioModel
from FeatureFusionModels import wj_2022_01_08_LSTM as FusionModel
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2022_01_03_name_logfbank_dict_40d.pickle', help='root-path for audio')
parser.add_argument('--videoRoot', default='E:/Dataset/IEMOCAP_full_release/faceallavi/', help='root-path for video')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=16, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--step_size', type=float, default=10, help='step of learning rate changes')
parser.add_argument('--gamma', type=float, default=0.5, help='weight decay of learning rate')
parser.add_argument('--cudaNum',  default='0', help='the cuda number to be used')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_channel',  default=1, type=int,help='the channel number of input feature')
parser.add_argument('--length',  default=5, type=int,help='the clip number of input audio feature')
parser.add_argument('--frameWin',  default=224, type=int,help='the frame number of each input melspec feature map')
parser.add_argument('--featureDim',  default=40, type=int,help='the feature dimension of audio input')


opt = parser.parse_args()
print(opt)



from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np


def test_UA_WA(test_loader,model_D,model_A,model_F):

    model_D.eval()
    model_A.eval()
    model_F.eval()


    right, right1, right2, right3, right4 = 0, 0, 0,0,0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y1,trace_pre_y2,trace_pre_y3,trace_pre_y4=[],[],[],[],[]

    with torch.no_grad():
        for i, (video, index) in enumerate(test_loader):
            video = video.float().cuda()  # batch,10,3,224,224
            label = index.cuda()

            trace_y.append(label.cpu().detach().numpy())

            pre,dom,fea_d,emo_d = model_D(video)
            pre1,fea_a,emo_a = model_A(video)

            emo_d = emo_d.unsqueeze(3)
            emo_a = emo_a.unsqueeze(3)

            video = np.concatenate((fea_a.cpu().detach().numpy(), fea_d.cpu().detach().numpy()), axis=2)
            video = torch.from_numpy(video).cuda()

            emo_audio = np.concatenate((emo_a.cpu().detach().numpy(), emo_d.cpu().detach().numpy()), axis=3)
            maxpool = torch.nn.MaxPool2d((1, 2))
            emo_audio = torch.from_numpy(emo_audio)
            emo_audio = maxpool(emo_audio).squeeze(3).cuda()

            pre2,pre3,pre4 = model_F(video,emo_audio)

            _, preds = torch.max(pre.data, 1)
            right += float(torch.sum(preds == label.data))
            trace_pre_y.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre1.data, 1)
            right1 += float(torch.sum(preds == label.data))
            trace_pre_y1.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre2.data, 1)
            right2 += float(torch.sum(preds == label.data))
            trace_pre_y2.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre3.data, 1)
            right3 += float(torch.sum(preds == label.data))
            trace_pre_y3.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre4.data, 1)
            right4 += float(torch.sum(preds == label.data))
            trace_pre_y4.append(preds.cpu().detach().numpy())

            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y3 = np.concatenate(trace_pre_y3)



        weighted_accuracy3 = accuracy_score(trace_y, trace_pre_y3)
        unweighted_accuracy3 = balanced_accuracy_score(trace_y, trace_pre_y3)


    print('Test *Prec@Fusion   WA:{:.3f};   UA:{:.3f};   '.format(weighted_accuracy3,unweighted_accuracy3))

    return weighted_accuracy3,unweighted_accuracy3

def main():


    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))
    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)
    train_loader_audio, test_loader_audio = load_dataset_audio.AudioLoadData(opt.audioFeatureRoot,train_list, test_list, name_emotionLabel,
                                                                           opt.batchSize,framelen=opt.length,winlen=opt.frameWin)

    model_D = DomainModel.resnet18(classes=opt.nClasses,channel=1)
    checkpoint = torch.load("Checkpoint/AVDAL.pth")
    model_D.load_state_dict(checkpoint)

    model_A = AudioModel.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=opt.frameWin,width=opt.featureDim)
    checkpoint = torch.load("Checkpoint/MDSCM.pth")
    model_A.load_state_dict(checkpoint)

    model_F = FusionModel.EmotionRecog(classes=opt.nClasses,featureDim=768+512)
    checkpoint = torch.load("Checkpoint/FF.pth")
    model_F.load_state_dict(checkpoint)

    model_D.cuda()
    model_A.cuda()
    model_F.cuda()

    acc, acc1 = test_UA_WA(test_loader_audio, model_D,model_A,model_F)



if __name__=="__main__":
    main()