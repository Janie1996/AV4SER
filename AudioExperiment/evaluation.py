# -*- coding: UTF-8 -*-
"""
@file:evaluation.py
@author: Wei Jie
@date: 2022/2/20
@description: 只进行测试
"""

import random
import sys
import os
import warnings

from torch.backends import cudnn

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
warnings.filterwarnings('ignore')


import argparse
import torch
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2022_01_11_dataload as load_dataset
from AudioModels import wj_2022_01_08_DSCASA_Transformer as Model
import pickle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

# np.random.seed(12)
# torch.manual_seed(12)  #为CPU设置种子用于生成随机数，以使得结果是确定的   　　
# torch.cuda.manual_seed(12) #为当前GPU设置随机种子；  　　
# cudnn.deterministic = True

num=0

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2022_01_03_name_logfbank_dict_40d.pickle', help='root-path for audio')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=32, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.08, help='learning rate, default=0.0002')
parser.add_argument('--step_size', type=float, default=80, help='step of learning rate changes')
parser.add_argument('--gamma', type=float, default=0.5, help='weight decay of learning rate')
parser.add_argument('--cudaNum',  default='0', help='the cuda number to be used')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_channel',  default=1, type=int,help='the channel number of input feature')
parser.add_argument('--length',  default=5, type=int,help='the clip number of input audio feature')
parser.add_argument('--frameWin',  default=224, type=int,help='the frame number of each input melspec feature map')


opt = parser.parse_args()
print(opt)



def test_UA_WA(test_loader,model):

    model.eval()
    right, right1, right2 = 0, 0, 0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y2=[],[]

    with torch.no_grad():
        for i, (video, index) in enumerate(test_loader):
            video = video.float().cuda()  # batch,10,3,224,224
            label = index.cuda()

            trace_y.append(label.cpu().detach().numpy())

            pre, fea,score = model(video)

            _, preds = torch.max(pre.data, 1)
            right += float(torch.sum(preds == label.data))
            trace_pre_y.append(preds.cpu().detach().numpy())

            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y = np.concatenate(trace_pre_y)


        weighted_accuracy = accuracy_score(trace_y,trace_pre_y)
        unweighted_accuracy = balanced_accuracy_score(trace_y,trace_pre_y)
        cm= confusion_matrix(trace_y,trace_pre_y)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Test *Prec@Audio WA:{:.3f};  UA:{:.3f}'.format(weighted_accuracy,unweighted_accuracy))
    return weighted_accuracy,unweighted_accuracy

def main():

    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))

    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)

    train_loader, test_loader = load_dataset.AudioLoadData(opt.audioFeatureRoot,train_list, test_list, name_emotionLabel, opt.batchSize,framelen=opt.length,winlen=opt.frameWin)

    model = Model.inno_model1(input_channel=opt.input_channel, classes=opt.nClasses, height=224, width=40)
    checkpoint = torch.load("Checkpoint/MDSCM.pth")
    model.load_state_dict(checkpoint)
    model.cuda()
    acc, acc1 = test_UA_WA(test_loader, model)

if __name__=="__main__":
    main()