# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_08_LSTM.py
@author: Wei Jie
@date: 2022/1/8
@description:
"""
# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_07_LSTM.py
@author: Wei Jie
@date: 2022/1/7
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import torch.nn as nn
import numpy as np


class RNN_feature(nn.Module):
    def __init__(self,in_dim,hidden_dim,classes):
        super(RNN_feature, self).__init__()
        self.lstm=nn.LSTM(in_dim,hidden_dim,1,batch_first=True)
        self.fc=nn.Linear(hidden_dim,classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # 此时可以从out中获得最终输出的状态h
        x = out[:, -1, :]
        # x = h_n[-1, :, :]
        x = self.fc(x)
        return x


class EmotionRecog(nn.Module):
    def __init__(self,classes,featureDim):
        super(EmotionRecog, self).__init__()
        self.classes=classes
        self.featureDim = featureDim

        self.lstm1 = RNN_feature(self.featureDim,128,self.classes)
        self.lstm2 = RNN_feature(self.classes,128,self.classes)

        # self.temp_lstm1 = nn.LSTM(input_size=self.featureDim, hidden_size=128, num_layers=1, batch_first=True,
        #                           bidirectional=False)
        # self.temp_lstm2 = nn.LSTM(input_size=128, hidden_size=self.classes, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self,feature,score):


        output1=self.lstm1(feature)
        output2=self.lstm2(score)


        return 0.7*output1+0.3*output2,output1,output2



if __name__ == "__main__":


    import torch
    a=torch.ones([2,5,2048])
    b=torch.ones([2,5,512])

    c = torch.ones([2, 5, 4,1])
    c[0][0][1][0]=12
    d = torch.ones([2, 5, 4,1])
    d[0][0][1][0] = 122

    input1 = np.concatenate((a, b), axis=2)
    input1 = torch.from_numpy(input1)

    input2 = np.concatenate((c, d), axis=3)
    input2 = torch.from_numpy(input2)

    soft=nn.MaxPool2d((1,2))
    input2 = soft(input2).squeeze(3)

    model = EmotionRecog(4,2048+512)
    output,output1,output2 = model(input1,input2)

    print()