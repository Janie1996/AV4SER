# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_08_DSCASA_Transformer.py
@author: Wei Jie
@date: 2022/1/8
@description:传统CNN + 深度可分离卷积 + CASA + transforme  , 返回所有length的score和feature
"""

import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import torch
import torch.nn as nn
import torch.nn.functional as F


# 深度可分离卷积  ==  逐通道卷积 + kernel 为 1 卷积
class SeperableConv2d(nn.Module):

    #  An “extreme” version of our Inception module, with one spatial convolution per output channel of the 1x1 convolution.
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size,
                                   groups=input_channels,
                                   bias=False,
                                   **kwargs
                                   )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio


        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)  # batch, channel
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool) # batch, channel
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)
        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 3):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class inno_model1(nn.Module):
    def __init__(self, input_channel, classes,height,width):
        super(inno_model1, self).__init__()

        self.batch_size = 0
        self.classes = classes
        self.channel =input_channel
        self.height = height
        self.width = width

        # 传统卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 深度可分离卷积
        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 残差结构
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )
        self.drop = nn.Dropout(p=0.5, inplace=True)


        # CASA
        self.c_attention = ChannelAttention(256)
        self.s_attention = SpatialAttention()

        """
        128 --> 2048  
        40  -->768
        80  -->1280
        """
        if(width==128):
            self.featureDim = 2048
        elif(width==80):
            self.featureDim = 1280
        elif(width==40):
            self.featureDim = 768  #8192

        self.classifier = nn.Linear(self.featureDim, self.classes)

        transf_layer = nn.TransformerEncoderLayer(d_model=self.featureDim, nhead=4,dim_feedforward=512, dropout=0.4)

        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=2)


        self.score_avg = nn.AdaptiveAvgPool1d(1)



    def forward(self, x):  # 16,5,3,224,40  b,l,c,frame,feature

        self.length = x.shape[1]
        self.real_batch = x.shape[0]
        self.batch_size = x.shape[0] * self.length

        x=x.view(-1,self.channel,self.height,self.width)
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        # x = residual + shortcut
        x = self.drop(residual + shortcut)
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        # x = residual + shortcut
        x = self.drop(residual + shortcut)



        c_attn = self.s_attention(x)
        c_attn_context = c_attn * x
        spat_att = self.c_attention(c_attn_context)
        cs_attn_context = spat_att * c_attn_context  # 80,256,14,10


        out = cs_attn_context.transpose(1, 2).reshape(self.batch_size, -1, self.featureDim).transpose(0,1) #14,80,2560

        out = self.transf_encoder(out)
        attn_out = torch.mean(out, dim=0)#.view(self.real_batch,self.length,-1)

        #  B,L,F
        cla = self.classifier(attn_out).view(-1,self.length,self.classes)
        cla_out = torch.transpose(cla,1,2)
        cla_out = self.score_avg(cla_out).squeeze(2)  #b,classes
        cla_out = F.softmax(cla_out,dim=1)

        return cla_out,attn_out.view(self.real_batch,self.length,-1),cla



if __name__ == "__main__":

    a = torch.rand(2,5, 3, 224, 128)
    model = inno_model1(input_channel=3,classes=4,height=224,width=128)
    cla,fea,cla_all = model(a)
    print()

    print()