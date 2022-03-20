# -*- coding: UTF-8 -*-
"""
@file:wj_2021_12_27_dataset.py
@author: Wei Jie
@date: 2021/12/27
@description: 数据集load时就将分段好的全部feature全部加载到内存中， 每次直接检索
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import torch.utils.data as data

try:
    import cPickle as pickle
except:
    import pickle

def get_audio(index,allfeatures,name_emotion):


    labels = list(name_emotion.values())
    index_label = labels[index]
    features = allfeatures[index]
    # if(index==5530):
    #     print(features[0][0][132][34])

    return features,index_label

# audio数据集
class AudioDataset(data.Dataset):
    def __init__(self, feature ,audio_list,name_emotin):


        self.allfeatures = feature
        self.audio_list=audio_list
        self.name_emotin=name_emotin


    def __getitem__(self, index):
        #print(self.audio_list[index])
        feature,label=get_audio(self.audio_list[index],self.allfeatures,self.name_emotin)
        return feature, label

    def __len__(self):
        return len(self.audio_list)

if __name__ == "__main__":
    print()

