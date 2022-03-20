# -*- coding: UTF-8 -*-
"""
@file:2021_12_28_AugDataFbankFea.py
@author: Wei Jie
@date: 2021/12/28
@description:  提取Fbank特征
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import gc
from typing import List
import wave
import librosa
import pickle
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
import random
import python_speech_features as ps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import *

# from feature.kappa import avg_kappa

"""
保存文件名和label对应关系
#eg:   ‘Ses01F_impro01_F000’   'neu'
name_emotion_dict = pickle.load(open('wavname_emotion_dict.pkl', 'rb'))
dest_emotion = ['ang', 'hap', 'neu', 'sad','exc']
class_id = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3,'exc':1}
name_emotionLable={}
allname=name_emotion_dict.keys()
for i in allname:
    emotion = name_emotion_dict[i]
    if emotion in dest_emotion:
        label=class_id[emotion]
        name_emotionLable[i]=label
file=open('name_emotionLabel_dict.pickle','wb')
pickle.dump(name_emotionLable,file)
file.close()
"""

name_emotion_dict = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))

# chunk_wavs = 'E:/Dataset/IEMOCAP_full_release/allwav/5emotion/'
chunk_wavs = 'E:/Dataset/IEMOCAP_full_release/allwav/AugWav/'


# 2. 提取logfbank特征
def read_wav_file(wav_filename):
    """Read the audio files in wav format and store the wave data"""
    filter_num = 40 #128
    wav_file = wave.open(wav_filename, 'r')
    params = wav_file.getparams()  # 声道数；采样精度（量化位数byte）；采样率；帧数(采样点数)
    _, _, framerate, wav_length = params[:4]
    str_data = wav_file.readframes(wav_length)  # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位），readframes返回的是二进制数据
    wave_data = np.frombuffer(str_data, dtype=np.short)  # length=帧数（采样点数）;array
    wav_file.close()
    #wave_data = ps.sigproc.preemphasis(wave_data, coeff=0.97)

    """
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    """
    # mel_spec = ps.fbank(wave_data, framerate, nfilt=filter_num)[0]
    # mel_spec = ps.fbank(wave_data, framerate, nfilt=filter_num, nfft=1024)[0]   #  比logfbank数值大

    mel_spec = ps.logfbank(wave_data, framerate, nfilt=filter_num)  # 帧数（窗口数），滤波器数量    194,40
    # delta1 = ps.delta(mel_spec, 2)
    # delta2 = ps.delta(delta1, 2)
    # spec_data = np.array((mel_spec, delta1, delta2)).transpose((1, 0, 2))
    # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # delta1_db = librosa.power_to_db(delta1, ref=np.max)
    # delta2_db = librosa.power_to_db(delta2, ref=np.max)
    # spec_data = np.array((mel_spec_db, delta1_db, delta2_db)).transpose((1, 0, 2))

    # return spec_data
    return np.expand_dims(mel_spec,1)


# read_wav_file('E:\\Dataset\\IEMOCAP_full_release\\allwav\\5emotion\\Ses01F_impro01_F000.wav')

# 保存文件的所有logfbank特征
def read_file_logmel(path_names):  # path_names:  所有文件地址列表

    name_feature = {}
    for path in path_names:
        logmelsepc = read_wav_file(chunk_wavs + path + '_7.wav')  # 特征  【帧数，3，维度40】
        name_feature[path] = logmelsepc

    file = open('DATA/2022_01_26_name_logfbank_dict_40d_7.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()





if __name__ == '__main__':
    names = list(name_emotion_dict.keys())
    read_file_logmel(names)

    # x = pickle.load(open('Data/name_melspectrogram_128_signal0_dict.pickle', 'rb'))
    # key = x.keys()
    # value = list(x.values())
    # for i in range(len(key)):
    #     # print(i)
    #     if i == 0:
    #         feature = np.expand_dims(value[i][:251], 0)
    #     else:
    #         feature = np.vstack((feature, np.expand_dims(value[i][:251], 0)))
    #
    # file = open('Data/feature.pickle', 'wb')
    # pickle.dump(feature, file)
    # file.close()

    """
    for i in range(1):

        # names = list(name_emotion_dict.keys())
        # saveMELspec(names)


        x = pickle.load(open('Data/name_melspectrogram_128_signal0_dict.pickle', 'rb'))
        key = x.keys()
        value = list(x.values())
        for i in range(len(key)):
            #print(i)
            if i == 0:
                feature = np.expand_dims(value[i][:251],0)
            else:
                feature = np.vstack((feature,np.expand_dims(value[i][:251],0)))

        file = open('Data/feature.pickle', 'wb')
        pickle.dump(feature, file)
        file.close()


        random.seed(0)
        # length=100
        # testNum=20
        test_list = random.sample(range(0, 5531), int(1106))
        #         # print(test_list)
        # print(sum(test_list))
        train_list = list(set(np.arange(5531)) - set(test_list))
        np.random.shuffle(train_list)



        # 1106,1,128,188

        # y = pickle.load('Data/name_melspectrogram_128_signal0_dict.pickle')
        # test_id = str(i + 1)
        # all_files = split_file()
        # mfcc_from_file(all_files[0])
        # get_melspectrogram(all_files[0])
        # read_wav_file(all_files[0])

        qqq = 8
        for j in range(qqq,qqq+1):
            print(j)

            # read_file_mfcc(all_files)  保存mfcc特征
            # name_specdata_dict = pickle.load(open('Data/name_logmelfeature_dict.pickle','rb'))
            #
            # features=list(name_specdata_dict.values())
            names = list(name_emotion_dict.keys())
            saveMELspec(names)
            # read_file_logmel(names)



            # split_input_target(name_specdata_dict,224,10)

            """
    """
            统计所有audio帧数最大值、最小值、均值
            allframeLength=0
            min=10000
            max=0
            for i in range(len(features)):
                allframeLength+=features[i].shape[0]
                if(features[i].shape[0]>max):
                    max=features[i].shape[0]     # 3413
                if(features[i].shape[0]<min):
                    min=features[i].shape[0]     # 57
            print(allframeLength/len(features))  # 453
            """

