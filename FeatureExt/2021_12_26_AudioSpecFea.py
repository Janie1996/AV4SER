# -*- coding: UTF-8 -*-
"""
@file:2021_12_26_AudioSpecFea.py
@author: Wei Jie
@date: 2021/12/26
@description:   提取频谱特征，涉及logfbank、fbank、melspectrogram、mfcc  4类特征
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

chunk_wavs = 'E:/Dataset/IEMOCAP_full_release/allwav/'

"""
各类情感数据量
allemotion=name_emotion_dict.values()
dest_emotion = [0,1,2,3]
num_emotion = {dest_emotion[t]: 0 for t in range(len(dest_emotion))}
for i in allemotion:
    num_emotion[i]+=1
"""

import shutil


# 从所有wave文件中取情感稳定的5531条数据
def split_file():
    allname = name_emotion_dict.keys()
    train_files = []
    for root, dirs, files in os.walk(chunk_wavs):
        for file in files:
            if not file.startswith('.') and file.endswith('.wav'):
                wav_name = file[:-4]
                if wav_name in allname:
                    src = root + file
                    dst = root + '5emotion/' + file
                    # shutil.move(src, dst)
                    train_files.append(src)
    return train_files


"""
 #画图
信号经傅里叶变换，产生频谱，频谱分解为振幅谱和相位谱
（瞬时功率是瞬时振幅的平方 --> 功率谱图是振幅谱图的平方）
librosa.amplitude_to_db : 将振幅谱图转换为dB比例谱图
librosa.power_to_db : 将功率谱图（振幅平方）转换为分贝（dB）单位

"""


def show_spec(spectrogram, sr):  # spectrogram ,  shape (f,t)
    plt.figure(figsize=(10, 4))
    #  amplitude_to_db 数据更稀疏
    # S_dB = librosa.amplitude_to_db(spectrogram, ref=np.max)
    # librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    #  power_to_db  数据更密集
    # S_dB = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


# 1. 提取梅尔频谱特征
def get_melspectrogram(file):
    # 音频数据加载
    data, sr = librosa.load(file, sr=16000)
    # 音频数据预加重 Pre-emphasis  送入一个高通滤波器
    data = ps.sigproc.preemphasis(data, coeff=0.97)

    """
    提取梅尔频谱特征

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None    语音数据
    sr : number > 0 [scalar]               采样率     # y 和 sr 是输入， 先计算得到 S
    S : np.ndarray [shape=(d, t)]          频谱       # S也是输入，不为 None 就可以不要 y 和 sr        
    n_fft : int > 0 [scalar]               快速傅里叶变换窗口大小
    hop_length : int > 0 [scalar]          步长
    n_mels :                               特征维度  

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)] 
    """
    # 针对采样率 16000， 当前函数窗口大小25ms，步长10ms
    spectrogram = librosa.feature.melspectrogram(data, sr, n_fft=1024, win_length=400, hop_length=160, window='hamming',
                                                 n_mels=40)  # 40，帧数195

    # spectrogram = librosa.feature.melspectrogram(y=data,
    #                                                                                         sr=sr,
    #                                                                                         n_fft=1024,
    #                                                                                         win_length = 512,
    #                                                                                         window='hamming',
    #                                                                                         hop_length = 256,
    #                                                                                         n_mels=128,
    #                                                                                         fmax=sr/2
    #                                                                                        )   # 128,  帧数122
    spectrogram_delta = librosa.feature.delta(spectrogram)
    spectrogram_delta2 = librosa.feature.delta(spectrogram, order=2)

    #  将振幅谱图转换为dB比例谱图
    logmelspec = librosa.amplitude_to_db(spectrogram).T  # 195,40
    # logmelspec = librosa.power_to_db(spectrogram).T
    logmelspec_delta = librosa.amplitude_to_db(spectrogram_delta).T
    logmelspec_delta2 = librosa.amplitude_to_db(spectrogram_delta2).T

    """
    logmelspec = librosa.power_to_db(spectrogram).T
    logmelspec_delta = librosa.power_to_db(spectrogram_delta).T
    logmelspec_delta2 = librosa.power_to_db(spectrogram_delta2).T
    """

    spec_data = np.hstack((logmelspec, logmelspec_delta, logmelspec_delta2))

    print(spec_data.shape)  # 195,120

    show_spec(spectrogram, sr)
    show_spec(logmelspec.T, sr)
    show_spec(librosa.power_to_db(spectrogram), sr)
    return spec_data


# get_melspectrogram('E:\\Dataset\\IEMOCAP_full_release\\allwav\\5emotion\\Ses01F_impro01_F000.wav')


# 同样是计算梅尔频谱 Calculate mel spectrograms
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )
    mfccs_delta = librosa.feature.delta(mel_spec)
    mfccs_delta_delta = librosa.feature.delta(mfccs_delta)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    delta_db = librosa.power_to_db(mfccs_delta, ref=np.max)
    delta_delta_db = librosa.power_to_db(mfccs_delta_delta, ref=np.max)

    spec_data = np.array((mel_spec_db, delta_db, delta_delta_db)).transpose((2, 0, 1))

    return spec_data


def saveMELspec(path_names):
    SAMPLE_RATE = 16000
    name_feature = {}

    for path in path_names:
        file_path = chunk_wavs + path + '.wav'

        """
    Parameters
    ----------
    path : string,                  path to the input file.
    sr   : number > 0 [scalar]      sampling rate   'None' uses the native sampling rate
    offset : float                  start reading after this time (in seconds)
    duration : float                only load up to this much audio (in seconds)

    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]        audio time series

    sr   : number > 0 [scalar]                      sampling rate of ``y``        

        """

        # audio, sample_rate = librosa.load(file_path, duration=4, offset=0.5,
        # sr=SAMPLE_RATE)
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # 至少保存6秒数据量
        if (audio.shape[0] < int(16000 * 6)):
            signal = np.zeros(int(16000 * 6))
            signal[:len(audio)] = audio
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)  # .transpose((1, 0))
        else:
            mel_spectrogram = getMELspectrogram(audio, sample_rate=SAMPLE_RATE)  # .transpose((1,0))

        name_feature[path] = mel_spectrogram

    file = open('name_melspectrogram_128_3_6_dict.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()


# 2. 提取logfbank特征
def read_wav_file(wav_filename):
    """Read the audio files in wav format and store the wave data"""
    filter_num = 128
    wav_file = wave.open(wav_filename, 'r')
    params = wav_file.getparams()  # 声道数；采样精度（量化位数byte）；采样率；帧数(采样点数)
    _, _, framerate, wav_length = params[:4]
    str_data = wav_file.readframes(wav_length)  # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位），readframes返回的是二进制数据
    wave_data = np.frombuffer(str_data, dtype=np.short)  # length=帧数（采样点数）;array
    wav_file.close()

    """
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    """
    # mel_spec = ps.fbank(wave_data, framerate, nfilt=filter_num, nfft=1024)[0]   #  比logfbank数值大
    mel_spec = ps.logfbank(wave_data, framerate, nfilt=filter_num, nfft=1024)  # 帧数（窗口数），滤波器数量    194,40
    delta1 = ps.delta(mel_spec, 2)
    delta2 = ps.delta(delta1, 2)

    show_spec(mel_spec.T, framerate)
    show_spec(librosa.amplitude_to_db(mel_spec.T), framerate)
    show_spec(librosa.power_to_db(mel_spec.T), framerate)

    spec_data = np.array((mel_spec, delta1, delta2)).transpose((1, 0, 2))

    return spec_data


# read_wav_file('E:\\Dataset\\IEMOCAP_full_release\\allwav\\5emotion\\Ses01F_impro01_F000.wav')

# 保存文件的所有logfbank特征
def read_file_logmel(path_names):  # path_names:  所有文件地址列表

    name_feature = {}
    for path in path_names:
        logmelsepc = read_wav_file(chunk_wavs + path + '.wav')  # 特征  【帧数，3，维度40】
        name_feature[path] = logmelsepc
        # file = path.split('/')[-1]
        # if file[7] in ['i', 's']:
        #     logmelsepc = read_wav_file(chunk_wavs+path+'.wav')    # 特征  【帧数，3，维度40】
        #     name_feature[file[:-4]]=logmelsepc

    # file = open('name_logmelfeature_dict.pickle', 'wb')
    file = open('name_logmel_80_dict.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()


# 3. 获取MFCC特征     get mfcc feature from audio file
def mfcc_from_file(audio_path, n_mfcc=14, sample_interval=0.01, window_len=0.025):
    # load audio to time serial wave, a 2D array
    audio, sr = librosa.load(audio_path, sr=None)  # 返回audio data 和采样率
    # (特征，帧数)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=int(sample_interval * sr),
                                 n_fft=int(window_len * sr))
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta_delta = librosa.feature.delta(mfccs_delta)
    spec_data = np.array((mfccs, mfccs_delta, mfccs_delta_delta)).transpose((2, 0, 1))

    # show_spec(mfccs, sr)
    # show_spec(mfccs_delta, sr)
    # show_spec(mfccs_delta_delta, sr)

    return spec_data

    # mfccs = np.concatenate((mfccs, mfccs_delta), axis=0)
    # # MFCC mean and std
    # mean, std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
    # mfccs = (mfccs-mean)/std
    # padding_front = np.zeros((28, 15))
    # padding_back = np.zeros((28, 15))
    # front = np.column_stack((padding_front, mfccs))
    # mfccs = np.column_stack((front, padding_back))
    # return mfccs


# 保存文件的所有mfcc特征
def read_file_mfcc(path_names):
    name_feature = {}
    for path in path_names:
        file = path.split('/')[-1]
        if file[7] in ['i', 's']:
            # mfcc = mfcc_from_file(path)    # 特征  【帧数，3，维度】
            mfcc = mfcc_from_file(path)
            name_feature[file[:-4]] = mfcc

    # file = open('name_mfccfeature_dict.pickle', 'wb')
    file = open('name_logmel_80_dict.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()


# 将频谱特征分割为相同sequence长度；resize窗口+特征 224*224
import cv2
import time


def split_input_target(file, winlen, framenum):
    names = list(file.keys())
    mfccs = list(file.values())

    # all_audio_feature=np.zeros((len(mfccs),10,3,224,224))  # batch,len,

    for i in range(len(mfccs)):
        start_time = time.time()
        mfcc = mfccs[i]  # 当前音频的特征： frame,3,filter
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
            # sequence_idx=1  # 原先代码是否出现错误error  sequence_idx+=1
            if (sequence_idx == framenum):
                break
        print(1, time.time() - start_time)

        # start_time=time.time()
        # features = np.load('E:/Dataset/IEMOCAP_full_release/mfccfeatures/' + names[i] + '.npy')
        # features = torch.from_numpy(features)  # 10,3,224,224
        # print(2,time.time() - start_time)

        # np.save(file=os.path.join("E:/Dataset/IEMOCAP_full_release/logmelfeatures/", names[i] + ".npy"), arr=sequence_list)

        # all_audio_feature[i]=sequence_list

    # return all_audio_feature


if __name__ == '__main__':
    names = list(name_emotion_dict.keys())
    saveMELspec(names)

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

