# -*- coding: UTF-8 -*-
"""
@file:2021_12_26_AWGN.py
@author: Wei Jie
@date: 2021/12/26
@description:  添加高斯噪声，达到数据增强效果
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import numpy as np
import python_speech_features as ps
import librosa
import soundfile as sf
# Augment signals by adding AWGN
def addAWGN(signal, num_bits=16, augmented_num=7, snr_low=15, snr_high=30):
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    # Generate noisy signal
    return signal + K.T * noise

if __name__ == "__main__":

    file = 'E:\\Dataset\\IEMOCAP_full_release\\allwav\\5emotion\\Ses01F_impro01_F000.wav'
    # 音频数据加载
    data, sr = librosa.load(file, sr=16000)
    # 音频数据预加重 Pre-emphasis  送入一个高通滤波器
    data = ps.sigproc.preemphasis(data, coeff=0.97)
    x = addAWGN(data)   # 增强的副本数，采样数
    sf.write( '1.wav', x[0], 16000)


    print()