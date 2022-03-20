# -*- coding: UTF-8 -*-
"""
@file:2021_12_26_VTLP.py
@author: Wei Jie
@date: 2021/12/26
@description:   采用声道长度微扰（VTLP）进行数据增强，以提高泛化能力
                对每一句语音，随机生成一个扭曲因子用于对频谱的频率轴进行扭曲，来去除声道差别对识别结果的影响
                对原始数据进行 7 个副本的生成
"""

import nlpaug.augmenter.audio as naa
import librosa
import glob
from tqdm import tqdm
import sys
import os
import soundfile as sf

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

if __name__ == "__main__":

    wlist = glob.glob(r'E:/Dataset/IEMOCAP_full_release/allwav/5emotion/*.wav')
    targetDir = 'E:/Dataset/IEMOCAP_full_release/allwav/AugWav/'
    aug = naa.VtlpAug(16000, zone=(0.0, 1.0), coverage=1, fhi=4800, factor=(0.8, 1.2))
    # aug = naa.VtlpAug(16000, zone=(0.0, 1.0), coverage=1, duration=None, fhi=4800, factor=(0.8, 1.2))
    for w in tqdm(wlist):
        for i in range(7):
            wav, _ = librosa.load(w, 16000)
            wavAug = aug.augment(wav)
            wavName = os.path.basename(w)
            # librosa.output.write_wav(targetDir+wavName+'.'+str(i+1),wavAug,16000)
            sf.write(targetDir + wavName[:-4] + '_' + str(i + 1) + '.wav', wavAug, 16000)




