from scipy.signal import butter,lfilter
import torch
from tools.pytorch.modules.fDomainHelper import FDomainHelper
from scipy import signal
import librosa
import numpy as np
from tools.file.wav import *
from tools.pytorch.losses import *

from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly

f_helper = None

# ----------------------------------------------------------------
# README:
# Define several functions for data generation.
# ----------------------------------------------------------------

def stft_hard_lowpass_v0(data, lowpass_ratio):
    length = data.shape[0]
    if(type(data) is not torch.Tensor):
        data = torch.tensor(data,dtype=torch.float32)
    else:
        data = data.float()
    sps, coss, sins = f_helper.wav_to_spectrogram_phase(data[None,None,...])
    cut_frequency_dim = int(sps.size()[-1]*lowpass_ratio)
    sps[...,cut_frequency_dim:] = torch.zeros_like(sps[...,cut_frequency_dim:])
    data = f_helper.spectrogram_phase_to_wav(sps, coss, sins,length)
    data = data[0, 0, :].numpy()
    return data

def align_length(x, y):
    """ align the length of y to that of x

    Args:
        x (np.array): reference signal
        y (np.array): the signal needs to be length aligned

    Return:
        yy (np.array): signal with the same length as x
    """
    Lx = len(x)
    Ly = len(y)

    if Lx == Ly:
        return y
    elif Lx > Ly:
        # pad y with zeros
        return np.pad(y, (0, Lx-Ly), mode='constant')
    else:
        # cut y
        return y[:Lx]


def bandpass_filter(x, lowcut, highcut, fs, order, ftype):
    """ process input signal x using bandpass filter

    Args:
        x (np.array): input signal
        lowcut (float): low cutoff frequency
        highcut (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    lo = lowcut / nyq
    hi = highcut / nyq

    if ftype == 'butter':
        # b, a = butter(order, [lo, hi], btype='band')
        sos = butter(order, [lo, hi], btype='band', output='sos')
    elif ftype == 'cheby1':
        sos = cheby1(order, 0.1, [lo, hi], btype='band', output='sos')
    elif ftype == 'cheby2':
        sos = cheby2(order, 60, [lo, hi], btype='band', output='sos')
    elif ftype == 'ellip':
        sos = ellip(order, 0.1, 60, [lo, hi], btype='band', output='sos')
    elif ftype == 'bessel':
        sos = bessel(order, [lo, hi], btype='band', output='sos')
    else:
        raise Exception(f'The bandpass filter {ftype} is not supported!')

    # y = lfilter(b, a, x)
    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)
    return y

def lowpass_filter(x, highcut, fs, order, ftype):
    """ process input signal x using lowpass filter

    Args:
        x (np.array): input signal
        highcut (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    global f_helper
    if(f_helper is None):
        f_helper = FDomainHelper()
    nyq = 0.5 * fs
    hi = highcut / nyq

    if ftype == 'butter':
        sos = butter(order, hi, btype='low', output='sos')
    elif ftype == 'cheby1':
        sos = cheby1(order, 0.1, hi, btype='low', output='sos')
    elif ftype == 'cheby2':
        sos = cheby2(order, 60, hi, btype='low', output='sos')
    elif ftype == 'ellip':
        sos = ellip(order, 0.1, 60, hi, btype='low', output='sos')
    elif ftype == 'bessel':
        sos = bessel(order, hi, btype='low', output='sos')
    else:
        raise Exception(f'The lowpass filter {ftype} is not supported!')

    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)

    return y

def stft_hard_lowpass(data, lowpass_ratio, fs_ori = 44100):
    fs_down = int(lowpass_ratio * fs_ori)
    # downsample to the low sampling rate
    y = resample_poly(data, fs_down, fs_ori)

    # upsample to the original sampling rate
    y = resample_poly(y, fs_ori, fs_down)

    if len(y) != len(data):
        y = align_length(data, y)
    return y


def limit(integer, high, low):
    if(integer > high): return high
    elif(integer < low): return low
    else: return int(integer)

def lowpass(data, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param highcut: cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :return: filtered data, (samples,)
    """
    global f_helper
    if(f_helper is None):
        f_helper = FDomainHelper()

    if(len(list(data.shape)) != 1):
        raise ValueError("Error (chebyshev_lowpass_filter): Data "+str(data.shape)+" should be type 1d time array, (samples,) , can not be (samples, 1)")
    if(_type in "butter"):
        order = limit(order, high=10, low=2)
        return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="butter")
    elif(_type in "cheby1"):
        order = limit(order, high=10, low=2)
        return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="cheby1")
    # elif(_type in "cheby2"):
    #     order = limit(order, high=10, low=2)
    #     return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="cheby2")
    elif(_type in "ellip"):
        order = limit(order, high=10, low=2)
        return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="ellip")
    elif(_type in "bessel"):
        order = limit(order, high=10, low=2)
        return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="bessel")
    elif(_type in "stft"):
        return stft_hard_lowpass(data, lowpass_ratio=highcut / int(fs / 2))
    elif(_type in "stft_hard"):
        return stft_hard_lowpass_v0(data, lowpass_ratio=highcut / int(fs / 2))
    else:
        raise ValueError("Error: Unexpected filter type " + _type)

def bandpass(data, lowcut, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param lowcut: low cutoff frequency
    :param highcut: high cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :param _type: type of filter
    :return: filtered data, (samples,)
    """
    if(len(list(data.shape)) != 1):
        raise ValueError("Error (chebyshev_lowpass_filter): Data "+str(data.shape)+" should be type 1d time array, (samples,) , can not be (samples, 1)")
    if(_type in "butter"):
        order = limit(order, high=10, low=2)
        return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="butter")
    elif(_type in "cheby1"):
        order = limit(order, high=10, low=2)
        return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="cheby1")
    # elif(_type in "cheby2"):
    #     return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="cheby2")
    elif(_type in "ellip"):
        order = limit(order, high=10, low=2)
        return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="ellip")
    elif(_type in "bessel"):
        order = limit(order, high=10, low=2)
        return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="bessel")
    else:
        raise ValueError("Error: Unexpected filter type " + _type)

def test1():

    wav_spec_loss = L1_Wav_L1_Sp()

    data, sr = librosa.load(librosa.ex('trumpet'), sr=44100)
    cutoff,fs,order = 7000,44100,12

    data_butter = lowpass(data, cutoff, fs, order, _type="butter")
    data_chebyshev = lowpass(data, cutoff, fs, order, _type="chebyshev")
    data_stft = lowpass(data, cutoff, fs, order, _type="stft")
    data_resample = lowpass(data, cutoff, fs, order, _type="resample")

    print(wav_spec_loss(torch.tensor(data_stft[None,None,:],dtype=torch.float32),torch.tensor(data[None,None,:])))
    print(wav_spec_loss(torch.tensor(data_butter[None,None,:], dtype=torch.float32),torch.tensor(data[None,None,:])))
    print(wav_spec_loss(torch.tensor(data_chebyshev[None,None,:], dtype=torch.float32),torch.tensor(data[None,None,:])))
    print(wav_spec_loss(torch.tensor(data_resample[None,None,:], dtype=torch.float32),torch.tensor(data[None,None,:])))

    # print(data_butter.shape,data_chebyshev.shape,data_stft.shape)
    save_wave(data_butter[:,None]*2**15,"butter.wav",sample_rate=44100)
    save_wave(data[:,None]*2**15,"original.wav",sample_rate=44100)
    save_wave(data_chebyshev[:,None]*2**15,"chebyshev.wav",sample_rate=44100)
    save_wave(data_stft[:,None]*2**15,"stft.wav",sample_rate=44100)
    save_wave(data_resample[:,None]*2**15,"resample.wav",sample_rate=44100)

if __name__ == "__main__":
    from tools.file.wav import *
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import time

    path = "/Users/admin/Documents/projects/arnold_workspace/datasets/se/vd_test/noisy_testset_wav/p257_418.wav"
    wav = read_wave(path,sample_rate=44100)[...,0]

    wav = lowpass(wav,highcut=8000,fs=44100,order=2,_type="stft_hard")

    # wav = bandpass_filter(wav,lowcut=1000,highcut=8000,fs=44100,order=8,ftype="cheby1")
    save_wave(wav[...,None],"resample_stft_hard.wav",sample_rate=44100)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(wav))))
    plt.show()