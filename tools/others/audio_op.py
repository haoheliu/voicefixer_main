# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 3:03 下午
# @Author  : Haohe Liu
# @contact: liu.8948@osu.edu
# @FileName: audio_util.py

import torch
from scipy.interpolate import interp1d
from tools.file.wav import *
import librosa

def normalize_energy(audio, alpha = 1):
    '''
    :param audio: 1d waveform, [batchsize, *],
    :param alpha: the value of output range from: [-alpha,alpha]
    :return: 1d waveform which value range from: [-alpha,alpha]
    '''
    val_max = activelev(audio)
    return (audio / val_max) * alpha

def normalize_energy_torch(audio, alpha = 1):
    '''
    If the signal is almost empty(determined by threshold), if will only be divided by 2**15
    :param audio: 1d waveform, 2**15
    :param alpha: the value of output range from: [-alpha,alpha]
    :return: 1d waveform which value range from: [-alpha,alpha]
    '''
    val_max = activelev_torch(audio)
    return (audio / val_max) * alpha

def unify_energy(*args):
    max_amp = activelev(args)
    mix_scale = 1.0/max_amp
    return [x * mix_scale for x in args]

def unify_energy_torch(*args):
    max_amp = activelev_torch(args)
    mix_scale = 1.0/max_amp
    return [x * mix_scale for x in args]

def activelev(*args):
    '''
        need to update like matlab
    '''
    return np.max(np.abs([*args]))


def activelev_torch(*args):
    '''
        need to update like matlab
    '''
    res = []
    args = args[0]
    for each in args:
        res.append(torch.max(torch.abs(each)))
    return max(res)

def calculate_total_times(dir):
    total = 0
    for each in os.listdir(dir):
        fname = os.path.join(dir,each)
        duration = get_duration(fname)
        total += duration
    return total

def unify_length(*args):
    # length: shape[0]
    min_length = min([x.shape[0] for x in args])
    return [x[:min_length,...] for x in args]

def max_mag_unify(ref:np.ndarray,est:np.ndarray):
    '''
    Unify the energy before calculate the subjective metrics
    '''
    max_val_ref = np.max(np.abs(ref))
    max_val_est = np.max(np.abs(est))
    return ref,est*(max_val_ref/max_val_est)

def trim_tail_empty(wave,threshold = 500, sample_rate = 44100, frame_length = 0.02):
    # [samples,...]
    if(wave is None): return None
    seg_len = int(sample_rate*frame_length)
    length = wave.shape[0]
    while(length > seg_len and np.max(np.abs(wave[length-seg_len:length])) < threshold):
        length -= seg_len
    if(length < seg_len):
        return None
    if(wave.shape[0] == length):
        return wave[:length,...]
    else:
        return wave[:length-seg_len, ...]

def trim_head_empty(wave,threshold = 500,sample_rate = 44100,frame_length = 0.02):
    # [samples,...]
    if(wave is None): return None
    seg_len = int(sample_rate*frame_length)
    length = 0
    while(length < wave.shape[0]-seg_len and np.max(np.abs(wave[length:length+seg_len])) < threshold):
        length += seg_len
    if(length >= wave.shape[0]-seg_len):
        return None
    if(length == 0):
        return wave[length:,...]
    else:
        return wave[length-seg_len:, ...]

def trim_empty(wave,threshold = 500,sample_rate = 44100,frame_length = 0.02):
    wave = trim_tail_empty(wave,threshold = threshold,sample_rate = sample_rate, frame_length=frame_length)
    return trim_head_empty(wave,threshold = threshold,sample_rate = sample_rate,frame_length=frame_length)

def clean_up_wav_all_base_100(clean):
    stft = librosa.stft(clean.astype(np.float))
    stft[4, :] = stft[4, :] * 0.01
    stft[:4, :] = stft[:4, :] * 0.0001
    filtedData = librosa.istft(stft)
    return filtedData

def has_long_empty(wave,length = 3,sample_rate = 44100):
    if(wave is None): return None
    seg_len = int(sample_rate*length)
    length = wave.shape[0]
    while(length > seg_len):
        if(np.max(np.abs(wave[length-seg_len:length,...])) < 400):
            return True
        length -= sample_rate
    return False

def get_all_active_segment_index(wave, threshold=500, sample_rate=44100, frame_length = 0.02, frame_shift = 0.01):
    # [samples] np.array, Single Channel
    """
    Label each frame of wave's stft (whether active or not)
    :param wave: [samples], single channel, np.array
    :param threshold: active level
    :param sample_rate: sample rate
    :param frame_length: stft windows length
    :param frame_shift: stft windows shift
    :return: List, e.g. [0,1,0,1,1,1,1,0,0,0,0,...]
    """
    wave = librosa.util.pad_center(wave,wave.shape[0]+int(frame_shift*sample_rate)*2, mode = "reflect")
    frame_length, frame_shift = int(frame_length * sample_rate), int(frame_shift * sample_rate)
    start, isActive = 0,[]
    while(start+frame_length <= wave.shape[0]):
        window = wave[start: start+frame_length]
        if(is_valid_signal(window, threshold=threshold)): isActive.append(1)
        else: isActive.append(0)
        start += frame_shift
    return isActive

def is_valid_signal(wave, threshold = 1000):
    return np.max(np.abs(wave)) > threshold

def smooth(signal,smooth_center, smooth_samples = 50, interval = 4):
    '''
    :param signal: signal needed to be smoothed
    :param interval: sample interval for interpolaration
    :return:
    '''
    FLAG = False
    if(smooth_center < smooth_samples or signal.shape[0]-smooth_center < smooth_samples):
        return signal
    if(len(list(signal.shape)) > 1):
        signal = signal[:,0]
        FLAG = True
    samples = signal[smooth_center-smooth_samples:smooth_center+smooth_samples]
    length = samples.shape[0]
    xnew = np.array(list(range(length)))
    samples = samples[0::interval]
    x = xnew[0::interval]
    xnew = xnew[:x[-1]]

    smoothed = interp1d(x,samples,kind="cubic")
    smoothed = smoothed(xnew)
    signal[smooth_center-smooth_samples:smooth_center-smooth_samples+smoothed.shape[0]] = smoothed
    return signal[:,None] if (FLAG) else signal

if __name__ == "__main__":
    data = np.random.randn(44100*3)
    res = get_all_active_segment_index(data)
    print(len(res))