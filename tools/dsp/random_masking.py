import matplotlib.pyplot as plt
import time
import numpy as np
import numpy
from tools.pytorch.modules.fDomainHelper import FDomainHelper
from tools.file.wav import *
from tools.pytorch.pytorch_util import *

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    length = x.shape[0]
    assert x.ndim == 1
    assert x.size >= window_len

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y[:length]

def get_random_sequence():
    x = np.arange(1,1026,1)
    y = np.random.uniform(0.5,1.0,(1025))
    random_empty_sequence_num = int(np.random.uniform(1.0, 256.0, (1))[0])
    for i in range(random_empty_sequence_num):
        start = int(np.random.uniform(20, 1024-20, (1))[0])
        y[start:start+20] *= 0
    z1 = np.polyfit(x, y, 12)
    p1 = np.poly1d(z1)
    yvals = p1(x)
    max_val = np.max(yvals[200:800])
    return x,np.clip(yvals * (1/max_val),a_min=0,a_max=1.0)

def get_random_mask(size):
    """
    :param data: tuple or torch.tensor.size(), (t-steps, f-bins)
    :return:
    """
    mask = np.ones(size).astype(np.float32)
    length = size[0]
    i = 0
    while(i < length):
        x,y = get_random_sequence()
        mask[i:i+10] *= y[None,...]
        i = i+10
    for i in range(mask.shape[1]):
        mask[:,i] = smooth(mask[:,i],window_len=5,window="flat")
    return torch.tensor(mask)

def add_random_mask(x: torch.Tensor):
    """
    :param x: [batchsize, channel, tstep, fbins]
    :return: x with the same size
    """
    batchsize, channel, tstep, fbins = x.size()
    assert channel == 1
    for i in range(batchsize):
        if(i % 4 == 0):
            mask = get_random_mask(x[i, 0, ...].size()).type_as(x)
            x[i,0,...] = x[i,0,...] * mask
    return x

if __name__ == '__main__':
    f_helper = FDomainHelper()
    import torch

    for i in range(10):
        wav = read_wave("/Users/admin/Desktop/p232_005.wav",sample_rate=44100).T[None,...]
        wav = torch.tensor(wav)
        spec,cos,sin = f_helper.wav_to_spectrogram_phase(wav)
        start = time.time()
        mask = get_random_mask(spec[0,0,...].size())
        end = time.time()
        spec = torch.tensor(mask[None,None,...])*spec
         #用3次多项式拟合  可以改为5 次多项式。。。。 返回三次多项式系数
        print(end-start)
        output = f_helper.spectrogram_phase_to_wav(spec,cos,sin,length=wav.size()[-1])
        save_wave(tensor2numpy(output),fname=str(i)+".wav",sample_rate=44100)
        plt.figure(figsize=(8,5))
        # plt.subplot(211)
        plt.imshow(mask.T)
        # plt.subplot(212)
        # plt.imshow(orig_mask.T)
        plt.colorbar()
        plt.show()
        # time.sleep(2)

