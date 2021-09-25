import sys

sys.path.append("/Users/admin/Documents/projects/arnold_workspace/src")
sys.path.append("/opt/tiger/lhh_arnold_base/arnold_workspace/src")
import math

import torch.nn as nn
from tools.pytorch.modules.fDomainHelper import FDomainHelper
from tools.pytorch.metrics.sisnr import *
from torchlibrosa.stft import STFT
import torch.nn.functional as F
# from kornia.filters import *
from tools.pytorch.metrics.lsd import LSD

EPS = 1e-8

mel_weight_44k_128 = torch.tensor([ 19.40951426,  19.94047336,  20.4859038 ,  21.04629067,
        21.62194148,  22.21335214,  22.8210215 ,  23.44529231,
        24.08660962,  24.74541882,  25.42234287,  26.11770576,
        26.83212784,  27.56615283,  28.32007747,  29.0947679 ,
        29.89060111,  30.70832636,  31.54828121,  32.41121487,
        33.29780773,  34.20865341,  35.14437675,  36.1056621 ,
        37.09332763,  38.10795802,  39.15039691,  40.22119881,
        41.32154931,  42.45172373,  43.61293329,  44.80609379,
        46.031602  ,  47.29070223,  48.58427549,  49.91327905,
        51.27863232,  52.68119708,  54.1222372 ,  55.60274206,
        57.12364703,  58.68617876,  60.29148652,  61.94081306,
        63.63501986,  65.37562658,  67.16408954,  69.00109084,
        70.88850318,  72.82736101,  74.81985537,  76.86654792,
        78.96885475,  81.12900906,  83.34840929,  85.62810662,
        87.97005418,  90.37689804,  92.84887686,  95.38872881,
        97.99777002, 100.67862715, 103.43232942, 106.26140638,
       109.16827015, 112.15470471, 115.22184756, 118.37439245,
       121.6122689 , 124.93877158, 128.35661454, 131.86761321,
       135.47417938, 139.18059494, 142.98713744, 146.89771854,
       150.91684347, 155.0446638 , 159.28614648, 163.64270198,
       168.12035831, 172.71749158, 177.44220154, 182.29556933,
       187.28286676, 192.40502126, 197.6682721 , 203.07516896,
       208.63088733, 214.33770931, 220.19910108, 226.22363072,
       232.41087124, 238.76803591, 245.30079083, 252.01064464,
       258.90261676, 265.98474   , 273.26010248, 280.73496362,
       288.41440094, 296.30489752, 304.41180337, 312.7377183 ,
       321.28877878, 330.07870237, 339.10812951, 348.38276173,
       357.91393924, 367.70513992, 377.76413924, 388.09467408,
       398.70920178, 409.61813793, 420.81980127, 432.33215467,
       444.16083117, 456.30919947, 468.78589276, 481.61325588,
       494.78824596, 508.31969844, 522.2238331 , 536.51163441,
       551.18859414, 566.26142988, 581.75006061, 597.66210737]) / 19.40951426

mel_weight_44k_128 = mel_weight_44k_128[None,None,None,...]

def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return L1()
    elif loss_type == "l1_sp":
        return L1_Sp()
    elif loss_type == "l1_log_sp":
        return L1_Log_Sp()
    elif loss_type == "l1":
        return L1()
    elif loss_type == "sisnr":
        return SiSnr()
    elif loss_type == "sispec":
        return SiSpec()
    elif loss_type == "simelspec":
        return SiMelSpec()
    elif loss_type == "sispeclog":
        return SiSpecLog()
    elif loss_type == "snr":
        return Snr()
    elif loss_type == "bce":
        return BCE()
    elif loss_type == 'l1_wav_l1_sp':
        return L1_Wav_L1_Sp()
    elif loss_type == 'l1_wav_l1_log_sp':
        return L1_Wav_L1_Log_Sp()
    elif loss_type == 'wm44k':
        return WM44k(mel_weight_44k_128)
    elif loss_type == 'lsd':
        return LSD()
    else:
        raise NotImplementedError("Error!")

class WM44k(nn.Module):
    def __init__(self, weight):
        super(WM44k, self).__init__()
        self.loss = L1()
        self.weight = weight

    def __call__(self, output, target):
        """
        :param output: [batchsize, channel, t-steps, mel-bins]
        :param target: [batchsize, channel, t-steps, mel-bins]
        :return:
        """
        self.weight = self.weight.type_as(output)
        return self.loss(output*self.weight, target*self.weight)

def si_snr(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return -torch.sum(sdr)/sdr.size()[0]

class SiMelSpec(nn.Module):
    def __init__(self):
        super(SiMelSpec, self).__init__()
        self.l1 = L1()

    def __call__(self, output, target: torch.Tensor, log_op=False):
        output, target = energy_unify(output, target)
        noise = output-target
        # print(pow_p_norm(target) , pow_p_norm(noise), pow_p_norm(target) / (pow_p_norm(noise) + EPS))
        sp_loss = 10*torch.log10((pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS))
        return -torch.sum(sp_loss)/sp_loss.size()[0]

class SiSpec(nn.Module):
    def __init__(self):
        super(SiSpec, self).__init__()
        self.loss = L1_Sp_Enhanced()

    def __call__(self, output, target):
        output,target = energy_unify(output,target)
        return self.loss(output,target)

class SiSpecLog(nn.Module):
    def __init__(self):
        super(SiSpecLog, self).__init__()
        self.loss = L1_Sp()

    def __call__(self, output, target):
        output,target = energy_unify(output,target)
        return self.loss(output,target,log_op=True)

class Snr(nn.Module):
    def __init__(self):
        super(Snr, self).__init__()
        self.loss = snr

    def __call__(self, output, target):
        return self.loss(output,target)

class SiSnr(nn.Module):
    def __init__(self):
        super(SiSnr, self).__init__()
        self.loss = si_snr

    def __call__(self, output, target):
        return self.loss(output,target)

class L1_Sp_Enhanced(nn.Module):
    def __init__(self):
        super(L1_Sp_Enhanced, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        self.l1 = L1()

    def __call__(self, output, target: torch.Tensor, log_op=False):
        output, target = self.f_helper.wav_to_spectrogram(output, eps=1e-8), self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        output, target = output-1e-4, target-1e-4
        noise = output-target
        # print(pow_p_norm(target) , pow_p_norm(noise), pow_p_norm(target) / (pow_p_norm(noise) + EPS))
        sp_loss = 10*torch.log10((pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS))

        max_amp = [torch.max(target[i,...]) for i in range(target.size()[0])]
        temp = sum(max_amp)+EPS
        max_amp = torch.tensor([each/temp for each in max_amp])
        max_amp = max_amp.type_as(target)
        for i in range(1, len(list(sp_loss.size()))):
            max_amp = max_amp.unsqueeze(-1)

        # print(max_amp.size(),sp_loss.size())
        # print(max_amp)
        sp_loss = sp_loss * max_amp
        return -torch.sum(sp_loss)/sp_loss.size()[0]

class L1_Log_Sp(nn.Module):
    def __init__(self):
        super(L1_Log_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        self.l1 = L1()

    def __call__(self, output, target, log_op=False):
        sp_loss = self.l1(
                torch.log10(self.f_helper.wav_to_spectrogram(output, eps=1e-8)),
                torch.log10(self.f_helper.wav_to_spectrogram(target, eps=1e-8))
            )
        return sp_loss

class L1_Sp(nn.Module):
    def __init__(self):
        super(L1_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        self.l1 = L1()

    def __call__(self, output, target, log_op=False):
        sp_loss = self.l1(
                self.f_helper.wav_to_spectrogram(output, eps=1e-8),
                self.f_helper.wav_to_spectrogram(target, eps=1e-8)
            )
        return sp_loss

class L1_Wav_L1_Sp(nn.Module):
    def __init__(self):
        super(L1_Wav_L1_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.l1 = L1()
        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output, target, alpha_t=0.85):
        wav_loss = self.l1(output, target)

        sp_loss = self.l1(
            self.f_helper.wav_to_spectrogram(output, eps=1e-8),
            self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        )

        sp_loss /= math.sqrt(self.window_size)

        return alpha_t*wav_loss + (1-alpha_t)*sp_loss

class L1_Wav_L1_Log_Sp(nn.Module):
    def __init__(self):
        super(L1_Wav_L1_Log_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.l1 = L1()
        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output, target, alpha_t=0.85):
        wav_loss = self.l1(output, target)

        sp_loss = self.l1(
            torch.log10(self.f_helper.wav_to_spectrogram(output, eps=1e-8)),
            torch.log10(self.f_helper.wav_to_spectrogram(target, eps=1e-8))
        )

        sp_loss /= math.sqrt(self.window_size)

        return alpha_t*wav_loss + (1-alpha_t)*sp_loss


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()

    def __call__(self, output, target):
        return self.loss(output,target)

class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.loss = F.binary_cross_entropy

    def __call__(self, output, target):
        return self.loss(output,target)

# class GRADIENT_LOSS(nn.Module):
#     def __init__(self):
#         super(GRADIENT_LOSS, self).__init__()
#         self.loss = L1()
#
#     def masking(self, data):
#         mean = torch.mean(data,dim=(1,2,3),keepdim=True)
#         mask = data > mean
#         data = data * mask
#         return data
#
#     def pre(self, output, target):
#         return self.masking(output), self.masking(target)
#
#     def __call__(self, output, target):
#         """
#         :param output: [batchsize, channels, H, W]
#         :param target: [batchsize, channels, H, W]
#         :return:
#         """
#         p_output, p_target = self.pre(output, target)
#         g_output_1 = spatial_gradient(p_output,order=1)
#         g_target_1 = spatial_gradient(p_target,order=1)
#         return self.loss(g_output_1,g_target_1)

# class GRADIENT_LOSS_L1(nn.Module):
#     def __init__(self):
#         super(GRADIENT_LOSS_L1, self).__init__()
#         self.loss = L1()
#
#     def masking(self, data):
#         mean = torch.mean(data,dim=(1,2,3),keepdim=True)
#         mask = data > mean
#         data = data * mask
#         return data
#
#     def pre(self, output, target):
#         return self.masking(output), self.masking(target)
#
#     def __call__(self, output, target):
#         """
#         :param output: [batchsize, channels, H, W]
#         :param target: [batchsize, channels, H, W]
#         :return:
#         """
#         p_output, p_target = self.pre(output, target)
#         g_output_1 = spatial_gradient(p_output,order=1)
#         g_target_1 = spatial_gradient(p_target,order=1)
#         # g_output_2 = spatial_gradient(p_output,order=2)
#         # g_target_2 = spatial_gradient(p_target,order=2)
#         return self.loss(output,target) + self.loss(g_output_1,g_target_1) # + self.loss(g_output_2,g_target_2)

if __name__ == "__main__":
    loss = SiMelSpec()
    a = torch.randn((3,1,100,200))
    b = torch.randn((3,1,100,200))
    print(loss(a,a*2))
    print(loss(a,a))
    print(loss(a,a*300))
    print(loss(a,a+10))
    print(loss(a,a+1))
    print(loss(a,b))