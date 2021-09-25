import torch
import torch.nn as nn
from tools.pytorch.modules.fDomainHelper import FDomainHelper
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display

EPS=1e-8

class LSD(nn.Module):
    def __init__(self):
        super(LSD, self).__init__()
        self.f_helper = FDomainHelper()

    def __call__(self, output, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels]
        """
        output = self.f_helper.wav_to_spectrogram(output, eps=1e-8)
        target = self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        # temp[...,:-2,:] = output[...,2:,:]
        lsd = torch.log10((target**2/(output**2 + EPS)) + EPS)**2
        # plt.figure(figsize=(15,5))
        # librosa.display.specshow(lsd[0,0,...].permute(1,0).numpy())
        # plt.show()
        lsd = torch.mean(torch.mean(lsd,dim=3)**0.5,dim=2)
        return lsd[...,None,None]


class IMG_LSD(nn.Module):
    def __init__(self):
        super(IMG_LSD, self).__init__()

    def __call__(self, output, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels]
        """
        output = torch.clip(output,min=1e-5)
        target = torch.clip(target,min=1e-5)

        lsd = torch.log10(output**2/target**2)**2
        lsd = torch.mean(torch.mean(lsd,dim=2)**0.5,dim=2)

        return lsd

class IMG_LSD_MASK(nn.Module):
    def __init__(self):
        super(IMG_LSD_MASK, self).__init__()

    def __call__(self, output, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels]
        """
        output = torch.clip(output,min=1e-5)
        target = torch.clip(target,min=1e-5)

        mask = target/torch.sum(target)
        mask = mask/torch.max(mask)

        lsd = torch.log10(output**2/target**2)**2 * mask
        lsd = torch.mean(torch.mean(lsd,dim=2)**0.5,dim=2)

        return lsd

if __name__ == "__main__":
    est = torch.randn((3,1,44100))
    tar = torch.randn((3,1,44100))

    l = LSD()
    print(l(est,tar))