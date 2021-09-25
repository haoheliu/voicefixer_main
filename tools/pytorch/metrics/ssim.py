import torch
import torch.nn as nn
from tools.pytorch.modules.fDomainHelper import FDomainHelper
import numpy as np
from skimage.metrics import structural_similarity as ssim

class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.f_helper = FDomainHelper()

    def __call__(self, output, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels]
        """
        output = self.f_helper.wav_to_spectrogram(output, eps=1e-8)
        target = self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        if("cuda" in str(target.device)):
            target, output = target.detach().cpu().numpy(), output.detach().cpu().numpy()
        else:
            target, output = target.numpy(), output.numpy()
        res = np.zeros([output.shape[0],output.shape[1]])
        for bs in range(output.shape[0]):
            for c in range(output.shape[1]):
                res[bs,c] = ssim(output[bs,c,...],target[bs,c,...],win_size=7)
        return torch.tensor(res)[...,None,None]

if __name__ == "__main__":
    p = SSIM()
    est = torch.randn((3,1,44100))
    target = torch.randn((3, 1, 44100))
    print(p(est,target).size())
