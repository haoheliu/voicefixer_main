import torch
import torch.nn as nn
from tools.pytorch.metrics.lsd import LSD,IMG_LSD_MASK,IMG_LSD
from tools.pytorch.metrics.psnr import PSNR
from tools.pytorch.metrics.ssim import SSIM
from tools.pytorch.metrics.sisnr import *
import numpy as np
import speechmetrics as sm

# in: ref-[batchsize, channel, samples], target-[batchsize, channel, samples]
# out: [batchsize, channel, metrics_value]
class ImgMetrics(nn.Module):
    def __init__(self):
        super(ImgMetrics, self).__init__()
        self._img_lsd_mask = IMG_LSD_MASK()
        self._img_lsd = IMG_LSD()

    def numpy2tensor(self, est, targets):
        toTensor = False
        if(type(est) == np.ndarray):
            est = torch.tensor(est)
            toTensor = True
        if(type(targets) == np.ndarray):
            targets = torch.tensor(targets)
            toTensor = True
        return est,targets,toTensor

    def tensor2numpy(self, est, targets):
        toArray = False
        if(type(est) == torch.Tensor):
            est = est.numpy()
            toArray = True
        if(type(targets) == torch.Tensor):
            targets = targets.numpy()
            toArray = True
        return est,targets,toArray

    def LSD_MASK(self, est, target):
        est, target, convertOrNot = self.numpy2tensor(est, target)
        res = self._img_lsd_mask(est, target)
        return res.numpy() if (convertOrNot) else res

    def LSD(self, est, target):
        est, target, convertOrNot = self.numpy2tensor(est, target)
        res = self._img_lsd(est, target)
        return res.numpy() if (convertOrNot) else res


class AudioMetrics(nn.Module):
    """
    :param output: raw wave, torch.Tensor or np.ndarray, shape: [batchsize, channels, samples]
    :param target: raw wave, shape: [batchsize, channels, samples]
    :return: np.ndarray, LSD value [batchsize, channels]
    """
    def __init__(self):
        super(AudioMetrics, self).__init__()
        self.lsd = LSD()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.sisnr = si_snr
        self.snr = snr
        self.bsseval = sm.load(['bsseval'], window=1.0)

    def numpy2tensor(self, est, targets):
        toTensor = False
        if(type(est) == np.ndarray):
            est = torch.tensor(est)
            toTensor = True
        if(type(targets) == np.ndarray):
            targets = torch.tensor(targets)
            toTensor = True
        return est,targets,toTensor

    def tensor2numpy(self, est, targets):
        toArray = False
        if(type(est) == torch.Tensor):
            est = est.numpy()
            toArray = True
        if(type(targets) == torch.Tensor):
            targets = targets.numpy()
            toArray = True
        return est,targets,toArray

    def check_shape(self, est, target):
        if(type(est) == np.ndarray):
            assert type(est) == type(target), str(type(est)) + " " + str(type(target))
            assert est.shape == target.shape, str(est.shape) + " " + str(target.shape)
            assert len(list(est.shape)) == 3, "The shape of the input need to be [batchsize, channel, samples]"
            assert est.shape[1] < 100  # channel dimension
        if (type(est) == torch.Tensor):
            assert type(est) == type(target), str(type(est)) + " " + str(type(target))
            assert est.size() == target.size(),  str(est.size()) + " " + str(target.size())
            assert len(list(est.size())) == 3, "The shape of the input need to be [batchsize, channel, samples]"
            assert est.size()[1] < 100  # channel dimension

    def PSNR(self, est, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels, 1, 1]
        """
        est,target,convertOrNot = self.numpy2tensor(est,target)
        self.check_shape(est,target)
        res = self.psnr(est,target)
        return res.numpy() if(convertOrNot) else res


    def SSIM(self, est, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels, 1, 1]
        """
        est,target,convertOrNot = self.numpy2tensor(est,target)
        self.check_shape(est,target)
        res = self.ssim(est,target)
        return res.numpy() if(convertOrNot) else res

    def LSD(self, est, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels, 1, 1]
        """
        est,target,convertOrNot = self.numpy2tensor(est,target)
        self.check_shape(est,target)
        res = self.lsd(est,target)
        return res.numpy() if(convertOrNot) else res

    def SISNR(self, est, target):
        """
        :param est: [batchsize, channels, samples]
        :param target: [batchsize, channels, samples]
        :return: [batchsize, 1]
        """
        est, target, convertOrNot = self.numpy2tensor(est, target)
        self.check_shape(est, target)
        batchsize = est.size()[0]
        res = []
        for i in range(batchsize):
            res.append(-self.sisnr(est[i,...],target[i,...]))
        res = torch.stack(res)[...,None]
        return res.numpy() if (convertOrNot) else res

    def SNR(self, est, target):
        """
        :param est: [batchsize, channels, samples]
        :param target: [batchsize, channels, samples]
        :return: [batchsize, 1]
        """
        est, target, convertOrNot = self.numpy2tensor(est, target)
        self.check_shape(est, target)
        batchsize = est.size()[0]
        res = []
        for i in range(batchsize):
            res.append(-self.snr(est[i,...],target[i,...]))
        res = torch.stack(res)[...,None]
        return res.numpy() if (convertOrNot) else res

if __name__ == "__main__":
    am = AudioMetrics()

    est = torch.randn((3, 1, 4000))
    target = torch.randn((3, 1, 4000))
    print(am.LSD(est,target))
    print(am.SISNR(est, target))
    print(am.SNR(est, target))
