# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 2:57 下午
# @Author  : Haohe Liu
# @contact: liu.8948@osu.edu
# @FileName: audaug.py
import sys
sys.path.append("../../tools")

from tools.pytorch.random_ import *
from dataloaders.augmentation.magical_effects import MagicalEffects
from tools.others.audio_op import *

class AudioAug:
    def __init__(self,
                 config = None,
                 sample_rate = 44100,
                 rir_dir="",
                 ):
        self.sample_rate = sample_rate
        self.me = MagicalEffects(p_effects=config, rir_dir=rir_dir)
        self.rir_dir = rir_dir

    def perform(self,frames, effects, rir = None,return_effects=False):
        """
        :param frames: required, np.array, [samples, channels], should be normalized to range [-1, 1]
        :param effects: required, str or list(str), effects to be performed.
        :param rir: optional, np.array, rir filter
        :return: same shape, but augmented
        """
        if(isinstance(effects,str)):effects = [effects]
        return self.me.effect(frames, effects=effects, sample_rate=self.sample_rate, rir=rir, return_effects=return_effects)

def add_noise_and_scale(front, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    """
    :param front: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr_l: Optional
    :param snr_h: Optional
    :param scale_lower: Optional
    :param scale_upper: Optional
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    snr = None
    noise, front = normalize_energy_torch(noise), normalize_energy_torch(front)  # set noise and vocal to equal range [-1,1]
    # print("normalize:",torch.max(noise),torch.max(front))
    if(snr_l is not None and snr_h is not None):
        front, noise, snr = _random_noise(front, noise, snr_l=snr_l, snr_h=snr_h)  # remix them with a specific snr
    noisy, noise, front = unify_energy_torch(noise + front, noise, front)  # normalize noisy, noise and vocal energy into [-1,1]
    # print("unify:", torch.max(noise), torch.max(front), torch.max(noisy))
    scale = _random_scale(scale_lower, scale_upper)  # random scale these three signal
    # print("Scale",scale)
    noisy, noise, front = noisy * scale, noise * scale, front * scale  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    return front, noise, snr, scale

def add_noise_and_scale_with_HQ_with_Aug(HQ, front, augfront, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    """
    :param front: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr_l: Optional
    :param snr_h: Optional
    :param scale_lower: Optional
    :param scale_upper: Optional
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    snr = None
    noise = normalize_energy_torch(noise)  # set noise and vocal to equal range [-1,1]
    HQ, front, augfront = unify_energy_torch(HQ, front, augfront)
    # some clipping noise is extremly noisy
    front_level = torch.mean(torch.abs(augfront))
    if(front_level > 0.02):
        noise_front_energy_ratio = torch.mean(torch.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    if(snr_l is not None and snr_h is not None):
        augfront, noise, snr = _random_noise(augfront, noise, snr_l=snr_l, snr_h=snr_h)  # remix them with a specific snr
    _, augfront, noise, front, HQ = unify_energy_torch(noise + augfront, augfront,  noise, front, HQ)  # normalize noisy, noise and vocal energy into [-1,1]
    # print("unify:", torch.max(noise), torch.max(front), torch.max(noisy))
    scale = _random_scale(scale_lower, scale_upper)  # random scale these three signal
    # print("Scale",scale)
    noise, front, augfront, HQ = noise * scale, front * scale, augfront*scale, HQ*scale  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    return HQ, front, augfront, noise, snr, scale

def add_noise_and_scale_with_HQ(HQ, front, noise, snr_l=-5, snr_h=35, scale_lower=0.6, scale_upper=1.0):
    """
    :param front: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr_l: Optional
    :param snr_h: Optional
    :param scale_lower: Optional
    :param scale_upper: Optional
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    snr = None
    noise = normalize_energy_torch(noise)  # set noise and vocal to equal range [-1,1]
    HQ, front = unify_energy_torch(HQ, front)
    # some clipping noise is extremly noisy
    front_level = torch.mean(torch.abs(front))
    if(front_level > 0.02):
        noise_front_energy_ratio = torch.mean(torch.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    if(snr_l is not None and snr_h is not None):
        front, noise, snr = _random_noise(front, noise, snr_l=snr_l, snr_h=snr_h)  # remix them with a specific snr
    _, noise, front, HQ = unify_energy_torch(noise + front, noise, front, HQ)  # normalize noisy, noise and vocal energy into [-1,1]
    # print("unify:", torch.max(noise), torch.max(front), torch.max(noisy))
    scale = _random_scale(scale_lower, scale_upper)  # random scale these three signal
    # print("Scale",scale)
    noise, front, HQ = noise * scale, front * scale, HQ*scale  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    return HQ, front, noise, snr, scale

def _random_scale(lower = 0.3, upper=0.9):
    return float(uniform_torch(lower,upper))

def _random_noise( clean, noise, snr_l = None, snr_h = None):
    snr = uniform_torch(snr_l,snr_h)
    clean_weight = 10 ** (float(snr) / 20)
    return clean,noise/clean_weight, snr


if __name__ == "__main__":
    import numpy as np
    from tools.file.wav import *

    vocal = "/Users/admin/Documents/projects/arnold_workspace/datasets/mss/musdb18hq/train/A Classic Education - NightOwl/vocals.wav"
    other = "/Users/admin/Documents/projects/arnold_workspace/datasets/mss/musdb18hq/train/A Classic Education - NightOwl/other.wav"

    vocal = read_wave(vocal,convert_to_mono=True)
    other = read_wave(other,convert_to_mono=True)


    aug = AudioAug()
    print(vocal.shape,other.shape)
    vocal,other, snr, scale = add_noise_and_scale(vocal,other)
    noisy = vocal+other
    # res,effects = aug.perform(frames=vocal/2**15,effects=[])
    save_wave(frames=noisy*2**15,fname="noisy.wav",channels=1)
