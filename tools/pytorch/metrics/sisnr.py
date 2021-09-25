# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 5:02 下午
# @Author  : Haohe Liu
# @FileName: sisnr.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

EPS = 1e-8

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal

def pow_p_norm(signal):
    """Compute 2 Norm"""
    shape = list(signal.size())
    dimension = []
    for i in range(len(shape)):
        if(i == 0):continue
        dimension.append(i)
    return torch.pow(torch.norm(signal, p=2, dim=dimension, keepdim=True), 2)

def pow_norm(s1, s2):
    shape = list(s1.size())
    dimension = []
    for i in range(len(shape)):
        if(i == 0 or i == 1):continue
        dimension.append(i)
    return torch.sum(s1 * s2, dim=dimension, keepdim=True)

def unify(source,target):
    source_max = torch.max(torch.abs(source))
    target_max = torch.max(torch.abs(target))
    source = source.astype(torch.float32)/source_max
    return (source*target_max).astype(torch.int16),target

def snr(estimated, original):
    # estimated, original = unify(estimated, original)
    estimated, original = estimated.float(), original.float()
    noise = estimated - original
    sdr = 10 * torch.log10(EPS + pow_p_norm(original) / (pow_p_norm(noise) + EPS))
    return -torch.sum(sdr)/sdr.size()[0]

def si_snr(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    noise = estimated - target
    sdr = 10 * torch.log10(pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
    return -torch.sum(sdr)/sdr.size()[0]

def energy_unify(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    return estimated, target

def squeeze(signal):
    # signal shape [batch_size, 1, length]
    return torch.squeeze(signal, dim=1)

if __name__ == "__main__":
    import torch
    a = torch.Tensor(torch.randn(3,1,3000))
    b = torch.Tensor(torch.randn(3,1 ,3000)) * 0.001
    print(si_snr(a,a+b))
    print(si_snr(a,a))