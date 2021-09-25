import torch
import time
from pynvml import *
nvmlInit()

deviceCount = nvmlDeviceGetCount()
buffer = []

for i in range(deviceCount):
    if(i == 0): continue
    buffer.append(torch.randn((8,11025)).cuda(i))

while(1):
    for i in range(len(buffer)):
        if(i == 0): continue
        res = torch.stft(buffer[i],n_fft=1024)