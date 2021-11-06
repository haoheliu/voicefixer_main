import torch
import wave
import json
import os
import glob
from progressbar import *

EPS=1e-8

import torch
import time
from pynvml import *


def convert_wav_to_flac(dir):
    current = "wav"
    files = glob.glob(os.path.join(dir, "*." + current)) + \
                glob.glob(os.path.join(dir, "*/*." + current)) + \
                glob.glob(os.path.join(dir, "*/*/*." + current)) + \
                glob.glob(os.path.join(dir, "*/*/*/*." + current)) + glob.glob(os.path.join(dir, "*/*/*/*/*." + current))
    widgets = [
        "Convert wav to flac",
        ' [', Timer(), '] ',
        Bar(),
        ' (', ETA(), ') ',
    ]
    pbar = ProgressBar(widgets=widgets).start()
    for i,path in enumerate(files):
        if (current == "wav"):
            cmd = "sox " + path + " " + path[:-4] + ".flac"
        if (current == "flac"):
            cmd = "sox " + path + " " + path[:-5] + ".wav"
        os.system(cmd)
        os.remove(path)
        pbar.update(int((i / (len(files) - 1)) * 100))
    pbar.finish()


def get_framesLength(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3]

def write_json(my_dict, fname):
    # print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)

def load_json(fname):
    with open(fname,'r') as f:
        data = json.load(f)
        return data

def get_sample_rate(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[2]

def to_log(input):
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input

def write_list(list, fname):
    with open(fname,'w') as f:
        for word in list:
            f.write(word)
            f.write('\n')

def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def pow_p_norm(signal):
    """Compute 2 Norm"""
    shape = list(signal.size())
    dimension = []
    for i in range(len(shape)):
        if(i == 0):continue
        dimension.append(i)
    return torch.pow(torch.norm(signal, p=2, dim=dimension, keepdim=True), 2)

def energy_unify(estimated, original):
    target = pow_norm(estimated, original) * original
    target /= pow_p_norm(original) + EPS
    return estimated, target

def pow_norm(s1, s2):
    shape = list(s1.size())
    dimension = []
    for i in range(len(shape)):
        if(i == 0 or i == 1):continue
        dimension.append(i)
    return torch.sum(s1 * s2, dim=dimension, keepdim=True)