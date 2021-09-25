from tools.file.wav import *
from tools.pytorch.random_ import *
from tools.file.io import read_list
import torch
import random

def construct_data_folder(additional_data: dict, audio=True):
    folder = {}
    for each in additional_data.keys():
        if (each not in folder.keys()): folder[each] = []
        folder[each] += read_list(additional_data[each])
    if (audio):
        keys, weights = construct_average_durations(folder)
        # print(keys, weights)
        return folder, keys, weights
    else:
        return folder

def get_approximate_durations(dir_list: list, top=100):
    duration, cnt = 0, 0
    for i, each in enumerate(dir_list):
        if (i == top):
            break
        else:
            cnt += 1
            duration += get_duration(each)
    return duration / cnt

def construct_average_durations(folder_dict: dict, top=100):
    weight, sum = {}, 0
    for each in folder_dict.keys():
        weight[each] = get_approximate_durations(folder_dict[each], top=top) * len(folder_dict[each])
        sum += weight[each]
        # print(each, str(weight[each] / 3600) + " hours", str(len(folder_dict[each])) + " files")
    return list(weight.keys()), [weight[k] / sum for k in weight.keys()]

def constrain_length(array,length):
    if(array.shape[0] == length):
        return array
    elif(array.shape[0] > length):
        return array[:int(length),...]
    else:
        array = np.pad(array, ((0, length - array.shape[0]), (0, 0)))
        return array

def constrain_length_torch(tensor, length):
    if(tensor.size()[0] == length):
        return tensor
    elif(tensor.size()[0] > length):
        return tensor[:int(length), ...]
    else:
        return torch.nn.functional.pad(tensor, [0, 0, 0, length - tensor.size()[0]], 'constant', 0.0)

def select(prob):
    chance = random_torch(1000)
    return chance < prob * 1000,chance

def unify_energy(clean, noise, noisy):
    max_amp = activelev(clean,noise,noisy)
    mix_scale = 1.0/max_amp
    return clean * mix_scale, noise * mix_scale, noisy * mix_scale

def activelev(*args):
    return np.max(np.abs([*args]))

def get_random_key(keys:list,weights:list):
    return random.choices(keys, weights=weights)[0]

