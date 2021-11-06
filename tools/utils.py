import os
import glob
import sys
import argparse
import logging
import json
from tools.pytorch.pytorch_util import tensor2numpy
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import matplotlib.cm as cm

EPS=1e-9

def draw_and_save(mel: torch.Tensor, clip_max=None, clip_min=None, needlog=True):
    plt.figure(figsize=(15, 5))
    mel = np.transpose(tensor2numpy(mel)[0, 0, ...], (1, 0))
    # assert np.sum(mel < 0) == 0, str(np.sum(mel < 0)) + str(np.sum(mel < 0))

    if (needlog):
        mel_log = np.log10(mel + EPS)
    else:
        mel_log = mel

    # plt.imshow(mel)
    librosa.display.specshow(mel_log, sr=44100, x_axis='frames', y_axis='mel', cmap=cm.jet, vmax=clip_max,
                             vmin=clip_min)
    plt.colorbar()
    plt.show()

def load_wav_energy(path, sample_rate, threshold=0.95):
    wav_10k, _ = librosa.load(path, sr=sample_rate)
    stft = np.log10(np.abs(librosa.stft(wav_10k))+1.0)
    fbins = stft.shape[0]
    e_stft = np.sum(stft, axis=1)
    for i in range(e_stft.shape[0]):
        e_stft[-i-1] = np.sum(e_stft[:-i-1])
    total = e_stft[-1]
    for i in range(e_stft.shape[0]):
        if(e_stft[i] < total*threshold):continue
        else: break
    return wav_10k, int((sample_rate//2) * (i/fbins))

def load_wav(path, sample_rate, threshold=0.95):
    wav_10k, _ = librosa.load(path, sr=sample_rate)
    return wav_10k

def amp_to_original_f(mel_sp_est, mel_sp_target, cutoff=0.2):
    freq_dim = mel_sp_target.size()[-1]
    mel_sp_est_low, mel_sp_target_low = mel_sp_est[..., 5:int(freq_dim * cutoff)], mel_sp_target[..., 5:int(freq_dim * cutoff)]
    energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(mel_sp_target_low, dim=(2, 3))
    amp_ratio = energy_target / energy_est
    return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if (est.shape[-1] == ref.shape[-1]):
        return est, ref
    elif (est.shape[-1] > ref.shape[-1]):
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2):-int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2):-int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    # parser.add_argument('-m', '--model', type=str, required=True,
    #                     help='Model name')
    args = parser.parse_args()

    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    model_dir = os.path.join("./logs", exp_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        # with open(config_save_path, "w") as f:
        #     f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams["config_path"]=config_path
    return hparams, parser


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()