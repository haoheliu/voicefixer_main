import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloaders.dataloader.FixLengthAugRandomDataLoader import FixLengthAugRandomDataLoader
from dataloaders.dataloader.PairedFullLengthDataLoader import PairedFullLengthDataLoader
from torch.utils.data.distributed import DistributedSampler
from tools.pytorch.random_ import *
from torch.utils.data import ConcatDataset
from tools.dsp.lowpass import *
from tools.pytorch.random_ import *
import torch
from scipy.signal import butter, lfilter
from tools.pytorch.random_ import uniform_torch

class LowpassTrainCollator(object):
    def __init__(self, hp):
        self.hp = hp

    def __call__(self, batch):
        keys = list(batch[0].keys())  # vocals
        ret = {}
        cutoffs, orders, filters, snr = [], [], [], []

        if(self.hp["augment"]["params"]["low_pass_2"]["low_pass_range"][0] == self.hp["augment"]["params"]["low_pass_2"]["low_pass_range"][1]):
            for key in keys:
                ret[key+"_LR"] = ret[key].clone()
            return ret

        for id, _ in enumerate(batch):
            cutoffs.append(int(uniform_torch(lower=int(self.hp["augment"]["params"]["low_pass_2"]["low_pass_range"][0] // 2),
                                             upper=int(self.hp["augment"]["params"]["low_pass_2"]["low_pass_range"][1] // 2))))
            orders.append(int(uniform_torch(lower=self.hp["augment"]["params"]["low_pass_2"]["filter_order_range"][0],
                                            upper=self.hp["augment"]["params"]["low_pass_2"]["filter_order_range"][1])))
            filters.append(random_choose_list(self.hp["augment"]["params"]["low_pass_2"]["filter_type"]))

        for key in keys:
            if ("fname" in key):
                ret[key] = [x[key] for x in batch]
                continue
            ret[key] = stack_convert([x[key][..., 0:1] for x in batch])
            if ("vocals" in key):
                lowpass_data = []
                for x, c, o, f in zip(batch, cutoffs, orders, filters):
                    chance = uniform_torch(lower=0, upper=1000)
                    data = lowpass(x[key][..., 0], highcut=c, fs=self.hp["data"]["sampling_rate"], order=o, _type=f)[..., None]
                    if (int(chance) % 2 == 0):
                        lowpass_data.append(
                            lowpass(data[..., 0], highcut=c, fs=self.hp["data"]["sampling_rate"], order=o, _type="stft")[
                                ..., None])
                    else:
                        lowpass_data.append(data)
                ret[key + "_LR"] = stack_convert(lowpass_data)

            if ("noise" in key):
                lowpass_data = []
                for x, c, o, f in zip(batch, cutoffs, orders, filters):
                    chance = uniform_torch(lower=0, upper=1000)
                    if (int(chance) % 2 == 0):
                        data = x[key]
                        lowpass_data.append(data)
                    else:
                        data = lowpass(x[key][..., 0], highcut=c, fs=self.hp["data"]["sampling_rate"], order=o, _type=f)[
                            ..., None]
                        if (int(chance) % 3 == 0):
                            lowpass_data.append(
                                lowpass(data[..., 0], highcut=c, fs=self.hp["data"]["sampling_rate"], order=o, _type="stft")[
                                    ..., None])
                        else:
                            lowpass_data.append(data)
                ret[key + "_LR"] = stack_convert(lowpass_data)
        return ret


class LowpassValCollator(object):
    def __init__(self, hp):
        self.hp = hp
    def __call__(self, batch):
        pass

class SrRandSampleRate(pl.LightningDataModule):
    def __init__(self, hp, distributed):

        super(SrRandSampleRate, self).__init__()
        self.hp = hp
        self.collate_fn = LowpassTrainCollator(hp)
        self.distributed = distributed
        self.train_loader = "FixLengthAugRandomDataLoader"
        datasets = []
        for k in hp["data"]["val_dataset"].keys():
            datasets+=list(hp["data"]["val_dataset"][k].keys())
        self.val_datasets = list(set(datasets))
    def setup(self, stage = None):
        if(stage == 'fit' or stage is None):
            self.train = eval(self.train_loader)(frame_length=self.hp["train"]["input_segment_length"],
                                                   sample_rate=self.hp["data"]["sampling_rate"],
                                                    type_of_sources=self.hp["data"]["source_types"],
                                                   data=self.hp["data"]["train_dataset"],
                                                   aug_conf=self.hp["augment"]["params"],
                                                   aug_sources=self.hp["augment"]["source"],
                                                   aug_effects=self.hp["augment"]["effects"],
                                                   hours_for_an_epoch = self.hp["train"]["hours_of_data_for_an_epoch"])

            val_datasets = []
            if(len(list(self.hp["data"]["val_dataset"].keys())) != 0):
                for val_name in self.val_datasets:
                    val_datasets.append(PairedFullLengthDataLoader(dataset_name=val_name,
                                                                   sample_rate=self.hp["data"]["sampling_rate"],
                                                                   data=self.hp["data"]["val_dataset"]))
            self.val = ConcatDataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        if(self.distributed):
            sampler = DistributedSampler(self.train)
            return DataLoader(self.train, sampler = sampler, batch_size=self.hp["train"]["batch_size"], num_workers=self.hp["train"]["num_works"], pin_memory=True, collate_fn=self.collate_fn)
        else:
            return DataLoader(self.train, batch_size=self.hp["train"]["batch_size"], shuffle=True, num_workers=self.hp["train"]["num_works"],collate_fn=self.collate_fn)

    def val_dataloader(self):
        if(self.distributed):
            sampler = DistributedSampler(self.val,shuffle=False)
            return DataLoader(self.val, sampler = sampler, batch_size=1, pin_memory=True, collate_fn=collate_fn_val)
        else:
            return DataLoader(self.val, batch_size=1, shuffle=False, collate_fn=collate_fn_val)

def stack_convert(li: list):
    for i in range(len(li)):
        if(type(li[i]) != torch.Tensor):
            li[i] = torch.tensor(li[i].copy(),dtype=torch.float32)
    return torch.stack(li)

def collate_fn_val(list_data):
    keys = list(list_data[0].keys()) # vocals
    ret = {}
    cutoffs, orders, filters = [],[],[]
    cutoffs=[1000, 2000,4000,8000,12000]
    for key in keys:
        if ("fname" in key):
            ret[key] = [x[key] for x in list_data]
            continue
        ret[key] = stack_convert([x[key][..., 0:1] for x in list_data])
        for c in cutoffs:
            # lowpass processes, Only the first channel is used
            lowpass_data=[]
            for x in list_data:
                data = lowpass(x[key][...,0], highcut=c, fs=44100, order=8, _type="butter")[..., None]
                lowpass_data.append(lowpass(data[...,0], highcut=c, fs=44100, order=8, _type="stft")[..., None])
            ret[key+"LR"+"_"+str(c)] = stack_convert(lowpass_data)
            ret['cutoffs'+"_"+str(c)] = cutoffs
            ret['orders'+"_"+str(c)] = orders
            ret['filters'+"_"+str(c)] = filters
    return ret

def activelev(*args):
    '''
        need to update like matlab
    '''
    return np.max(np.abs([*args]))

def add_random_noise(clean, noise, snr):
    clean_weight = 10 ** (float(snr) / 20)
    return clean,noise/clean_weight

def unify_energy_and_random_scale(clean, noise, noisy):
    # print(target.shape,noise.shape,noisy.shape)
    max_amp = activelev(clean,noise,noisy)
    mix_scale = 1.0/max_amp
    return clean * mix_scale, noise * mix_scale, noisy * mix_scale
