import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataloaders.dataloader.PairedFullLengthDataLoader import PairedFullLengthDataLoader
from dataloaders.dataloader.FixLengthAugRandomDataLoader import FixLengthAugRandomDataLoader
from torch.utils.data.distributed import DistributedSampler
from tools.pytorch.random_ import *
from torch.utils.data import ConcatDataset
from tools.dsp.lowpass import *
from tools.pytorch.random_ import *
from IPython import embed
import torch
from scipy.signal import butter, lfilter
from tools.pytorch.random_ import uniform_torch

glob_source_sample_rate_low = 1500
glob_source_sample_rate_high = 44100
glob_target_sample_rate = 44100

class SrRandSampleRate(pl.LightningDataModule):
    def __init__(self, source_sample_rate_low,source_sample_rate_high, target_sample_rate, train_data: dict, val_data:dict,
                 train_data_type:list, val_datasets: list,
                 train_loader = "FixLengthRandomDataLoader",
                 overlap_num = 1,
                 distributed = False,
                 batchsize = 12, frame_length=4.0, num_workers = 12, sample_rate = 44100,
                 aug_effects = None, aug_sources = None, aug_conf = None,
                 quiet_threshold = 0,
                 hours_for_an_epoch = 10
                 ):

        super(SrRandSampleRate, self).__init__()
        self.train_data = train_data
        self.test_data = val_data
        self.source_sample_rate_low = source_sample_rate_low
        self.source_sample_rate_high = source_sample_rate_high
        self.target_sample_rate = target_sample_rate

        assert glob_target_sample_rate == target_sample_rate, str(glob_target_sample_rate)+" "+str(target_sample_rate)
        assert glob_source_sample_rate_low == source_sample_rate_low, str(glob_source_sample_rate_low)+" "+str(source_sample_rate_low)
        assert glob_source_sample_rate_high == source_sample_rate_high, str(glob_source_sample_rate_high)+" "+str(source_sample_rate_high)

        self.overlap_num = overlap_num

        self.train_data_type = train_data_type
        self.val_datasets = val_datasets
        self.distributed=distributed
        self.batchsize = batchsize
        self.frame_length = frame_length
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.aug_effects = aug_effects
        self.aug_conf = aug_conf
        self.aug_sources = aug_sources
        self.train_loader = train_loader
        self.quiet_threshold = quiet_threshold
        self.hours_for_an_epoch = hours_for_an_epoch

    def setup(self, stage = None):
        if(stage == 'fit' or stage is None):
            self.train = eval(self.train_loader)(frame_length=self.frame_length,
                                                   sample_rate=self.sample_rate,
                                                    overlap_num = self.overlap_num,
                                                    type_of_sources=self.train_data_type,
                                                   data=self.train_data,
                                                   aug_conf=self.aug_conf,
                                                   aug_sources=self.aug_sources,
                                                   aug_effects=self.aug_effects,
                                                   hours_for_an_epoch = self.hours_for_an_epoch)
            # val_datasets = []
            # for val_name in self.val_datasets:
            #     val_datasets.append(PairedFullLengthDataLoader(dataset_name=val_name,
            #                                                    sample_rate=self.sample_rate,
            #                                                    data=self.test_data))
            # self.val = ConcatDataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        if(self.distributed):
            sampler = DistributedSampler(self.train)
            return DataLoader(self.train, sampler = sampler, batch_size=self.batchsize, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)
        else:
            return DataLoader(self.train, batch_size=self.batchsize, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn)

    # def val_dataloader(self):
    #     if(self.distributed):
    #         sampler = DistributedSampler(self.val,shuffle=False)
    #         return DataLoader(self.val, sampler = sampler, batch_size=1, num_workers=8, pin_memory=True, collate_fn=collate_fn_val)
    #     else:
    #         return DataLoader(self.val, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn_val)

    # def val_dataloader(self):
    #     if(self.distributed):
    #         sampler = DistributedSampler(self.val,shuffle=False)
    #         return DataLoader(self.val, sampler = sampler, batch_size=1, pin_memory=True, collate_fn=collate_fn_val)
    #     else:
    #         return DataLoader(self.val, batch_size=1, shuffle=False, collate_fn=collate_fn_val)

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
                data = lowpass(x[key][...,0], highcut=c, fs=glob_target_sample_rate, order=8, _type="butter")[..., None]
                lowpass_data.append(lowpass(data[...,0], highcut=c, fs=glob_target_sample_rate, order=8, _type="stft")[..., None])
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

def collate_fn(list_data):
    keys = list(list_data[0].keys()) # vocals
    ret = {}
    cutoffs, orders, filters, snr = [],[],[],[]
    for id,_ in enumerate(list_data):
        cutoffs.append(int(uniform_torch(lower=int(glob_source_sample_rate_low // 2),upper=int(glob_source_sample_rate_high // 2))))
        orders.append(int(uniform_torch(lower=2,upper=10)))
        # filters.append(random_choose_list(["stft_hard"]))
        filters.append(random_choose_list(["cheby1"]))
        # snr.append(float(uniform_torch(SNRLOW, SNR_HIGH)))

    for key in keys:
        if("fname" in key):
            ret[key] = [x[key] for x in list_data]
            continue
        ret[key] = stack_convert([x[key][...,0:1] for x in list_data])
        # lowpass processes, Only the first channel is used

        if("vocals" in key):
            lowpass_data=[]
            for x, c, o, f in zip(list_data, cutoffs, orders, filters):
                chance = uniform_torch(lower=0, upper=1000)
                data = lowpass(x[key][...,0], highcut=c, fs=glob_target_sample_rate, order=o, _type=f)[..., None]
                lowpass_data.append(lowpass(data[...,0], highcut=c, fs=glob_target_sample_rate, order=o, _type="stft")[..., None])
            ret[key+"_LR"] = stack_convert(lowpass_data)

        if("noise" in key):
            lowpass_data=[]
            for x, c, o, f in zip(list_data, cutoffs, orders, filters):
                chance = uniform_torch(lower=0, upper=1000)
                if(int(chance) % 2 == 0):
                    data = x[key]
                    lowpass_data.append(data)
                else:
                    data = lowpass(x[key][...,0], highcut=c, fs=glob_target_sample_rate, order=o, _type=f)[..., None]
                    lowpass_data.append(lowpass(data[...,0], highcut=c, fs=glob_target_sample_rate, order=o, _type="stft")[..., None])
            ret[key+"_LR"] = stack_convert(lowpass_data)
    return ret
