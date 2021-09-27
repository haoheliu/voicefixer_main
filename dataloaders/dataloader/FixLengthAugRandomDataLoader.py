
import sys

sys.path.append("../../tools")
from torch.utils.data import Dataset
from dataloaders.dataloader.utils import *
from dataloaders.augmentation.base import AudioAug
import os
import time

class FixLengthAugRandomDataLoader(Dataset):
    '''
        {
            "type-of-source"{               # e.g. vocal, bass
                "dataset-name": "<path to .lst file (a list of path to wav files)>",
            }
            ...
        }
    '''
    def __init__(self,
                 frame_length:float,
                 sample_rate:int,
                 data: dict,
                 type_of_sources: list,
                 overlap_num=1,
                 # for augmentation use only
                 aug_conf = None,
                 aug_sources = [],
                 aug_effects = [],
                 hours_for_an_epoch = 10,
                 ):
        """
        :param frame_length: segment length in seconds
        :param sample_rate: sample rate of target dataset
        :param data: a dict object containing the path to .lst file
        :param augmentation(deprecated): Optional
        :param aug_conf: Optional, used to update random server
        :param aug_sources: the type-of-source needed for augmentation, for example, vocal
        :param aug_effects: the effects to take, for example: ['tempo','pitch'].
        """
        print(data)
        self.init_processes = []
        self.overlap_num = overlap_num
        self.sample_rate = sample_rate
        self.hours_for_an_epoch = hours_for_an_epoch
        self.data_all = {}
        if(type_of_sources is None):
            for k in data.keys():
                self.data_all[k] = construct_data_folder(data[k])
        else:
            for k in type_of_sources:
                self.data_all[k] = construct_data_folder(data[k])
        self.frame_length = frame_length
        self.aug = AudioAug(config=aug_conf, sample_rate=self.sample_rate, rir_dir=aug_conf['rir_root'])
        self.aug_sources = aug_sources
        self.aug_effects = aug_effects

    def random_fname(self,type, dataset_name = None):
        if(dataset_name is None):
            dataset_name = get_random_key(self.data_all[type][1],self.data_all[type][2])
        return self.data_all[type][0][dataset_name][random_torch(high=len(self.data_all[type][0][dataset_name]), to_int=True)],dataset_name

    def random_trunk(self, frame_length, type=None):
        # [samples, channel]
        trunk, length, sr = None , 0, self.sample_rate
        while (length-frame_length < -0.1):
            fname, dataset_name = self.random_fname(type=type)
            trunk_length = frame_length - length
            try:
                segment, duration, sr = random_chunk_wav_file(fname, trunk_length)
            except Exception as e:
                print("Error:0",fname)
                raise e
            segment = segment[:,0:1] # todo force convert to mono !!!!!!!!!!!!!!!!!!!!!!
            # assert sr == self.sample_rate, fname
            length += duration
            if (trunk is None): trunk = segment
            else: trunk = np.concatenate([trunk, segment], axis=0)
            # assert torch.sum(torch.isnan(segment)) < 1, fname
        return trunk

    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def __getitem__(self, item):
        # [samples, channels], 2**15, int
        data = {}

        if(os.getpid() not in self.init_processes):
            self.init_processes.append(os.getpid())
            self.set_seed(os.getpid())

        for k in self.data_all.keys():
            wave_samples = []
            for _ in range(self.overlap_num):
                wave_samples.append(self.random_trunk(self.frame_length, type=k))
            data[k] = np.sum(wave_samples,axis=0)
            # augmentation
            data[k] = torch.tensor(data[k].astype(np.float32))
            data[k] = constrain_length_torch(data[k], int(self.sample_rate * self.frame_length))
            if(self.aug_sources is not None and len(self.aug_sources) != 0 and k in self.aug_sources):
                data[k+"_aug"] = self.aug.perform(frames=data[k],effects=self.aug_effects)
                data[k+"_aug"] = torch.tensor(data[k+"_aug"].astype(np.float32))
                data[k+"_aug"] = constrain_length_torch(data[k+"_aug"][...,None], int(self.sample_rate * self.frame_length))
        return data

    def __len__(self):
        # A Epoch = every 100 hours of dataloaders
        return int(3600*self.hours_for_an_epoch / self.frame_length)

def aug_test():
    from general_speech_restoration.config import Config
    from dataloaders.main import DATA
    from tools.file.wav import save_wave
    dl = FixLengthAugRandomDataLoader(data=DATA.get_trainset("vctk") , type_of_sources=["vocals"], overlap_num=1, frame_length=3.0,sample_rate=44100,
                                   aug_effects = ["quant","low_pass","high_pass","reverb_rir","clip","reverb_freeverb"], aug_sources = ["vocals"], aug_conf = Config.aug_conf)
    # dl = torch.utils.data.DataLoader(dataset=dl,batch_size=4,shuffle=False,num_workers=2)
    # ,"bass","treble","clip","low_pass","high_pass","fade"
    for id,each in enumerate(dl):
        save_wave(each['vocals'].numpy(),str(id)+".wav",sample_rate=44100)
        save_wave(each['vocals_aug'].numpy(),str(id)+"aug.wav",sample_rate=44100)
        print(id)
        if(id>20):break
        print(each.keys())
        print(each['vocals'].size())

if __name__ == "__main__":
    aug_test()
