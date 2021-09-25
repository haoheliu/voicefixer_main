
import sys

sys.path.append("../../tools")
from torch.utils.data import Dataset
from dataloaders.dataloader.utils import *

class PairedFullLengthDataLoader(Dataset):
    def __init__(self,
                 dataset_name,
                 data: dict,
                 sample_rate = 44100,
                 ):
        self.init_processes = []
        self.sample_rate = sample_rate
        self.data_all = {}
        self.length = None
        self.test_set = dataset_name
        for k in data.keys():
            if(self.test_set not in data[k].keys()):
                continue
            self.data_all[k],_,_ = construct_data_folder(data[k])
            if(self.length is None):
                self.length = len(self.data_all[k][dataset_name])
        self.pairs = self.get_paired_fnames()
    '''
        {
            "type-of-source"{               # e.g. vocal, bass
                "dataset-name": "<path to .lst file>",
            }
            ...
        }
    '''
    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def get_paired_fnames(self):
        res = []
        for i in range(self.length):
            data = {}
            for k in self.data_all.keys():
                data[k] = self.data_all[k][self.test_set][i]
            res.append(data)
        return res

    def __getitem__(self, item):
        if(os.getpid() not in self.init_processes):
            self.init_processes.append(os.getpid())
            self.set_seed(os.getpid())
        # [samples, channel], 2**15, int
        data = {}
        d = self.pairs[item]
        for k in d.keys():
            data[k] = read_wave(d[k],self.sample_rate)
            # assert self.sample_rate == get_sample_rate(d[k]), d[k]
            name = os.path.splitext(os.path.split(d[k])[-1])[0]
            data['fname'] = os.path.basename(os.path.dirname(d[k])) + "_" + name
        return data

    def __len__(self):
        # A Epoch = every 100 hours of datas
        return self.length