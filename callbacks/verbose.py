from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class ReportDatasets(Callback):
    """
    Every save_step_frequency steps, this call back will send things in log_dir to HDFS
    the directory structure should look like:
    |-> src
        |-> task_<task name>
            |-> <version/comments>
                |-> <model name>

    The backup remote path is "hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2/task_<task name>/<version/comments>/<model name>"

    """
    def __init__(self, dm: pl.LightningDataModule, config):
        self.dm = dm
        self.config = config

    @rank_zero_only
    def on_init_end(self, trainer) -> None:
        for type in self.dm.train_data_type:
            print("For training type: ",type, ", use datasets: ")
            for pair in self.config.train_data[type].items():
                print(pair[0],pair[1])

        if(len(self.dm.val_datasets) == 0):
            print("Validation data not specified")
        for dataset in self.dm.val_datasets:
            print("For validation dataset: ",dataset)
            for type in self.config.test_data.keys():
                for ds in self.config.test_data[type].keys():
                    if(ds == dataset):
                        print(ds, self.config.test_data[type][ds])

    def check_attribute(self):
        if (not hasattr(self.dm, "train_data_type")):
            raise AttributeError("Pl Data Module should have train_data_type attribute.")
        if (not hasattr(self.dm, "test_datasets")):
            raise AttributeError("Pl Data Module should have test_datasets attribute.")
        if (not hasattr(self.dm, "val_datasets")):
            raise AttributeError("Pl Data Module should have val_datasets attribute.")
        if (not hasattr(self.config, "train_data")):
            raise AttributeError("Config should have train_data attribute.")
        if (not hasattr(self.config, "test_data")):
            raise AttributeError("Config should have test_data attribute.")