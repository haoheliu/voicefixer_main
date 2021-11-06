from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch

class EarlyStop(Callback):
    def __init__(self, tolerance, type="min"):
        """
        Model need to return a {"loss":validation-loss} in each validation step
        :param tolerance:
        :param type:
        """
        self.result_for_each_epoch = [] # "validation-step": [val-losses,...]
        self.cache = []
        self.tolerance = tolerance
        self.type = type

    def on_validation_start(self, trainer, pl_module: pl.LightningModule) -> None:
        self.cache = []

    def on_validation_batch_end(
        self, trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.cache.append(outputs["loss"])

    def decide(self):
        print("Comparing validation loss...")
        eariler = self.result_for_each_epoch[-self.tolerance]
        print("Compare ", eariler, "to records: ", self.result_for_each_epoch[-(self.tolerance+1):])
        for value in self.result_for_each_epoch[-(self.tolerance+1):]:
            if(self.type == "min" and eariler > value):
                return False
            if (self.type == "max" and eariler < value):
                return False
        return True

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule) -> None:
        self.result_for_each_epoch.append(torch.mean(torch.tensor(self.cache)))
        if(len(self.result_for_each_epoch) > self.tolerance):
            if(self.decide()):
                print(self.result_for_each_epoch, self.type)
                print("Validation Loss haven't improve for "+str(self.tolerance)+" epoches, program will be terminated")
                exit(0)


