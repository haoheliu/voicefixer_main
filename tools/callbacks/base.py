from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import logging as lg
from tools.logger.logger import setup_logging
from tools.file.io import *
import time
import glob
import git
import os
import os.path
import re

class initLogDir(Callback):
    """
        Only when using Tensorboard with pytorch lightning can you use this call back.

        **note**
        log_dir attribute need to be explicitly declared in pl module
            log_dir:
                code
                log

        pl module need to have the following attributes:
            val_step = 0
            val_save_interval = int
    """
    def __init__(self, hp, current_dir):
        self.hp = hp
        self.current_dir = current_dir
        os.system("rm temp_path.json") # To avoid the previous .json file to affect this experiment

    def get_log_dir(self, pl_module: pl.LightningModule) -> str:
        return pl_module.logger.experiment.log_dir

    def code_backup(self, pl_module: pl.LightningModule):
        # dir_save_code = os.path.join(pl_module.log_dir,"code")
        # os.makedirs(dir_save_code,exist_ok=True)
        # for file in glob.glob(os.path.join("*.sh")) + glob.glob(os.path.join("*.py")) + glob.glob(os.path.join("../*.py")) + glob.glob(os.path.join("../../*.py")):
        #     cmd = "cp "+file+" "+ dir_save_code
        #     lg.info("Backing up file: "+cmd)
        #     os.system(cmd)
        # Save git repo info
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        os.system("echo "+sha+ " >> "+os.path.join(pl_module.log_dir,"git_version"))

    def init_path(self, trainer, pl_module: pl.LightningModule):
        try:
            # If pl_module don't have this attribute
            if(not hasattr(pl_module,"log_dir")):
                pl_module.log_dir = None
            # Have already initialized
            if(pl_module.log_dir is not None):
                return
        except AttributeError as e:
            raise AttributeError("pl_module should have log_dir attribute, whether it's initialized or not")

        if (trainer.global_rank == 0):  # Only for the main process
            pl_module.log_dir = self.get_log_dir(pl_module=pl_module)
            print("LOG DIR: ", pl_module.log_dir)
            write_json({"path": pl_module.log_dir}, fname="temp_path.json")
            os.system("cp %s %s" % (self.hp["config_path"], pl_module.log_dir))
            self.code_backup(pl_module = pl_module)
        else:
            while (not os.path.exists("temp_path.json")):
                time.sleep(1)
                lg.info("Child threading awaiting for temp_path.json ...")
            pl_module.log_dir = load_json("temp_path.json")["path"]
        # Logging directory
        setup_logging(save_dir=os.path.join(pl_module.log_dir, "log"))

    def on_train_start(self, trainer, pl_module: pl.LightningModule) -> None:
        self.init_path(trainer,pl_module)

    def on_validation_start(self, trainer, pl_module: pl.LightningModule) -> None:
        if(not hasattr(pl_module,"val_step")):
            raise AttributeError("pl_module need to have attribute val_step (track how many validation epoches have been done) set to use LogDir Callback")
        if(not hasattr(pl_module,"check_val_every_n_epoch")):
            raise AttributeError("pl_module need to have attribute check_val_every_n_epoch (the same value to check_val_every_n_epoch) set to use LogDir Callback")

        self.init_path(trainer,pl_module)
        pl_module.val_step += 1
        # the save dir of all_mel_e2e validation steps
        pl_module.val_result_save_dir = os.path.join(pl_module.log_dir, "validations")
        os.makedirs(pl_module.val_result_save_dir, exist_ok=True)

        # the save dir of this validation step
        pl_module.val_result_save_dir_step = os.path.join(pl_module.val_result_save_dir,str(pl_module.val_step * pl_module.check_val_every_n_epoch))
        os.makedirs(pl_module.val_result_save_dir_step, exist_ok=True)

    def on_test_start(self, trainer, pl_module: pl.LightningModule) -> None:
        self.init_path(trainer,pl_module)

    # @rank_zero_only
    # def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule, outputs) -> None:
    #     print("Working Directory:", os.getcwd())


class ArgsSaver(Callback):
    def __init__(self, args):
        self.args = args

    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module: pl.LightningModule) -> None:
        if( not os.path.exists(os.path.join(pl_module.log_dir,"args.json") ) ):
            save_pickle(self.args, os.path.join(pl_module.log_dir,"args.pkl") )
