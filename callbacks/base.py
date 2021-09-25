from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from tools.file.hdfs import *
import logging as lg
from tools.logger.logger import setup_logging
from tools.file.io import *
import time
import glob
import git
import os
import os.path
import re

class BackUpHDFS(Callback):
    """
    Every save_step_frequency steps, this call back will send things in log_dir to HDFS
    the directory structure should look like:
    |-> src
        |-> task_<task name>
            |-> <version/comments>
                |-> <model name>

    The backup remote path is "hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2/task_<task name>/<version/comments>/<model name>"

    """
    def __init__(self, model: nn.Module,
                 current_dir:str,
                 save_step_frequency:int):
        self.model = model
        self.current_dir = current_dir
        self.save_step_frequency = save_step_frequency

    def backup(self, trainer, pl_module: pl.LightningModule) -> None:
        global_step = trainer.global_step
        if(global_step < 2):
            model_name = os.path.basename(self.current_dir)
            version = os.path.basename(os.path.dirname(self.current_dir))
            task = os.path.basename(os.path.dirname(os.path.dirname(self.current_dir)))
            target = os.path.join("hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2", task, version, model_name,
                                  os.path.dirname(pl_module.log_dir))
            os.system("echo " + target + " >> " + os.path.join(pl_module.log_dir, "hdfs_path"))
            hdfs_mkdir(target)
            os.system("echo \"hdfs dfs -put "+pl_module.log_dir+" "+target + "\" >> hdfs_put.sh")

        if(global_step>5 and global_step % self.save_step_frequency == 0):
            print("Saving files to HDFS. Global step: " + str(global_step))
            lg.info("Saving files to HDFS. Global step: "+str(global_step))
            model_name = os.path.basename(self.current_dir)
            version = os.path.basename(os.path.dirname(self.current_dir))
            task = os.path.basename(os.path.dirname(os.path.dirname(self.current_dir)))
            target = os.path.join("hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2", task, version, model_name,
                                  os.path.dirname(pl_module.log_dir))
            hdfs_put(pl_module.log_dir, target)
            lg.info("Done")

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module: pl.LightningModule) -> None:
        self.backup(trainer=trainer,pl_module=pl_module)

    def delete_part_model_val_res_hdfs(self, trainer, pl_module: pl.LightningModule):
        model_name = os.path.basename(self.current_dir)
        version = os.path.basename(os.path.dirname(self.current_dir))
        task = os.path.basename(os.path.dirname(os.path.dirname(self.current_dir)))
        target = os.path.join("hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2", task, version, model_name, pl_module.log_dir)
        # Delete validation results
        val_res_dirs = [int(x) for x in os.listdir(os.path.join(pl_module.log_dir,"validations"))]
        if(len(val_res_dirs) > 0):
            preserve_val_dir = max(val_res_dirs)
            for dir in val_res_dirs:
                if(dir == preserve_val_dir):
                    lg.info("Preserve validation result "+str(dir))
                    print("Preserve validation result",dir)
                    continue
                else:
                    lg.info("Delete validation result "+str(dir))
                    print("Delete validation result",dir)
                    hdfs_remove_dir(os.path.join(target,"validations",str(dir)))

        # Delete Checkpoints
        model_epoch = [x for x in os.listdir(os.path.join(pl_module.log_dir,"checkpoints"))]
        if(len(model_epoch) > 0):
            max_epoch = max([int(re.findall("\d+",x)[0]) for x in model_epoch])
            for epoch in model_epoch:
                if(str(max_epoch) in epoch):
                    lg.info("Preserve checkpoint "+epoch)
                    print("Preserve checkpoint",epoch)
                    continue
                else:
                    lg.info("Delete checkpoint "+epoch+" on hdfs")
                    print("Delete checkpoint",epoch,"on hdfs")
                    hdfs_remove_dir(os.path.join(target, "checkpoints", epoch))

    def delete_hdfs(self, trainer, pl_module: pl.LightningModule):
        model_name = os.path.basename(self.current_dir)
        version = os.path.basename(os.path.dirname(self.current_dir))
        task = os.path.basename(os.path.dirname(os.path.dirname(self.current_dir)))
        target = os.path.join("hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2", task, version, model_name, pl_module.log_dir)
        print("Do you want to DELETE this experiment?")
        while(True):
            in_content = input("Type in yes/no/part: ")
            if (in_content == "yes"):
                hdfs_remove_dir(target)
                lg.info("Deleted!")
                print("Deleted!")
                break
            elif(in_content == "part"):
                self.delete_part_model_val_res_hdfs(trainer,pl_module)
                break
            elif(in_content == "no"):
                break
            else:
                continue

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module: pl.LightningModule) -> None:
        print("Do you want to backup experiment result to HDFS?")
        while(True):
            in_content = input("Type in yes/no: ")
            if (in_content == "yes"): break
            elif(in_content == "no"):
                self.delete_hdfs(trainer,pl_module)
                return
            else: continue
        self.backup(trainer=trainer,pl_module=pl_module)

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
    def __init__(self, current_dir):
        self.current_dir = current_dir
        os.system("rm temp_path.json") # To avoid the previous .json file to affect this experiment

    def get_log_dir(self, pl_module: pl.LightningModule) -> str:
        return pl_module.logger.experiment.log_dir

    def code_backup(self, pl_module: pl.LightningModule):
        dir_save_code = os.path.join(pl_module.log_dir,"code")
        os.makedirs(dir_save_code,exist_ok=True)
        for file in glob.glob(os.path.join("*.sh")) + glob.glob(os.path.join("*.py")) + glob.glob(os.path.join("../*.py")) + glob.glob(os.path.join("../../*.py")):
            cmd = "cp "+file+" "+ dir_save_code
            lg.info("Backing up file: "+cmd)
            os.system(cmd)
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

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module: pl.LightningModule, outputs: Any) -> None:
        print("Working Directory:", os.getcwd())


class ArgsSaver(Callback):
    def __init__(self, args):
        self.args = args

    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module: pl.LightningModule) -> None:
        if( not os.path.exists(os.path.join(pl_module.log_dir,"args.json") ) ):
            save_pickle(self.args, os.path.join(pl_module.log_dir,"args.pkl") )
