import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from pytorch_lightning import Trainer
from pynvml import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from dataloaders.data_module import SrRandSampleRate
from tools.callbacks.base import *
from tools.callbacks.verbose import *

import tools.utils
from tools.dsp.lowpass import *
from models.gsr_voicefixer import VoiceFixer

if (not torch.cuda.is_available()):
    raise RuntimeError("Hi bro, you need GPUs to run this program.")

hp, parser = tools.utils.get_hparams()

assert hp["data"]["sampling_rate"] == 44100
hp["root"]=git_root

for k in hp["data"]["train_dataset"].keys():
    for v in hp["data"]["train_dataset"][k].keys():
        hp["data"]["train_dataset"][k][v] = os.path.join(hp["root"], hp["data"]["train_dataset"][k][v])

for k in hp["data"]["val_dataset"].keys():
    for v in hp["data"]["val_dataset"][k].keys():
        hp["data"]["val_dataset"][k][v] = os.path.join(hp["root"], hp["data"]["val_dataset"][k][v])

hp["augment"]["params"]["rir_root"] = os.path.join(hp["root"], hp["augment"]["params"]["rir_root"])

parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

nvmlInit()
gpu_nums = int(nvmlDeviceGetCount())
accelerator = 'ddp'
distributed = True if (gpu_nums > 1) else False

logger = TensorBoardLogger(os.path.dirname(hp.model_dir), name=os.path.basename(hp.model_dir))

hp["log_dir"] = logger.log_dir

model = VoiceFixer(hp, channels=1, type_target="vocals")
# print(model)
dm = SrRandSampleRate(hp, distributed)

callbacks = []
checkpoint_callback = ModelCheckpoint(
                      filename='{epoch}-{step}-{val_l:.2f}',
                      dirpath=os.path.join(logger.log_dir,"checkpoints"),
                      save_top_k=hp["train"]["save_top_k"]
                      # mode='min'
                  )
callbacks.extend([
                  LearningRateMonitor(logging_interval='step'),
                  checkpoint_callback,
                  initLogDir(hp, current_dir=os.getcwd()),
                  TQDMProgressBar(refresh_rate=hp["log"]["progress_bar_refresh_rate"])
                  ]
                 )

trainer = Trainer.from_argparse_args(args,
                                     gpus=gpu_nums,
                                     strategy=DDPPlugin(find_unused_parameters=True) if (torch.cuda.is_available()) else None,
                                     max_epochs=hp["train"]["max_epoches"],
                                     detect_anomaly=True,
                                     num_sanity_val_steps=2,
                                     resume_from_checkpoint=hp["train"]["resume_from_checkpoint"] if (len(hp["train"]["resume_from_checkpoint"]) != 0) else None,
                                     callbacks=callbacks,
                                     sync_batchnorm=True,
                                     replace_sampler_ddp=False,
                                     check_val_every_n_epoch=hp["train"]["check_val_every_n_epoch"],
                                     logger=logger,
                                     log_every_n_steps=hp["log"]["log_every_n_steps"])
dm.setup('fit')
trainer.fit(model, datamodule=dm)
