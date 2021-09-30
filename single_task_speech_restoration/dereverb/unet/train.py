import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

import torchaudio

torchaudio.set_audio_backend("sox_io")
from pytorch_lightning import Trainer
from pynvml import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from single_task_speech_restoration.dereverb.get_model import *
from single_task_speech_restoration.dereverb.dm_sr_rand_sr_order import SrRandSampleRate
from dataloaders.main import DATA
from callbacks.base import *
from callbacks.verbose import *

from argparse import ArgumentParser
from single_task_speech_restoration.config import Config
from tools.dsp.lowpass import *

Config.aug_conf['reverb_rir']['prob'] = [1.0]

def report_dataset(names):
    res = "#"
    for each in names:
        res += each
    return res+"#"

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-m", "--model", required=True, default="lstm", help="Model name you wanna use.")
    parser.add_argument("-l", "--loss", default="l1_sp", help="Loss function")
    parser.add_argument("-t", "--train_dataset", nargs="+", default=["vctk","vocal_wav_44k","vd_noise","dcase"], help="Train dataset")
    parser.add_argument("-v", "--val_dataset", nargs="+", default=[], help="validation datasets.")
    parser.add_argument("-t_type", "--train_data_type", nargs="+", default=["vocals","noise"], help="Training data types.")
    parser.add_argument("-c", "--check_val_epoch", type=int, default=50,help="Every 10 hours of training data is called an epoch.")
    parser.add_argument("-r", '--reload', type=str, default="")
    parser.add_argument("-n", '--name', type=str, default="fix_samplerate")
    parser.add_argument("-g", '--gpu_nums', type=int, default=0)
    parser.add_argument("-san", '--sanity_val_steps', type=int, default=2)
    parser.add_argument("--dl", type=str, default="FixLengthAugRandomDataLoader") # "FixLengthFixSegRandomDataLoader", "FixLengthThreshRandDataLoader"
    parser.add_argument("--overlap_num", type=int, default=1)
    parser.add_argument("--aug_sources", nargs="+", default=["vocals"], help="conduct augmentation on which source")
    parser.add_argument("--aug_effects", nargs="+", default=[], help="augmentation effects")
    # experiment
    parser.add_argument("--source_sample_rate_low", type=int, default=8000)
    parser.add_argument("--source_sample_rate_high", type=int, default=24000)

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.8, help="lr exponential decay.")
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size.")
    parser.add_argument("--frame_length", type=float, default=3.0, help="frame length in seconds.")
    parser.add_argument("--warmup_data", type=float, default=26.6, help="Hours of warmup dataloaders.")
    parser.add_argument("--reduce_lr_period", type=float, default=400, help="How many hours of data per lr reduction.")
    parser.add_argument("--max_epoches", type=int, default=5000, help="Maximum epoches")
    parser.add_argument("--back_hdfs_every_hours", type=int, default=53, help="Every how many epoch do you want back up file to hdfs")
    parser.add_argument("--save_top_k", type=int, default=-1, help="")
    parser.add_argument("--save_metric_monitor", type=str, default="val_loss")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--early_stop_tolerance", type=int, default=5)
    parser.add_argument("--early_stop_crateria", default="min", help="min or max")
    # print(generate_doc(parser))
    ROOT = Config.ROOT

    if (os.path.exists("temp_path.json")):
        os.remove("temp_path.json")
    if (os.path.exists("path.json")):
        os.remove("path.json")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    Config.aug_sources = args.aug_sources
    Config.aug_effects = args.aug_effects

    current = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    name = current + "-" + args.model+"-"+report_dataset(args.train_data_type)+"-"+\
           report_dataset(args.train_dataset)+"-" + \
           report_dataset(args.val_dataset) + "-" +\
           args.name + "-" + args.loss + "#"+str(args.source_sample_rate_low)+"_"+ str(args.source_sample_rate_high) + "#"

    if (len(args.reload) != 0):
        name += "_reload"

    if (torch.cuda.is_available()):
        nvmlInit()
        if(args.gpu_nums == 0): gpu_nums = int(nvmlDeviceGetCount())
        else: gpu_nums = args.gpu_nums
        accelerator = 'ddp'
        distributed = True if (gpu_nums > 1) else False
    else:
        gpu_nums = args.gpu_nums
        accelerator = None
        distributed = False

    logger = TensorBoardLogger(save_dir=Config.TRAIL_NAME + "_log", name=name)

    if (gpu_nums != 0): seconds_per_step = gpu_nums * args.batchsize * args.frame_length
    else: seconds_per_step = args.batchsize * args.frame_length

    print("The model You choose: ")
    print(args.model)

    model = get_model(args.model)(channels=1, type_target="vocals", loss=args.loss,
                             # training
                             lr=args.lr,
                             gamma=args.gamma,
                             batchsize=args.batchsize,
                             frame_length=args.frame_length,
                             sample_rate=args.sample_rate,
                             check_val_every_n_epoch = args.check_val_epoch,
                             warm_up_steps=int(args.warmup_data * 3600 / seconds_per_step),
                             reduce_lr_steps=int(args.reduce_lr_period * 3600 / seconds_per_step))

    print(Config.aug_conf)
    print(Config.aug_sources)
    print(Config.aug_effects)

    dm = SrRandSampleRate(
        source_sample_rate_low = args.source_sample_rate_low,source_sample_rate_high = args.source_sample_rate_high,
        target_sample_rate=args.sample_rate,
        distributed=distributed, overlap_num=args.overlap_num,
        train_loader=args.dl,
        train_data=DATA.merge([DATA.get_trainset(set) for set in args.train_dataset]),
        val_data=DATA.merge([DATA.get_testset(set) for set in args.val_dataset]) if(len(args.val_dataset) != 0) else {},
        train_data_type=args.train_data_type, val_datasets=args.val_dataset,
        batchsize=args.batchsize, frame_length=args.frame_length, num_workers=22 if (torch.cuda.is_available()) else 0,
        sample_rate=args.sample_rate, aug_conf=Config.aug_conf, aug_sources=Config.aug_sources, aug_effects=Config.aug_effects,
        hours_for_an_epoch=500
    )

    callbacks = []
    callbacks.extend([
                      ArgsSaver(args),
                      LearningRateMonitor(logging_interval='step'),
                      ModelCheckpoint(
                          filename='{epoch}',
                          # monitor=args.save_metric_monitor,
                          save_top_k=-1,
                          mode='min',
                      ),
                      # BackUpHDFS(
                      #     model=model,
                      #     current_dir=os.getcwd(),
                      #     save_step_frequency=int(args.back_hdfs_every_hours * 103 * 3600 / seconds_per_step)
                      # ),
                      initLogDir(current_dir=os.getcwd()),
                      # ReportDatasets(dm=dm, config=Config),
                      # EarlyStop(tolerance=args.early_stop_tolerance,type=args.early_stop_crateria)
                      ]
                     )

    print("eval_callbacks: ")
    for each in callbacks: print(each)

    trainer = Trainer.from_argparse_args(args,
                                         gpus=gpu_nums,
                                         plugins=DDPPlugin(find_unused_parameters=True) if (torch.cuda.is_available()) else None,
                                         max_epochs=args.max_epoches,
                                         terminate_on_nan=True,
                                         num_sanity_val_steps=args.sanity_val_steps,
                                         resume_from_checkpoint=args.reload if (len(args.reload) != 0) else None,
                                         callbacks=callbacks,
                                         accelerator=accelerator,
                                         sync_batchnorm=True,
                                         replace_sampler_ddp=False,
                                         check_val_every_n_epoch=args.check_val_epoch,
                                         checkpoint_callback=True, logger=logger, log_every_n_steps=10,
                                         progress_bar_refresh_rate=1, flush_logs_every_n_steps=200)
    dm.setup('fit')
    trainer.fit(model, datamodule=dm)
    # trainer.save_checkpoint("example.ckpt")
