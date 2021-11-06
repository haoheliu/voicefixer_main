import os.path

import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from tools.file.wav import *
from models.gsr_voicefixer import VoiceFixer as Model
from tools.pytorch.pytorch_util import from_log, to_log
from evaluation_proc.config import Config
from evaluation_proc.eval import evaluation
from evaluation_proc.metrics import AudioMetrics
from tools.utils import *

EPS=1e-9

def pre(input, device):
    input = input[None, None, ...]
    input = torch.tensor(input).to(device)
    sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
    mel_orig = model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
    # return model.to_log(sp), model.to_log(mel_orig)
    return sp, mel_orig

model = None
am = None
hp = None

def refresh_model(ckpt):
    global model,am
    model = Model(hp, channels=2,type_target="vocals").load_from_checkpoint(ckpt)
    am = AudioMetrics(rate=44100)
    model.eval()

def handler(input, output, target,ckpt, device, needrefresh=False, meta={}):
    if(needrefresh): refresh_model(ckpt)
    global model
    model = model.to(device)
    metrics = {}
    with torch.no_grad():
        wav_10k = load_wav(input, sample_rate=44100)
        if(target is not None):
            target = load_wav(target, sample_rate=44100)
        res = []
        seg_length = 44100*60
        break_point = seg_length
        while break_point < wav_10k.shape[0]+seg_length:
            segment = wav_10k[break_point-seg_length:break_point]
            _, mel_noisy = pre(segment,device)
            out_model = model(mel_noisy)
            denoised_mel = from_log(out_model['mel'])
            if(meta["unify_energy"]):
                denoised_mel, mel_noisy = amp_to_original_f(mel_sp_est=denoised_mel,mel_sp_target=mel_noisy)
            if (target is not None):
                target_segment = target[break_point - seg_length:break_point]
                _, target_mel = pre(target_segment,device)
                metrics = {
                    "mel-lsd": float(am.lsd(denoised_mel,target_mel)),
                    "mel-sispec": float(am.sispec(out_model['mel'],to_log(target_mel))), # in log scale
                    "mel-non-log-sispec": float(am.sispec(from_log(out_model['mel']),target_mel)), # in log scale
                    "mel-ssim": float(am.ssim(denoised_mel,target_mel)),
                }

            out = model.vocoder(denoised_mel)
            # unify energy
            if(torch.max(torch.abs(out)) > 1.0):
                out = out / torch.max(torch.abs(out))
                print("Warning: Exceed energy limit,", input)
            # frame alignment
            out, _ = trim_center(out, segment)
            res.append(out)
            break_point += seg_length
        out = torch.cat(res,-1)
        save_wave(tensor2numpy(out[0,...]),fname=output,sample_rate=44100)
    return metrics

if __name__ == '__main__':

    from argparse import ArgumentParser
    from tools.utils import get_hparams_from_file
    parser = ArgumentParser()
    parser.add_argument("--config", default="", help="Your config file")
    parser.add_argument("--ckpt", default="", help="Model checkpoint you wanna use.")
    parser.add_argument("--limit_numbers", default=None, help="")
    parser.add_argument("--description", default="run_default_evaluation", help="")
    parser.add_argument("--testset", default="base", help="")
    args = parser.parse_args()

    hp = get_hparams_from_file(args.config)

    ckpt_path = args.ckpt

    testset = args.testset

    evaluation(
               output_path=Config.EVAL_RESULT,
                 handler=handler,
                 ckpt=ckpt_path,
                 description=os.path.splitext(os.path.basename(args.config))[0]+"_"+args.description.strip(),
                 limit_testset_to=Config.get_testsets(testset),
                 limit_phrase_number=int(args.limit_numbers) if(args.limit_numbers is not None) else None)



