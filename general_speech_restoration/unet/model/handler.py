import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from tools.file.wav import *
from general_speech_restoration.unet.model import ResUNet as Model

from tools.pytorch.pytorch_util import *
from tools.file.hdfs import *
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tools.file.hdfs import hdfs_get
from tools.pytorch.pytorch_util import from_log, to_log
from matplotlib import cm
from evaluation import Config
from evaluation import evaluation
from evaluation import AudioMetrics

EPS=1e-8

def load_wav_energy(path, sample_rate, threshold=0.95):
    wav_10k, _ = librosa.load(path, sr=sample_rate)
    stft = np.log10(np.abs(librosa.stft(wav_10k))+1.0)
    fbins = stft.shape[0]
    e_stft = np.sum(stft, axis=1)
    for i in range(e_stft.shape[0]):
        e_stft[-i-1] = np.sum(e_stft[:-i-1])
    total = e_stft[-1]
    for i in range(e_stft.shape[0]):
        if(e_stft[i] < total*threshold):continue
        else: break
    return wav_10k, int((sample_rate//2) * (i/fbins))

def load_wav(path, sample_rate, threshold=0.95):
    wav_10k, _ = librosa.load(path, sr=sample_rate)
    return wav_10k

def trim_center(est, ref):
    diff = np.abs(est.shape[-1] - ref.shape[-1])
    if (est.shape[-1] == ref.shape[-1]):
        return est, ref
    elif (est.shape[-1] > ref.shape[-1]):
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., int(diff // 2):-int(diff // 2)], ref
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref
    else:
        min_len = min(est.shape[-1], ref.shape[-1])
        est, ref = est, ref[..., int(diff // 2):-int(diff // 2)]
        est, ref = est[..., :min_len], ref[..., :min_len]
        return est, ref

def handler_copy(input, output, target, device) -> dict:
    """
    :param input: Input path of a .wav file
    :param output: Save path of your result. (.wav)
    :param device: Torch.device
    :return:
    """
    os.system("cp "+input+" "+output)
    return {}

# class cnn(nn.Module):
#     def __init__(self):
#         super(cnn, self).__init__()
#         self.model = nn.Conv2d(1,1,kernel_size=1)
#         self.f_helper = FDomainHelper()
#         self.mel = MelScale(n_mels=128, sample_rate=44100, n_stft=1025)
#         self.vocoder = Vocoder(sample_rate=44100)
#     def forward(self, y, x):
#         return {'mel': self.model(to_log(x))}

def pre(input, device):
    input = input[None, None, ...]
    input = torch.tensor(input).to(device)
    sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
    mel_orig = model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
    # return model.to_log(sp), model.to_log(mel_orig)
    return sp, mel_orig, input

model = None
am = None

def refresh_model(ckpt):
    global model,am
    # model = cnn()
    model = Model(channels=2,type_target="vocals", sample_rate=44100).load_from_checkpoint(ckpt)
    am = AudioMetrics(rate=44100)
    model.eval()

def handler(input, output, target, ckpt, device, needrefresh=False, meta={}):
    if(needrefresh): refresh_model(ckpt)
    global model
    # if(model.device != device):
    model = model.to(device)
    metrics = {}
    with torch.no_grad():
        wav_10k = load_wav(input, sample_rate=44100)
        if(target is not None):
            target = load_wav(target, sample_rate=44100)
        # wav_10k,cutoff = load_wav_energy(input, sample_rate=44100)
        # wav_10k = lowpass(wav_10k, highcut=cutoff, fs=44100, order=5, _type="stft")
        res = []
        seg_length = 44100*60
        break_point = seg_length
        while break_point < wav_10k.shape[0]+seg_length:
            segment = wav_10k[break_point-seg_length:break_point]
            sp,mel_noisy, segment = pre(segment, device=device)
            out_model = model(sp, segment)
            out = out_model['wav']

            sp, _, _ = model.f_helper.wav_to_spectrogram_phase(out)
            mel_out = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            if (target is not None):
                target_segment = target[break_point - seg_length:break_point]
                _, target_mel, target_segment = pre(target_segment, device)
                metrics = {
                    "mel-lsd": float(am.lsd(mel_out,target_mel)),
                    "mel-sispec": float(am.sispec(to_log(mel_out),to_log(target_mel))), # in log scale
                    "mel-non-log-sispec": float(am.sispec(mel_out, target_mel)),  # in log scale
                    "mel-ssim": float(am.ssim(mel_out,target_mel)),
                }

            # draw_and_save(out_model['clean'], needlog=True)
            # draw_and_save(out_model['addition'], needlog=False)
            # draw_and_save(mel_noisy,needlog=True)
            # draw_and_save(denoised_mel,needlog=True)

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
    models = {
        "5.1.2_unet_dereverb": "iclr_2022/dereverb/unet/a6249_log/2021-07-16-unet-#vocalsnoise#-#vctkvd_noisevocal_wav_44kdcasehq_ttsnoise_44k#-#vd_test#-fixed_4k_44k_mask_gan-l1#44100_44100#/version_0/checkpoints/epoch=20.ckpt",
    }

    key = "5.1.2_unet_dereverb"

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="", help="Model checkpoint you wanna use.")
    parser.add_argument("-l", "--limit_numbers", default=None, help="")
    parser.add_argument("-d", "--description", default="", help="")
    parser.add_argument("-t", "--testset", default="base", help="")
    parser.add_argument("-g", "--git", default=False, help="")
    args = parser.parse_args()

    if(len(args.ckpt) == 0):
        HDFS_BASE = "hdfs://haruna/home/byte_speech_sv/user/liuhaohe/exps2"
        hdfs_path = os.path.join(HDFS_BASE,models[key])
        local_path = os.path.join("checkpoints",key)
        os.makedirs(local_path,exist_ok=True)
        hdfs_get(hdfs_path=hdfs_path,local_path=local_path)
        ckpt_path = os.path.join(local_path,os.path.basename(models[key]))
    else:
        ckpt_path = args.ckpt

    description = os.getcwd().split("src")[-1] + "_"+ args.ckpt.split("log")[0]
    description = description.replace("/","_")
    if(description[0] == "."): description = description[1:]
    testset = args.testset
    assert args.limit_numbers > 1, "Error: The value of --limit_numbers(-l) shoud be greater than 1."
    evaluation(output_path=Config.EVAL_RESULT,
         handler=handler,
         ckpt=ckpt_path,
         description=key+"_"+args.description.strip()+"_"+description,
         limit_testset_to=Config.get_testsets(testset),
         limit_phrase_number=int(args.limit_numbers) if(args.limit_numbers is not None) else None)


