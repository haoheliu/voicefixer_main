import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from tools.file.wav import *
from general_speech_restoration.voicefixer.unet_small.model import ResUNet as Model

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

def draw_and_save(mel: torch.Tensor, clip_max=None, clip_min=None, needlog=True):
    plt.figure(figsize=(15, 5))
    mel = np.transpose(tensor2numpy(mel)[0, 0, ...], (1, 0))
    # assert np.sum(mel < 0) == 0, str(np.sum(mel < 0)) + str(np.sum(mel < 0))

    if (needlog):
        mel_log = np.log10(mel + EPS)
    else:
        mel_log = mel

    # plt.imshow(mel)
    librosa.display.specshow(mel_log, sr=44100, x_axis='frames', y_axis='mel', cmap=cm.jet, vmax=clip_max,
                             vmin=clip_min)
    plt.colorbar()
    plt.show()

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

def amp_to_original_f(mel_sp_est, mel_sp_target, cutoff=0.2):
    freq_dim = mel_sp_target.size()[-1]
    mel_sp_est_low, mel_sp_target_low = mel_sp_est[..., 5:int(freq_dim * cutoff)], mel_sp_target[..., 5:int(freq_dim * cutoff)]
    energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(mel_sp_target_low, dim=(2, 3))
    amp_ratio = energy_target / energy_est
    return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

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

def pre(input, device):
    input = input[None, None, ...]
    input = torch.tensor(input).to(device)
    sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
    mel_orig = model.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
    # return model.to_log(sp), model.to_log(mel_orig)
    return sp, mel_orig

model = None
am = None

def refresh_model(ckpt):
    global model,am
    # model = cnn()
    model = Model(channels=2,type_target="vocals", sample_rate=44100).load_from_checkpoint(ckpt)
    am = AudioMetrics(rate=44100)
    model.eval()

def handler(input, output, target,ckpt, device, needrefresh=False,meta={}):
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
            sp,mel_noisy = pre(segment,device)
            out_model = model(sp, mel_noisy)
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
    models = {
        "2.3.2.unet": "general_speech_restoration/voicefixer/unet/a6249_log/2021-07-16-unet-#vocalsnoise#-#vctkvd_noisevocal_wav_44kdcasehq_ttsnoise_44k#-#vd_test#-all_test-l1#1500_44100#/version_1/checkpoints/epoch=15.ckpt",
        "2.3.2.mel_e2e_no_freeverb": "general_speech_restoration/voicefixer/unet/a1982_log/2021-07-17-unet-#vocalsnoise#-#vctkvd_noisevocal_wav_44kdcasehq_ttsnoise_44k#-#vd_test#-all_test-l1#1500_44100#/version_2/checkpoints/epoch=15.ckpt",
        "small_data_vocoder": "general_speech_restoration/voicefixer/unet/a6467_log/2021-08-09-unet-#vocalsnoise#-#vctkvd_noisevocal_wav_44kdcasehq_ttsnoise_44k#-#vd_test#-all_test-l1#1500_44100#/version_0/checkpoints/epoch=35.ckpt",
    }

    key = "small_data_vocoder"

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="", help="Model checkpoint you wanna use.")
    parser.add_argument("-l", "--limit_numbers", default=None, help="")
    parser.add_argument("-d", "--description", default="", help="")
    parser.add_argument("-t", "--testset", default="base", help="")
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

    evaluation(output_path=Config.EVAL_RESULT,
         handler=handler,
         ckpt=ckpt_path,
         description=key+"_"+args.description.strip()+"_"+description,
         limit_testset_to=Config.get_testsets(testset),
         limit_phrase_number=int(args.limit_numbers) if(args.limit_numbers is not None) else None)



