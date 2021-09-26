import librosa
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import speechmetrics as sm
from torchaudio.transforms import MelScale
from evaluation.util import *

EPS = 1e-8

class ImageMetrics():
    def __init__(self):
        pass

    def evaluate(self, input, output):
        pass

class AudioMetrics():
    def __init__(self, rate):
        self.rate = rate
        self.metrics = sm.load(['sisdr','stoi','pesq'], np.inf)
        self.mel_44k = MelScale(n_mels=128, sample_rate=44100, n_stft=1025)
        self.mel_16k = MelScale(n_mels=80, sample_rate=16000, n_stft=372)

    def read(self, est, target, rate):
        est,sr = librosa.load(est,sr=rate,mono=True)
        target, sr = librosa.load(target, sr=rate, mono=True)
        return est, target

    def wav_to_spectrogram(self, wav, rate=44100):
        if(rate == 44100):
            hop_length = 441
            n_fft = 2048
            mel = self.mel_44k
        elif(rate == 16000):
            hop_length = 160
            n_fft = 743
            mel = self.mel_16k
        else:
            raise ValueError("Bad Samplerate")
        f = np.abs(librosa.stft(wav, hop_length=hop_length, n_fft=n_fft))
        f = np.transpose(f,(1,0))
        f = torch.tensor(f[None,None,...])
        return f, mel(f.permute(0,1,3,2)).permute(0,1,3,2)

    def evaluation(self, est, target):
        result = {}
        if(target is None): return result
        # time domain
        rate = get_sample_rate(target)
        scores = self.metrics(est, target, rate = rate) # slow
        result["sisdr"],result["stoi"] = scores["sisdr"],scores["stoi"]

        est_wav, target_wav = self.read(est, target, rate=rate)
        # est_16, target_16 = self.read(est, target, rate=16000)
        # result['pesq'] = float(pesq(fs=16000,ref=target_16,deg=est_16,mode="wb"))
        result["pesq"] = scores['pesq']

        est_sp, est_mel = self.wav_to_spectrogram(est_wav, rate=rate)
        target_sp, target_mel = self.wav_to_spectrogram(target_wav, rate=rate)

        # frequency domain
        result["lsd"] = self.lsd(est_sp.clone(), target_sp.clone())
        result["non_log_sispec"] = self.sispec(est_sp.clone(), target_sp.clone())
        result["sispec"] = self.sispec(to_log(est_sp.clone()), to_log(target_sp.clone()))
        result["ssim"] = self.ssim(est_sp.clone(), target_sp.clone())

        result["final_mel_lsd"] = self.lsd(est_mel.clone(), target_mel.clone())
        result["final_non_log_mel_sispec"] = self.sispec(est_mel.clone(), target_mel.clone())
        result["final_mel_sispec"] = self.sispec(to_log(est_mel.clone()), to_log(target_mel.clone()))
        result["final_mel_ssim"] = self.ssim(est_mel.clone(), target_mel.clone())

        for key in result: result[key] = float(result[key])
        return result

    def lsd(self,est, target):
        # in non-log scale
        lsd = torch.log10((target**2/(est**2 + EPS)) + EPS)**2
        lsd = torch.mean(torch.mean(lsd,dim=3)**0.5,dim=2)
        return lsd[...,None,None]

    def sispec(self,est, target):
        # in log scale
        output, target = energy_unify(est, target)
        noise = output - target
        # print(pow_p_norm(target) , pow_p_norm(noise), pow_p_norm(target) / (pow_p_norm(noise) + EPS))
        sp_loss = 10 * torch.log10((pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS))
        return torch.sum(sp_loss) / sp_loss.size()[0]

    def ssim(self,est, target):
        if("cuda" in str(target.device)):
            target, output = target.detach().cpu().numpy(), est.detach().cpu().numpy()
        else:
            target, output = target.numpy(), est.numpy()
        res = np.zeros([output.shape[0],output.shape[1]])
        for bs in range(output.shape[0]):
            for c in range(output.shape[1]):
                res[bs,c] = ssim(output[bs,c,...],target[bs,c,...],win_size=7)
        return torch.tensor(res)[...,None,None]

if __name__ == '__main__':
    au = AudioMetrics(rate=44100)
    res = au.evaluation(est="/Users/admin/Downloads/test_sample_result_test_0_orig.wav",
        target="/Users/admin/Downloads/test_sample_result_test_0_orig.wav")
    print(res)
    # a = torch.abs(torch.randn((1,1,100,128)))
    # b = torch.abs(torch.randn((1,1,100,128)))
    # print(au.lsd(a,b))
    # print(au.sispec(to_log(a),to_log(b)))
    # print(au.sispec(to_log(a)*10,to_log(b)))
    # print(au.sispec(to_log(a),to_log(b)*10))
    # print(au.sispec(to_log(a)*10,to_log(b)*10))
    # print(au.sispec(to_log(a)*10,to_log(a)*10))
    # print(au.ssim(a,b))