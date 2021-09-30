import torch.utils
from torchaudio.transforms import MelScale
import torch.utils.data
from voicefixer import Vocoder
from callbacks.base import *
from tools.pytorch.losses import *
from general_speech_restoration.config import Config
from tools.pytorch.pytorch_util import *
from general_speech_restoration.unet.model.model_kqq import UNetResComplex_100Mb
from tools.pytorch.random_ import *
from tools.file.wav import *
from tools.file.io import load_json, write_json
from dataloaders.augmentation.base import add_noise_and_scale_with_HQ

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_mel_weig(base=8):
    samplerate = 44100
    n_mel = 128
    alpha = 2595.0

    f_max = samplerate // 2
    # Converts a frequency in hertz to mel
    m_min = alpha * math.log10(1.0 + (0 / 700.0))
    m_max = alpha * math.log10(1.0 + (22050 / 700.0))
    # Quantify
    m_pts = torch.linspace(m_min, m_max, n_mel + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (base ** (m_pts / alpha) - 1.0)
    norm = (f_pts[2:n_mel + 2] - f_pts[:n_mel]) / 2.0
    return norm/norm[0]

def to_log(input):
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input

class Discriminator_7(nn.Module):
    def __init__(self, feature_height):
        super(Discriminator_7, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            nn.MaxPool2d((1,2),stride=(1,2)),
            *discriminator_block(16, 32),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            *discriminator_block(32, 64),
            nn.MaxPool2d((1, 2), stride=(1, 2)),
            *discriminator_block(64, 128),
        )

        self.adv_layer = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=1,kernel_size=3,stride=1,padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout2d(0.25),
                                       nn.Linear(feature_height,1),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity

class Discriminator_8(nn.Module):
    def __init__(self, feature_height):
        super(Discriminator_8, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        self.adv_layer = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=1,kernel_size=3,stride=1,padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout2d(0.25),
                                       nn.Linear(feature_height,1),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity

class Discriminator_9(nn.Module):
    def __init__(self, feature_height):

        super(Discriminator_9, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        self.adv_layer = nn.Sequential(nn.Conv2d(in_channels=512,out_channels=1,kernel_size=3,stride=1,padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout2d(0.25),
                                       nn.Linear(feature_height,1),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity


class BN_GRU(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,layer=1, bidirectional=False, batchnorm=True, dropout=0.0):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if(batchnorm):self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self,inputs):
        # (batch, 1, seq, feature)
        if(self.batchnorm):inputs = self.bn(inputs)
        out,_ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)

class Generator(nn.Module):
    def __init__(self,n_mel,hidden,channels):
        super(Generator, self).__init__()

        self.denoiser = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Linear(hidden, n_mel * 6),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel*6, n_mel * 4),
            nn.ReLU(inplace=True),
            BN_GRU(input_dim=n_mel * 4, hidden_dim=n_mel * 2, bidirectional=True, layer=2, batchnorm=True, dropout=0.1),
            BN_GRU(input_dim=n_mel * 4, hidden_dim=n_mel * 2, bidirectional=True, layer=2, batchnorm=True, dropout=0.1),

            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, n_mel * 4),
            nn.Dropout(0.1),

            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(n_mel * 4, hidden),
            nn.Sigmoid()
        )

        self.unet = UNetResComplex_100Mb(channels=channels)

    def forward(self,sp, noisy_wav):
        # Denoising
        clean = self.denoiser(sp) * sp
        unet_out = self.unet(clean, noisy_wav)['wav']
        return {'wav': unet_out, "clean": clean}

class ResUNet(pl.LightningModule):
    def __init__(self, channels, type_target, nsrc=1, loss="l1",
                 lr=0.002, gamma=0.9,
                 batchsize=None, frame_length=None,
                 sample_rate=None,
                 warm_up_steps=1000, reduce_lr_steps=15000,
                 # dataloaders
                 check_val_every_n_epoch=5,
                 ):
        super(ResUNet, self).__init__()

        if(sample_rate == 44100):
            window_size = 2048
            hop_size = 441
            n_mel = 128
        elif(sample_rate == 24000):
            window_size = 768
            hop_size = 240
            n_mel = 80
        elif(sample_rate == 16000):
            window_size = 512
            hop_size = 160
            n_mel = 80
        else:
            raise ValueError("Error: Sample rate "+str(sample_rate)+" not supported")

        center = True,
        pad_mode = 'reflect'
        window = 'hann'
        freeze_parameters = True

        self.save_hyperparameters()
        self.nsrc = nsrc
        self.type_target = type_target
        self.channels = channels
        self.lr = lr
        self.generated = None
        self.gamma = gamma
        self.sample_rate = sample_rate
        self.sample_rate = sample_rate
        self.batchsize = batchsize
        self.frame_length = frame_length
        # self.hparams['channels'] = 2
        self.simelspecloss = get_loss_function(loss_type="simelspec")
        self.l1loss = get_loss_function(loss_type="l1")
        self.bce_loss = get_loss_function(loss_type="bce")
        self.f_loss = get_loss_function(loss_type="l1_sp")

        # self.am = AudioMetrics()
        # self.im = ImgMetrics()
        #
        # self.local_discriminator = Discriminator_7(feature_height=n_mel)
        self.discriminator = Discriminator_7(feature_height=n_mel)

        self.valid = None
        self.fake = None

        self.train_step = 0
        self.val_step = 0
        self.val_result_save_dir = None
        self.val_result_save_dir_step = None
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.check_val_every_n_epoch = check_val_every_n_epoch

        hidden = window_size // 2 + 1

        self.mel = MelScale(n_mels=n_mel, sample_rate=sample_rate, n_stft=hidden)

        # masking
        self.generator = Generator(n_mel,hidden,channels)

        window_size = 2048
        hop_size = 441
        center = True,
        pad_mode = 'reflect'
        window = 'hann'
        freeze_parameters = True

        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
        )

        self.lr_lambda = lambda step: self.get_lr_lambda(step,
                                                        gamma = self.gamma,
                                                        warm_up_steps=warm_up_steps,
                                                        reduce_lr_steps=reduce_lr_steps)

        self.lr_lambda_2 = lambda step: self.get_lr_lambda(step,
                                                        gamma = self.gamma,
                                                        warm_up_steps=10,
                                                        reduce_lr_steps=reduce_lr_steps)

        self.mel_weight_44k_128 = Config.mel_weight_44k_128
        # self.mel_weight_loss = get_mel_weig(10)
        self.init_weights()

    def get_f_helper(self):
        return self.f_helper

    def get_lr_lambda(self,step, gamma, warm_up_steps, reduce_lr_steps):
        r"""Get lr_lambda for LambdaLR. E.g.,

        .. code-block: python
            lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

            from torch.optim.lr_scheduler import LambdaLR
            LambdaLR(optimizer, lr_lambda)
        """
        if step <= warm_up_steps:
            return step / warm_up_steps
        else:
            return gamma ** (step // reduce_lr_steps)

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def forward(self, sp, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(sp, mel_orig)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(),
                                       lr=self.lr, amsgrad=True, betas=(0.5, 0.999))

        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_g, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer_g], [scheduler_g]

    def preprocess(self, batch, train=False, cutoff=None):
        if(train):
            vocal = batch[self.type_target]
            noise = batch['noise_LR']
            LR = batch[self.type_target+'_aug_LR']

            vocal, LR, noise = vocal.float().permute(0, 2, 1), LR.float().permute(0, 2, 1), noise.float().permute(0, 2, 1)
            # LR, noise = self.add_random_noise(LR, noise)
            snr, scale = [],[]
            for i in range(vocal.size()[0]):
                vocal[i,...], LR[i,...], noise[i,...], _snr, _scale = add_noise_and_scale_with_HQ(vocal[i,...],LR[i,...], noise[i,...], snr_l=-5,snr_h=45, scale_lower=0.3, scale_upper=1.0)
                snr.append(_snr), scale.append(_scale)
            return vocal, LR, noise + LR
        else:
            if(cutoff is None):
                LR_noisy = batch["noisy"]
                LR = batch["vocals"]
                vocals = batch["vocals"]
                vocals, LR, LR_noisy = vocals.float().permute(0, 2, 1), LR.float().permute(0, 2, 1), LR_noisy.float().permute(0, 2, 1)
                return vocals, LR, LR_noisy, batch['fname'][0]
            else:
                LR_noisy = batch["noisy"+"LR"+"_"+str(cutoff)]
                LR = batch["vocals" + "LR" + "_" + str(cutoff)]
                vocals = batch["vocals"]
                vocals, LR, LR_noisy = vocals.float().permute(0, 2, 1), LR.float().permute(0, 2, 1), LR_noisy.float().permute(0, 2, 1)
                return vocals, LR, LR_noisy, batch['fname'][0]

    def info(self,string:str):
        lg.info("On trainer-" + str(self.trainer.global_rank) + ": " + string)

    def training_step(self, batch, batch_nb):
        # dict_keys(['vocals', 'vocals_aug', 'vocals_augLR', 'noise'])
        self.vocal, self.LR, self.LR_noisy = self.preprocess(batch, train=True)

        # for i in range(self.vocal.size()[0]):
        #     save_wave(tensor2numpy(self.vocal[i, ...]), str(i) + "vocal" + ".wav", sample_rate=44100)
        #     save_wave(tensor2numpy(self.LR[i, ...]), str(i) + "vocal" + ".wav", sample_rate=44100)
        #     save_wave(tensor2numpy(self.LR_noisy[i, ...]), str(i) + "LR_noisy" + ".wav", sample_rate=44100)

        # all_mel_e2e in non-log scale
        self.sp_target, self.mel_target = self.pre(self.vocal)
        self.sp_LR_target, self.mel_LR_target = self.pre(self.LR)
        self.sp_LR_target_noisy, self.mel_LR_target_noisy = self.pre(self.LR_noisy)

        self.generated = self(self.sp_LR_target_noisy, self.LR_noisy)

        targ_loss = self.f_loss(self.generated['wav'], self.vocal)
        denoise_loss = self.l1loss(self.generated['clean'], self.sp_LR_target)
        # addition_loss = self.loss(self.generated['addition'], target_addition)
        loss = targ_loss + denoise_loss

        self.log("targ-l", targ_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True, prog_bar=True)
        self.log("nois-l", denoise_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True, prog_bar=True)

        all_loss = loss
        self.train_step += 1.0
        return {"loss": all_loss}