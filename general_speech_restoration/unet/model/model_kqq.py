import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from single_task_speech_restoration.modules import *

from tools.pytorch.losses import *
from tools.file.hdfs import *
from tools.pytorch.pytorch_util import *
from tools.pytorch.modules.fDomainHelper import FDomainHelper
from torchlibrosa.stft import  magphase

def to_log(input):
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input

class UNetResComplex_100Mb(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(UNetResComplex_100Mb, self).__init__()
        window_size = 2048
        hop_size = 441
        activation = 'relu'
        momentum = 0.01
        center = True,
        pad_mode = 'reflect'
        window = 'hann'
        freeze_parameters = True

        self.nsrc = nsrc
        self.channels = channels
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
        )

        self.encoder_block1 = EncoderBlockABNRes(in_channels=channels * nsrc, out_channels=32,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockABNRes(in_channels=32, out_channels=64,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockABNRes(in_channels=64, out_channels=128,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockABNRes(in_channels=128, out_channels=256,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockABNRes(in_channels=256, out_channels=384,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockABNRes(in_channels=384, out_channels=384,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = ConvBlockABNRes(in_channels=384, out_channels=384,
                                           size=3, activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockABNRes(in_channels=384, out_channels=384,
                                                 stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockABNRes(in_channels=384, out_channels=384,
                                                 stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockABNRes(in_channels=384, out_channels=256,
                                                 stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockABNRes(in_channels=256, out_channels=128,
                                                 stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockABNRes(in_channels=128, out_channels=64,
                                                 stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockABNRes(in_channels=64, out_channels=32,
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = ConvBlockABNRes(in_channels=32, out_channels=32,
                                                 size=3, activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        # self.conv = nn.Conv1d(in_channels=channels,out_channels=1,kernel_size=1)

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, sp, wav):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        _, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(wav)
        # shapes: (batch_size, channels_num, time_steps, freq_bins)

        x = sp
        # (batch_size, chanenls, time_steps, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.time_downsample_ratio)) \
                  * self.time_downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # (batch_size, channels, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 1025 -> 1024.
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, both=True)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, both=True)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, both=True)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, both=True)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, both=True)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, both=True)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 1024 -> 1025.
        x = x[:, :, 0: origin_len, :]  # (bs, channels * 3, T, F)

        mask_mag = x[:,0:1, :, :]

        out_mag = mask_mag

        out_real = out_mag * cos_in
        out_imag = out_mag * sin_in

        length = wav.shape[2]

        wav_out = self.f_helper.istft(out_real, out_imag, length)
        output_dict = {'wav': wav_out[:, None, :]}

        # wav_out = self.conv(wav)
        #
        # output_dict = {'wav': wav_out}

        return output_dict

if __name__ == "__main__":
    fh = FDomainHelper()
    model = UNetResComplex_100Mb(channels=1)
    wav = torch.randn((1,1,44100))
    f = to_log(fh.wav_to_spectrogram(wav))
    out = model(f,wav)['wav']

