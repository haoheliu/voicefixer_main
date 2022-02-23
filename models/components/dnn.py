import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from models.components.modules import *

from tools.pytorch.losses import *
from tools.pytorch.pytorch_util import *
from tools.pytorch.modules.fDomainHelper import FDomainHelper


def to_log(input):
    return torch.log10(torch.clip(input, min=1e-8))

def from_log(input):
    input = torch.clip(input,min=-np.inf, max=5)
    return 10 ** input

class DNN(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(DNN, self).__init__()
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
        n_mel = 1025
        # n_mel = 128
        self.lstm = nn.Sequential(
            nn.Linear(n_mel, n_mel * 2),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel * 2, n_mel * 4),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel * 8, n_mel * 4),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(n_mel * 4, n_mel * 2),
            nn.ReLU(),
            nn.Linear(n_mel * 2, n_mel),
        )

        # self.conv = nn.Conv1d(in_channels=channels,out_channels=1,kernel_size=1)

        # self.init_weights()

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

        out_mag = self.lstm(sp)
        
        out_mag = torch.relu(out_mag) + sp

        out_real = out_mag * cos_in
        out_imag = out_mag * sin_in

        length = wav.shape[2]

        wav_out = self.f_helper.istft(out_real, out_imag, length)
        output_dict = {'wav': wav_out[:, None, :]}

        # wav_out = self.conv(wav)
        #
        # output_dict = {'wav': wav_out}

        return output_dict
