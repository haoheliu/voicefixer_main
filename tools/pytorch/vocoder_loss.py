from tools.pytorch.modules.fDomainHelper import *

EPS = 1e-8

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()

    def __call__(self, output, target):
        return self.loss(output,target)

class L1_Sp(nn.Module):
    def __init__(self):
        super(L1_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        self.l1 = L1()

    def __call__(self, output, target, log_op=False):
        sp_loss = self.l1(
                self.f_helper.wav_to_spectrogram(output, eps=1e-8),
                self.f_helper.wav_to_spectrogram(target, eps=1e-8)
            )
        return sp_loss

class LSD(nn.Module):
    def __init__(self):
        super(LSD, self).__init__()
        self.f_helper = FDomainHelper()

    def __call__(self, output, target):
        """
        :param output: raw wave, shape: [batchsize, channels, samples]
        :param target: raw wave, shape: [batchsize, channels, samples]
        :return: LSD value [batchsize, channels]
        """
        output = self.f_helper.wav_to_spectrogram(output, eps=1e-8)
        target = self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        # temp[...,:-2,:] = output[...,2:,:]
        lsd = torch.log10((target**2/(output**2 + EPS)) + EPS)**2
        # plt.figure(figsize=(15,5))
        # librosa.display.specshow(lsd[0,0,...].permute(1,0).numpy())
        # plt.show()
        lsd = torch.mean(torch.mean(lsd,dim=3)**0.5,dim=2)
        return lsd[...,None,None]

# class VocoderLoss(nn.Module):
#     def __init__(self, sample_rate, loss_type="l1"):
#         super(VocoderLoss, self).__init__()
#         Config.refresh(sample_rate)
#         self.vocoder = Vocoder(sample_rate=sample_rate)
#         self.sample_rate = sample_rate
#         self.f_helper = FDomainHelper()
#         # self.loss = get_loss_function(loss_type)
#         if(loss_type == "lsd"): self.loss = LSD()
#         elif(loss_type == "l1_sp"): self.loss = L1_Sp()
#
#         for p in self.vocoder.parameters():
#             p.requires_grad = False
#
#     def trim_center(self,est, ref):
#         diff = np.abs(est.shape[-1] - ref.shape[-1])
#         if (est.shape[-1] == ref.shape[-1]):
#             return est, ref
#         elif (est.shape[-1] > ref.shape[-1]):
#             min_len = min(est.shape[-1], ref.shape[-1])
#             est, ref = est[..., int(diff // 2):-int(diff // 2)], ref
#             est, ref = est[..., :min_len], ref[..., :min_len]
#             return est, ref
#         else:
#             min_len = min(est.shape[-1], ref.shape[-1])
#             est, ref = est, ref[..., int(diff // 2):-int(diff // 2)]
#             est, ref = est[..., :min_len], ref[..., :min_len]
#             return est, ref
#
#     def __call__(self, est, target):
#         """
#         :param est: [batchsize, 1,  n_step, n_mel ]
#         :param target: [batchsize, channels, samples]
#         :return:
#         """
#         est = self.vocoder(est)
#
#         # normalize energy
#         est = est / torch.max(est)
#         target = target / torch.max(target)
#
#         # alignment
#         est, target = self.trim_center(est, target)
#
#         # loss
#         return torch.mean(self.loss(est, target))


