# -*- coding: utf-8 -*-
# @Time    : 2020/11/6 3:36 下午
# @Author  : Haohe Liu
# @contact: haoheliu@gmail.com
# @FileName: magical_effects.py

import numpy as np
import torch
import augment
import scipy.signal as s
from dataloaders.augmentation.random_server import RandomServer
from tools.others.audio_op import smooth
import os.path as op
from os import listdir
from tools.pytorch.random_ import random_select

class MagicalEffects:
    def __init__(self, p_effects, rir_dir=None):
        if(rir_dir is not None):
            self.rir_list = [op.join(rir_dir,each) for each in listdir(rir_dir)]
            if(len(self.rir_list) == 0):
                raise RuntimeError("Error: no rir file found")

            self.ps = RandomServer(p_effects=p_effects, rir_nums=len(self.rir_list)-1)
        else:
            self.ps = RandomServer(p_effects=p_effects, rir_nums=0)

    def setEffect(self, p_effects):
        self.ps.setEffect(effects=p_effects)

    def updateEffect(self,effect,params):
        self.ps.p_effects[effect] = params

    def pick_out_effects(self, effects:dict, type_=None):
        if(type_ is None):raise ValueError("You should specify a type you wanna pick out.")
        effect_chain = []
        for key in effects.keys():
            if(self.ps.effects_type[key] == type_):effect_chain.append(key)
        return effect_chain

    def generate_effect_chain(self,effects, audio_shape, sample_rate = 44100):
        chain = augment.EffectChain()
        # type1.1, the effect which needs length param
        items = self.pick_out_effects(effects=effects, type_=1.1)
        for key in items:
            if(key == 'fade'): chain = self.fade(chain,effects['fade'],audio_len=audio_shape[0]/sample_rate)
            else:
                raise ValueError("Bad parameter",key,"in probability setting")
        items = self.pick_out_effects(effects=effects, type_=1)
        for key in items:
            if(key == 'pitch'): chain = self.pitch(chain,effects['pitch'],sample_rate=sample_rate)
            elif(key == 'tempo'): chain = self.tempo(chain,effects['tempo'])
            elif(key == 'speed'): chain = self.speed(chain,effects['speed'])
            elif(key == 'treble'): chain = self.treble(chain,effects['treble'])
            elif(key == 'bass'): chain = self.bass(chain,effects['bass'])
            elif(key == 'tremolo'): chain = self.tremolo(chain,effects['tremolo'])
            elif(key == 'clip'): chain = self.clip(chain,effects['clip'])
            elif(key == 'reverb_freeverb'):chain = self.reverb_freeverb(chain,effects['reverb_freeverb'])
            elif(key == 'low_pass'):chain = self.low_pass(chain,effects['low_pass'])
            elif(key == 'high_pass'):chain = self.high_pass(chain,effects['high_pass'])
            elif(key == 'reverse'):chain = self.reverse(chain,effects['reverse'])
            else:
                raise ValueError("Bad parameter",key,"in probability setting")
        return chain

    def effect(self, frames, effects: list, sample_rate = 44100, rir=None, return_effects=False):
        """
        :param _frames:
        :param effects:
        :param sample_rate:
        :param rir:
        :return:
        """
        if(isinstance(frames, torch.Tensor)): _frames = frames.clone()
        else: _frames = frames.copy()

        if(not isinstance(_frames, torch.Tensor)):
            _frames = torch.tensor(_frames, dtype=torch.float32)
        effects = self.ps.generate(effects)
        effect_chain_type_0 = self.generate_effect_chain(effects, audio_shape=_frames.size(), sample_rate=sample_rate)
        LARGESCALE = False
        if(torch.max(_frames) > 10): # todo
            LARGESCALE = True
            _frames /= 2 ** 15
        _frames = _frames.squeeze()
        _frames = effect_chain_type_0.apply(_frames, src_info={'rate':sample_rate}, target_info={'rate':sample_rate})
        _frames = _frames.squeeze()
        if(isinstance(_frames, torch.Tensor)):
            _frames = _frames.numpy()
        effects_type_0 = self.pick_out_effects(effects,type_=0)
        if('empty_c' in effects_type_0):
            _frames = self.empty_c(_frames)
            return _frames, effects
        for key in effects_type_0:
            if(key == 'reverb_rir'):
                # _frames = self.reverb_rir(_frames, np.load(self.rir_list[effects['reverb_rir'][1]], encoding='bytes', allow_pickle=True))
                _frames = self.reverb_rir(_frames, np.load(self.rir_list[effects['reverb_rir'][1]]))
            elif(key == 'time_dropout'):
                _frames = self.time_dropout(_frames, effects['time_dropout'])
            elif (key == 'quant'):
                _frames = self.quantification(_frames, effects['quant'])
        if(LARGESCALE):
            _frames *= 2 ** 15
        # print(effects)
        if(return_effects):
            return _frames, effects# , todo
        else:
            return _frames

    def tempo(self, chain, params):
        params = params[1]
        return chain.tempo(float(params))

    def speed(self, chain, params):
        params = params[1]
        return chain.speed(float(params))

    def fade(self, chain, params,audio_len):
        params = params[1]
        return chain.fade(float(params[0]*audio_len),float(audio_len),float(audio_len*params[1]))

    def pitch(self,chain, params, sample_rate=44100):
        params = params[1]
        return chain.pitch(float(params)).rate(sample_rate)

    def reverb_freeverb(self, chain, params):
        if(len(params[1]) != 3):raise ValueError("Bad parameters in reverb_freeverb()")
        reverberance, dumping, room_size = params[1]
        return chain.reverb(float(reverberance), float(dumping), float(room_size)).channels(1)

    def low_pass(self,chain,params):
        value = params[1]
        return chain.lowpass(float(value))

    def reverse(self,chain,params):
        return chain.reverse()

    def clip(self,chain,params):
        value = params[1]
        return chain.clip(float(1/value))

    def high_pass(self,chain,params):
        value = params[1]
        return chain.highpass(float(value))

    def treble(self,chain,params):
        value = params[1]
        return chain.treble(float(value))

    def bass(self,chain,params):
        value = params[1]
        return chain.bass(float(value))

    def tremolo(self,chain,params):
        value = params[1]
        return chain.tremolo(float(value))

    def reverb_rir(self,frames,rir):
        orig_frames_shape = frames.shape
        frames,filter = np.squeeze(frames),np.squeeze(rir)
        frames = s.convolve(frames,filter)
        actlev = np.max(np.abs(frames))
        if(actlev > 0.99):
            frames = (frames / actlev) * 0.98
        frames = frames[:orig_frames_shape[0]]
        # print(frames.shape, orig_frames_shape)
        return frames

    def time_dropout(self,frames,params):
        start,trunk_length = params[1]
        length = frames.shape[0]
        start,end = int(length*start),int(length*start+length*trunk_length)
        frames[start : end] = np.zeros_like(frames[start : end])
        frames=smooth(frames,smooth_center=start)
        frames=smooth(frames,smooth_center=end)
        return frames

    def quantification(self,samples, params):
        bins = params[1]
        level = 2 ** bins
        max = np.max(np.abs(samples))
        samples = samples / np.max(np.abs(samples))
        space = np.linspace(start=-(1 / level) * level + (1 / level), stop=(1 / level) * level - (1 / level), num=level,
                            endpoint=True)
        digitalized_samples = np.digitize(samples, space, right=True) - 1
        digitalized_samples = space[digitalized_samples]
        return digitalized_samples * max

    def empty_c(self, frames):
        return np.zeros_like(frames)



