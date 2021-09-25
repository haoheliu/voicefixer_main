# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 4:46 下午
# @Author  : Haohe Liu
# @contact: liu.8948@osu.edu
# @FileName: random_server.py

from tools.pytorch.random_ import *

class RandomServer:
    def __init__(self, p_effects = None, rir_nums = None):
        self.rir_nums = rir_nums
        self.effects_type = {
            'tempo': 1,
            'speed': 1,
            'fade': 1.1, # need length, should perform in the first place
            'pitch': 1,
            'treble': 1,
            'bass': 1,
            'tremolo': 1,
            'clip':1,
            'reverse':1,
            'reverb_freeverb': 1,
            'reverb_rir': 0,
            'low_pass': 1,
            'high_pass': 1,
            'time_dropout': 0,
            'empty_c': 0,
            'empty_n': 0,
            'beep': 0,
            'quant':0,
            'anytime': 3,
        }
        self.p_effects = {
            # target
            'tempo':{
                'prob':[0.5,0.5],
                'speed_up_range':[1.1,1.6],
                'speed_down_range':[0.7,0.95]
            },
            'speed':{
                'prob':[0.5,0.5],
                'speed_up_range': [1.1, 1.6],
                'speed_down_range': [0.7, 0.95]
            },
            'fade':{
                'prob': [0.3],
                # 'mode':'fade_same_in_out', # fade_in, fade_same_in_out, fade_out
                'fade_in_portion': [0.05,0.1],
                'fade_out_portion': [0.05,0.1]
            },
            'pitch':{
                'prob':[0.5, 0.5],
                'pitch_up_range':[100,350],
                'pitch_down_range': [-350,-100]
            },
            'treble': {
                'prob': [0.3],
                'level': [3,20]
            },
            'bass': {
                'prob': [0.3],
                'level': [3, 20]
            },
            'tremolo':{
                'prob':[0.3],
                'level':[5,50]
            },
            'reverb_freeverb':{
                'prob':[0.3],
                'reverb_level':[0,100], #0-100
                'dumping_factor':[0,100], #0-100
                'room_size':[0,100], #0-100
            },
            'reverb_rir':{
                'prob':[0.8],
                'rir_file_name':None
            },
            'low_pass':{
                'prob':[0.3],
                'low_pass_range': [3000,7000],
            },
            'high_pass':{
                'prob':[0.3],
                'high_pass_range': [500, 2000]
            },
            'clip':{
                'prob':[0.5],
                'louder_time':[1.5,10.0]
            },
            'reverse':{
                'prob':[0.1]
            },
            'time_dropout':{
                'prob':[0.1],
                'max_segment':0.2,
                'drop_range':[0.0,1.0]
            },
            'quant':{
                'prob': [0.2],
                'bins': [3, 12]
            },
            'empty_c':{
                'prob':[0.2]
            },
            # noise
            'empty_n':{
                'prob':[1.0]
            },
            'beep':{
                'prob':[1.0]
            },
            # others
            # 'inner_segment_scale':[0.1,1.0],
            # 'overall_scale':[0.6,1.0],
            # 'first_segment_portion':[0.25,0.75],
            # 'snr_range': [-5, 45]
        }

        if(p_effects is not None):
            for each in p_effects.keys():
                self.updateEffect(each,p_effects[each])

    def setEffect(self,effects):
        self.p_effects = effects

    def getEffect(self):
        return self.p_effects

    def updateEffect(self,effect,params):
        self.p_effects[effect] = params

    def mute_effect(self,effect):
        self.p_effects[effect]['prob'][0] = 0.0

    def generate(self, effect_list = None):
        result = {}
        if(effect_list is None):
            effect_list = list(self.p_effects.keys())
        for each in effect_list:
            decision,params = self.do(effect_name=each)
            if(decision):
                result[each] = [decision,params]
        return result

    def sample(self,range):
        assert len(range) == 2
        return float(uniform_torch(lower=range[0],upper=range[1]))

    def do(self, effect_name):
        decision, chance = random_select(self.p_effects[effect_name]['prob'])
        if(not decision[0]): return False, None

        if(effect_name == "empty_c"):
            return True,None
        elif(effect_name == "empty_n"):
            return True,None
        elif(effect_name == "reverse"): # todo
            return True,None
        elif(effect_name == "tempo"):
            if(decision[1]): speed = self.sample(self.p_effects['tempo']['speed_up_range'])
            else: speed = self.sample(self.p_effects['tempo']['speed_down_range'])
            return decision,speed
        elif(effect_name == "speed"):
            if(decision[1]): speed = self.sample(self.p_effects['speed']['speed_up_range'])
            else: speed = self.sample(self.p_effects['speed']['speed_down_range'])
            return decision,speed
        elif(effect_name == "pitch"):
            if(decision[1]): pitch = self.sample(self.p_effects['pitch']['pitch_up_range'])
            else: pitch = self.sample(self.p_effects['pitch']['pitch_down_range'])
            return decision, pitch
        elif(effect_name == "treble"):
            level = self.sample(self.p_effects['treble']['level'])
            return decision, level
        elif(effect_name == "bass"):
            level = self.sample(self.p_effects['bass']['level'])
            return decision, level
        elif(effect_name == "tremolo"):
            level = self.sample(self.p_effects['tremolo']['level'])
            return decision, level
        elif(effect_name == "clip"): #todo
            level = self.sample(self.p_effects['clip']['louder_time'])
            return decision, level
        elif(effect_name == "low_pass"):
            val = self.sample(self.p_effects['low_pass']['low_pass_range'])
            return decision, val
        elif(effect_name == "high_pass"):
            val = self.sample(self.p_effects['high_pass']['high_pass_range'])
            return decision, val
        elif(effect_name == "reverb_rir"):
            random_rir_index = int(uniform_torch(upper=self.rir_nums,lower=0))
            return True, random_rir_index
        elif(effect_name == "reverb_freeverb"):
            reverb_level = self.sample(self.p_effects['reverb_freeverb']['reverb_level'])
            dumping_factor = self.sample(self.p_effects['reverb_freeverb']['dumping_factor'])
            room_size = self.sample(self.p_effects['reverb_freeverb']['room_size'])
            return True,[reverb_level,dumping_factor,room_size]
        elif(effect_name == "time_dropout"):
            start = uniform_torch(lower=self.p_effects['time_dropout']['drop_range'][0],
                                  upper=self.p_effects['time_dropout']['drop_range'][1]-self.p_effects['time_dropout']['max_segment'])
            trunk_length = uniform_torch(lower=0.0,
                                         upper=self.p_effects['time_dropout']['max_segment'])
            return True,[start,trunk_length]
        elif(effect_name == "quant"):
            val = self.sample(self.p_effects['quant']['bins'])
            return decision, int(val)
        elif(effect_name == 'fade'):
            fade_in_portion=self.sample(self.p_effects['fade']['fade_in_portion'])
            fade_out_portion=self.sample(self.p_effects['fade']['fade_out_portion'])
            return True,[fade_in_portion,fade_out_portion]
        elif(effect_name == 'anytime'):
            inner_segment_scale = self.sample(self.p_effects['anytime']['inner_segment_scale'])
            overall_scale = self.sample(self.p_effects['anytime']['overall_scale'])
            first_segment_portion = self.sample(self.p_effects['anytime']['first_segment_portion'])
            snr_range = self.sample(self.p_effects['anytime']['snr_range'])
            return True, [inner_segment_scale,overall_scale,first_segment_portion,snr_range]
        else:
            raise ValueError("Unknown effect ",effect_name)