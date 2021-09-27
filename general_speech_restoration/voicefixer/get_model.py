from general_speech_restoration.voicefixer.unet.model import ResUNet as mel_e2e
from general_speech_restoration.voicefixer.lstm.model import DNN as mel_e2e_lstm
from general_speech_restoration.voicefixer.dnn.model import DNN as mel_e2e_dnn
from general_speech_restoration.voicefixer.unet_small.model import DNN as mel_e2e_unet_small

def get_model(name:str):
    if(name == "unet"):
        return mel_e2e
    elif(name == "lstm"):
        return mel_e2e_lstm
    elif(name == "dnn"):
        return mel_e2e_dnn
    elif(name == "unet_small"):
        return mel_e2e_unet_small
    else:
        raise ValueError("Model name "+name+" not recognized!")