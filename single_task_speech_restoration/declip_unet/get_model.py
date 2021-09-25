from single_task_speech_restoration.declip_unet.model_kqq_lstm_mask_gan.model import DNN as unet

def get_model(name:str):
    if(name == "unet"):
        return unet
    else:
        raise ValueError("Model name "+name+" not recognized!")