from single_task_speech_restoration.super_resolution.unet.model import ResUNet as unet

def get_model(name:str):
    if(name == "unet"):
        return unet
    else:
        raise ValueError("Model name "+name+" not recognized!")