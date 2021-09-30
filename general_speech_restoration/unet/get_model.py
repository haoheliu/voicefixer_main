from general_speech_restoration.unet.model.model import ResUNet as unet

def get_model(name:str):
    if(name == "unet"):
        return unet
    else:
        raise ValueError("Model name "+name+" not recognized!")