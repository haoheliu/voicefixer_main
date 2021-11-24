[![arXiv](https://img.shields.io/badge/arXiv-2109.13731-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2109.13731) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HYYUepIsl2aXsdET6P_AmNVXuWP1MCMf?usp=sharing) [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://haoheliu.github.io/demopage-voicefixer)

**2021-11-06: I have just updated the code structure to make it easier to understand. It may have potential bug now. I will do some test training later.**

~~**2021-11-01: I will update the code and make it easier to use later.**~~

# VoiceFixer

VoiceFixer is a framework for general speech restoration. We aim at the restoration of severely degraded speech and historical speech.

- [VoiceFixer](#voicefixer)
  * [Materials](#materials)
  * [Usage](#usage)
    + [Environment (Do this at first)](#environment--do-this-at-first-)
    + [VoiceFixer for general speech restoration](#voicefixer-for-general-speech-restoration)
    + [ResUNet for general speech restoration](#resunet-for-general-speech-restoration)
    + [ResUNet for single task speech restoration](#resunet-for-single-task-speech-restoration)
  * [Citation](#citation)
  
## Materials

- *Arxiv* preprint: https://arxiv.org/abs/2109.13731 
- [Demo page](https://haoheliu.github.io/demopage-voicefixer/) contains comparison between single task speech restoration, general speech restoration, and voicefixer.
- We wrote a [pip package](https://pypi.org/project/voicefixer) for voicefixer.
- The dataset we use in this repo: [training and testing datasets](https://zenodo.org/record/5546723#.YYaWE05BxaQ)

## Usage

### Environment (Do this at first)
```shell script
# Download dataset and prepare running environment
git clone https://github.com/haoheliu/voicefixer_main.git
cd voicefixer_main
source init.sh 
```

### VoiceFixer for general speech restoration
**Here we take *VF_UNet*(voicefixer with unet as analysis module) as an example.**

- Training
```shell
# pass in a configuration file to the training script
python3 train_gsr_voicefixer.py -c config/vctk_base_voicefixer_unet.json # you can modify the configuration file to personalize your training
```
You can checkout the *logs* directory for checkpoints, logging and validation results.

- Evaluation

Automatic evaluation and generating .csv file on all testsets.

For example, if you like to evaluate on all testset (default). 
```shell script
python3 eval_gsr_voicefixer.py  \
                    --config  <path-to-the-config-file> \
                    --ckpt  <path-to-the-checkpoint> 
```

For example, if you just wanna evaluate on GSR testset. 
```shell script
python3 eval_gsr_voicefixer.py  
                    --config  <path-to-the-config-file> \
                    --ckpt  <path-to-the-checkpoint> \
                    --testset  general_speech_restoration \ 
                    --description  general_speech_restoration_eval 
```

There are generally seven testsets you can pass to **--testset**: 
- **base**: all testset
- **clip**: testset with speech that have clipping threshold of 0.1, 0.25, and 0.5
- **reverb**: testset with reverberate speech
- **general_speech_restoration**: testset with speech that contain all kinds of random distortions
- **enhancement**: testset with noisy speech
- **speech_super_resolution**: testset with low resolution speech that have sampling rate of 2kHz, 4kHz, 8kHz, 16kHz, and 24kHz.

And if you would like to evaluate on a small portion of data, e.g. 10 utterance. You can pass the number to **--limit_numbers** argument.

```shell script
python3 eval_gsr_voicefixer.py  \
                    --config  <path-to-the-config-file> \
                    --ckpt  <path-to-the-checkpoint> \
                    --limit_numbers 10 
```

Evaluation results will be presented in the *exp_results* folder.

### ResUNet for general speech restoration

- Training 
```shell
# pass in a configuration file to the training script
python3 train_gsr_voicefixer.py -c config/vctk_base_voicefixer_unet.json
```
You can checkout the *logs* directory for checkpoints, logging and validation results.

- Evaluation (similar to voicefixer evaluation)
    ```shell script
    python3 eval_ssr_unet.py  
                        --config  <path-to-the-config-file> \
                        --ckpt  <path-to-the-checkpoint> \
                        --limit_numbers <int-test-only-on-a-few-utterance> \
                        --testset  <the-testset-you-want-to-use> \ 
                        --description  <describe-this-test>
    ```

### ResUNet for single task speech restoration

- Training 
  - Denoising
  ```shell
  # pass in a configuration file to the training script
  python3 train_ssr_unet.py -c config/vctk_base_ssr_unet_denoising.json
  ```

  - Dereverberation
  ```shell
  # pass in a configuration file to the training script
  python3 train_ssr_unet.py -c config/vctk_base_ssr_unet_dereverberation.json
  ```
  
  - Super Resolution
  ```shell
  # pass in a configuration file to the training script
  python3 train_ssr_unet.py -c config/vctk_base_ssr_unet_super_resolution.json
  ```
  
  - Declipping
  ```shell
  # pass in a configuration file to the training script
  python3 train_ssr_unet.py -c config/vctk_base_ssr_unet_declipping.json
  ```

You can checkout the *logs* directory for checkpoints, logging and validation results.

  - Evaluation (similar to voicefixer evaluation)
    ```shell script
    python3 eval_ssr_unet.py  
                        --config  <path-to-the-config-file> \
                        --ckpt  <path-to-the-checkpoint> \
                        --limit_numbers <int-test-only-on-a-few-utterance> \
                        --testset  <the-testset-you-want-to-use> \ 
                        --description  <describe-this-test>
    ```

## Citation

```bib
 @misc{liu2021voicefixer,   
     title={VoiceFixer: Toward General Speech Restoration With Neural Vocoder},   
     author={Haohe Liu and Qiuqiang Kong and Qiao Tian and Yan Zhao and DeLiang Wang and Chuanzeng Huang and Yuxuan Wang},  
     year={2021},  
     eprint={2109.13731},  
     archivePrefix={arXiv},  
     primaryClass={cs.SD}  
 }
```

![real-life-example](resources/pics/real.png)
![real-life-example](resources/pics/gsr-demo.png)
![real-life-example](resources/pics/SR-2k.png)




