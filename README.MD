# General Souce Separation Framework

**! IGNORE THIS README, HAVEN'T UPDATE FOR A LONG TIME**

## 1. Directory
- datasets: contains all datasets
- env: contains codes for initialization on a new arnold machine
- src: code for experiments

```shell
.  
|-- datasets  
|   |-- mss   #Dataset type 2
|   |   -- musdb18hq # Dataset 1
|   |       |-- test  
|   |       |-- train  
|   |   -- ...
|   |-- se    #Dataset type 2
|   |-- ...
|-- env  
|   |-- conf   # Contains scripts to iniailize a arnold machine
|-- src  
    |-- dataWarehouse   #Contains all *.lst metadata files. (Each line of .lst file is a path to a wav)
    |   |-- musdb18hq  
    |       |-- test    
    |       |-- train  
    |   |-- ...
    |-- msc         # Other arbitrary modules needed
    |   |-- speechmetrics  
    |-- task_music_source_separation  # (MAIN) A Specific Task
    |   |-- voice_separation  # All model for this task
    |       |-- kqq # Model name 1
    |       |   |-- mac_log  # {machine}_log, path to all logs
    |       |       |-- 2021-03-02_test  # {time}_{name}, trail name
    |       |           |-- version_0    
    |       |               |-- checkpoints # Path to all checkpoints  
    |       |               |-- validations # Validation result
    |       |                   |-- 0       # Validation result on step {val_step}
    |       |-- unet # Model name 2
    |       |-- ...
    |-- tests  # Any test scripts
    |-- tools  # Helper functions
        |-- augmentation      # voice augmentation module
        |-- datas              # Every thing about datas
        |   |-- dataloader      # Every kinds of dataloader
        |   |-- datamodules     # Data modules for each task
        |   |-- datasetParser   # Script to construct each dataset's metadata
        |-- dsp               # dsp related code
        |-- file              # Operarion for all types of files 
        |-- others            # Other operations
        |-- pytorch           # Pytorch related operations
            |-- modules       # Helper modules
                |-- filters   # subband filters
```

## 2. Usage
```shell
# clone repo
git clone git@code.byted.org:liuhaohe.7/arnold_workspace.git

# initialize arnold machine
cd others
source init.sh

# Get datas from HDFS
cd ~/arnold_workspace/datasets/mss
hdfs dfs -get hdfs://haruna/home/byte_speech_sv/user/liuhaohe/datas/musdb18hq.tar
tar -zxvf musdb18hq.tar

# Prepare metadata for musdb18hq
cd ~/arnold_workspace/src/tools/datas/datasetParser
python3 musdb.py

# Run training script
cd ~/arnold_workspace/src/subbandMSS/voice_separation/kqq/
python3 unet_kqq.py --gpus <your-gpu-nums> --name <experiment-name> --reload <reload-model-path>

```