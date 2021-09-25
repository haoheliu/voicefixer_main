import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
r = os.path.dirname(git_root)

from tools.file.path import find_and_build
from tools.file.io import write_list
from tools.file.wav import *

ROOT = os.path.join(r,"voicefixer_main/datasets/se/wav48")
DATA = os.path.join(r,"voicefixer_main/dataIndex")

convert_flac_to_wav(ROOT)

find_and_build("",ROOT)
find_and_build("",DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "vctk")

find_and_build(SOFTLINKSAVEDIR, "")

data = {
    "test":{
        "fname":[], "speech":[]
    },
    "train":{
        "fname":[], "speech":[]
    }
}

SubDir = os.path.join(ROOT,"test")
test = find_and_build(SOFTLINKSAVEDIR, "test")

for each in os.listdir(SubDir):
    if(".DS_Store" in each or ".pkf" in each): continue
    speaker = os.path.join(SubDir, each)
    for audio in os.listdir(speaker):
        if (".DS_Store" in audio or ".pkf" in audio): continue
        data['test']['speech'].append(os.path.join(speaker,audio))

SubDir = os.path.join(ROOT,"train")
train = find_and_build(SOFTLINKSAVEDIR, "train")

for each in os.listdir(SubDir):
    if(".DS_Store" in each or ".pkf" in each): continue
    speaker = os.path.join(SubDir, each)
    for audio in os.listdir(speaker):
        if (".DS_Store" in audio or ".pkf" in audio): continue
        data['train']['speech'].append(os.path.join(speaker,audio))

write_list(data['test']['speech'],os.path.join(test,"speech.lst"))

write_list(data['train']['speech'],os.path.join(train,"speech.lst"))


