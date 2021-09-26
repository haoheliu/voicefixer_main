import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
r = os.path.dirname(git_root)

from tools.file.path import find_and_build
from tools.file.wav import *

ROOT = os.path.join(r,"voicefixer_main/datasets/se/vd_noise")
DATA = os.path.join(r,"voicefixer_main/dataIndex")

convert_flac_to_wav(ROOT)

find_and_build("",ROOT)
find_and_build("",DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "vd_noise")

find_and_build(SOFTLINKSAVEDIR, "")

SubDir = ROOT
train = SOFTLINKSAVEDIR

print("find "+SubDir+" -iname *.wav > "+os.path.join(train,"vd_noise.lst"))
os.system("find "+SubDir+" -iname *.wav > "+os.path.join(train,"vd_noise.lst"))


