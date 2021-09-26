import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
r = os.path.dirname(git_root)

from tools.file.wav import *

from tools.file.path import find_and_build
from evaluation import Config
from progressbar import *

ROOT = os.path.join(r,"voicefixer_main/datasets/se/TestSets")

find_and_build("",ROOT)
convert_flac_to_wav(ROOT)

Config.refresh_lists()
Config.checklst()
print("You have the following test set")
for each in Config.get_all_testsets(): print(each,end=" ")