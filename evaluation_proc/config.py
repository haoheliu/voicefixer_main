import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from os.path import join,exists
from os import listdir, makedirs
from tools.file.wav import *
from tools.file.io import write_list, read_list

class Config:

    metrics = ['sisdr',"nb_pesq","pesq","stoi","lsd","sispec","ssim"]

    # no ellip and bessel
    SETTING1 = ["compression","declipping","enhancement_vctk","real","all_types","enhancement_dns","reverb","butter","cheby1","daps"]

    TESTSETS={
        "base": ['vctk_reverb', 'vctk_0.1', 'vctk_0.25', 'vctk_0.5',
                 'all_random_filter_type','vctk_demand',
                 'vctk_cheby1_1000', 'vctk_cheby1_2000',
                 'vctk_cheby1_4000', 'vctk_cheby1_8000', 'vctk_cheby1_12000'],
        "clip": ['vctk_0.1', 'vctk_0.25', 'vctk_0.5'],
        "reverb": ['vctk_reverb'],
        "general_speech_restoration": ['all_random_filter_type'],
        "enhancement": ['vctk_demand'],
        "speech_super_resolution": ['vctk_cheby1_1000', 'vctk_cheby1_2000', 'vctk_cheby1_4000', 'vctk_cheby1_8000', 'vctk_cheby1_12000']
    }

    r = os.path.dirname(git_root)

    # TEST_ROOT = "/Volumes/Haohe_SSD/TestSets_bak"
    TEST_ROOT = os.path.join(r, "voicefixer_main/datasets/se/TestSets")
    EVAL_RESULT = os.path.join(r, "voicefixer_main/exp_results")

    makedirs(EVAL_RESULT,exist_ok=True)

    SE_ROOT = join(TEST_ROOT,"DENOISE")  # Speech Enhancement
    SD_ROOT = join(TEST_ROOT,"DECLI")   # Speech Declipping
    SSR_ROOT = join(TEST_ROOT,"SR") # Speech Super Resolution
    SDR_ROOT = join(TEST_ROOT,"DEREV") # Speech Dereverb
    ALL_ROOT = join(TEST_ROOT,"ALL_GSR")

    ALL_DATA = {
        "all_random_filter_type": {
            "rate": 44100,
            "list": join(ALL_ROOT, "pair_random_filter_type.lst"),
            "unify_energy": False,
        },
    }

    SDR_DATA = {
        "vctk_reverb": {  # This is joint dereverb and denoising
            "rate": 44100,
            "list": join(SDR_ROOT, "pair_vctk_reverb.lst"),
            "unify_energy": False,
        }
    }

    SE_DATA = {
        "vctk_demand": {
            "rate": 44100,
            "list":  join(SE_ROOT,"vd_test","pair.lst"),
            "unify_energy": False,
        },
    }

    SD_DATA = {
        "vctk_0.1":{
            "rate": 44100,
            "list": join(SD_ROOT, "pair_0.10.lst"),
            "unify_energy": False,
        },
        "vctk_0.25": {
            "rate": 44100,
            "list": join(SD_ROOT, "pair_0.25.lst"),
            "unify_energy": False,
        },
        "vctk_0.5": {
            "rate": 44100,
            "list": join(SD_ROOT, "pair_0.50.lst"),
            "unify_energy": False,
        }
    }

    SSR_DATA = {
    }

    for type in ["cheby1"]:
        for c in [1000,2000,4000,8000,12000]:
            SSR_DATA["vctk_"+type+"_"+str(c)] = {
                "rate": 44100,
                "list": join(SSR_ROOT,type,"pair_%s_%d.lst" % (type, c)),
                "unify_energy": True,
            }

    @classmethod
    def init(cls):
        cls.refresh_lists()
        cls.checklst()
        print(cls.get_all_testsets())

    @classmethod
    def get_testsets(cls, names):
        if(type(names) != list):
            names = [names]
        res = []
        for name in names:
            if(name in cls.TESTSETS.keys()):
                res.extend(cls.TESTSETS[name])
            else:
                raise ValueError("Test set "+name+" undefined!")
        return res

    @classmethod
    def get_all_testsets(cls):
        res = []
        for each in [cls.SD_DATA, cls.SE_DATA, cls.SSR_DATA, cls.SDR_DATA,cls.ALL_DATA]:
            for k in each.keys():
                res.append(k)
        return res

    @classmethod
    def get_meta(cls, testsetname: str):
        for set in [cls.SD_DATA,cls.SSR_DATA,cls.SDR_DATA,cls.SE_DATA,cls.ALL_DATA]:
            if(testsetname in set.keys()):
                return set[testsetname]
        raise ValueError("Error: Test set not found "+testsetname)

    @classmethod
    def refresh_lists(cls):
        import re

        # ALL Types GSR
        Ground_Truth_Path = join(cls.ALL_ROOT,"target")  # todo attension here is no reverb ( instead of with reverb )
        source_dir = join(cls.ALL_ROOT,"simulated")
        ALL = []
        simulated = []
        clean = []
        for file in listdir(source_dir):
            if(file[-4:] != ".wav"): print(file);continue
            gt_name = re.findall("\d+_", file)[0]
            ALL.append(join(source_dir,file)+" "+join(Ground_Truth_Path,gt_name+"clean.wav"))
            simulated.append(join(source_dir,file))
            clean.append(join(Ground_Truth_Path,gt_name+"clean.wav"))
        write_list(ALL,join(cls.ALL_ROOT, "pair_random_filter_type.lst"))
        write_list(simulated, join(cls.ALL_ROOT, "simulated.lst"))
        write_list(clean, join(cls.ALL_ROOT, "target.lst"))

        # Speech Declipping
        Ground_Truth_Path = join(cls.SD_ROOT,"GroundTruth")
        for ratio in [0.1, 0.25, 0.5]:
            SD, source_dir = [], join(cls.SD_ROOT,str(ratio))
            for file in listdir(source_dir):
                if(file[-4:] != ".wav"): print(file);continue
                gt_name = re.findall("[sp]\d+_\d+", file)[0]
                SD.append(join(source_dir,file)+" "+join(Ground_Truth_Path,gt_name+"_mic1.wav"))
            write_list(SD,join(cls.SD_ROOT,"pair_%.2f.lst" % ratio))

        # Speech Enhancement
        Ground_Truth_Path = join(cls.SE_ROOT,"vd_test","clean_testset_wav")
        source_dir = join(cls.SE_ROOT, "vd_test", "noisy_testset_wav")
        SE = []
        for file in listdir(source_dir):
            if(file[-4:] != ".wav"): print(file);continue
            SE.append(join(source_dir,file)+" "+join(Ground_Truth_Path,file))
        write_list(SE,join(cls.SE_ROOT,"vd_test","pair.lst"))

        # Speech Super Resolution
        Ground_Truth_Path = join(cls.SSR_ROOT,"GroundTruth")
        for type in ["cheby1"]:
            for c in [1000,2000,4000,8000,12000]:
                SSR, source_dir = [], join(cls.SSR_ROOT,str(type),str(c))
                for file in listdir(source_dir):
                    if(file[-4:] != ".wav"): print(file);continue
                    gt_name = re.findall("[sp]\d+_\d+_mic1", file)[0]
                    SSR.append(join(source_dir,file)+" "+join(Ground_Truth_Path,gt_name+".wav"))
                write_list(SSR,join(cls.SSR_ROOT,str(type),"pair_"+type+"_"+str(c)+".lst"))

        # Speech Dereverberation
        Ground_Truth_Path = join(cls.SDR_ROOT,"GroundTruth")  # todo attension here is no reverb ( instead of with reverb )
        source_dir = join(cls.SDR_ROOT,"Reverb_Speech")
        SDR = []
        for file in listdir(source_dir):
            if(file[-4:] != ".wav"): print(file);continue
            gt_name = re.findall("[sp]\d+_\d+_mic1", file)[0]
            SDR.append(join(source_dir,file)+" "+join(Ground_Truth_Path,gt_name+".wav"))
        write_list(SDR,join(cls.SDR_ROOT, "pair_vctk_reverb.lst"),)

    @classmethod
    def checklst(cls):
        for each in [cls.SD_DATA,cls.SE_DATA,cls.SSR_DATA,cls.SDR_DATA,cls.ALL_DATA]:
            for key in each.keys():
                assert exists(each[key]['list']), each[key]['list'] + "not found"
        for files in glob.glob(join(cls.TEST_ROOT,"*/*.lst")) + glob.glob(join(cls.TEST_ROOT,"*/*/*.lst")):
            print(files)
            lst = read_list(join(cls.TEST_ROOT,files))
            lst = [x.split() for x in lst]
            for path in lst:
                if (len(path) == 2):
                    assert get_framesLength(path[0]) == get_framesLength(path[1]), str(path) + str((get_framesLength(path[0]), get_framesLength(path[1])))
                for item in path:
                    assert exists(item), item+" not found"
        print("Every thing is set!")

# Config.refresh_lists()
# Config.checklst()