import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

import os.path as op

class DATA:
    ROOT = os.path.dirname(git_root)
    TRAIL_NAME = os.environ['TRAIL_NAME']

    @classmethod
    def get_trainset(cls, name):
        print("Load train dataset "+name)
        if("vctk" == name):
            return DATA.VCTK_TRAIN
        elif("vd_noise" == name):
            return DATA.VD_NOISE
        else:
            raise ValueError("Error: unexpected training dataset "+ name)

    @classmethod
    def get_testset(cls, name):
        print("Load test dataset " + name)
        if("vctk" == name):
            return DATA.VCTK_TEST
        elif("gsr" == name):
            return DATA.ALL_GSR
        else:
            raise ValueError("Error: unexpected testing dataset "+ name)

    @classmethod
    def merge(cls, datasets: [dict]):
        res = {}
        dataset_names = set()
        for each in datasets:
            for type in each.keys():
                if(type not in res):
                    res[type] = {}
                for _set in each[type].keys():
                    dataset_names.add(_set)
                    res[type][_set] = each[type][_set]
        sample_rates = set([DATA.SAMPLE_RATE[each] for each in dataset_names])
        if(len(sample_rates) > 1):
            raise RuntimeError("Error: Encounter two or more kinds of sample rate in data "+str(sample_rates))
        return res

    @classmethod
    def Update(cls, data):
        for k in data.keys():
            for k2 in data[k].keys():
                data[k][k2] = op.join(DATA.ROOT, data[k][k2])

    SAMPLE_RATE = {
        "vctk":44100,
        "vd_noise": 44100,
        "gsr": 44100,
    }

    VD_NOISE = {
        "noise": {
            "vd_noise": "voicefixer_main/dataIndex/vd_noise/vd_noise.lst",
        }
    }

    VCTK_TRAIN = {
        "vocals":{
            "vctk": "voicefixer_main/dataIndex/vctk/train/speech.lst",
        },
    }

    # VD_TEST = {
    #     "vocals": {
    #         "vd_test": "arnold_workspace/dataIndex/vd_test/clean_testset_wav.lst",
    #     },
    #     "noisy": {
    #         "vd_test": "arnold_workspace/dataIndex/vd_test/noisy_testset_wav.lst",
    #     }
    # }

    VCTK_TEST = {
        "vocals":{
            "vctk": "voicefixer_main/dataIndex/vctk/test/speech.lst",
        }
    }

    ALL_GSR = {
        "vocals": {
            "gsr": "voicefixer_main/datasets/se/TestSets/ALL_GSR/target.lst",
        },
        "noisy": {
            "gsr": "voicefixer_main/datasets/se/TestSets/ALL_GSR/simulated.lst",
        }
    }

DATA.Update(DATA.VCTK_TRAIN)
DATA.Update(DATA.VCTK_TEST)
DATA.Update(DATA.VD_NOISE)
DATA.Update(DATA.ALL_GSR)

if __name__ == "__main__":
    print(DATA.merge([DATA.get_trainset(set) for set in ["vctk","vocal_wav_44k","vd_noise","dcase"]]))