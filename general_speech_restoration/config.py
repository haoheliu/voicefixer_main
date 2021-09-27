import git
import sys
import os

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
r = os.path.dirname(git_root)

import os.path as op
import torch


class Config:
    ROOT = r
    TRAIL_NAME = os.environ['TRAIL_NAME']

    # DATA
    """
    dict{<type-of-source>:<a file containing all_mel_e2e paths to wav files>}

    """
    train_data = {
        "vocals": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/train/vocals.lst",
            "vctk": "arnold_workspace/dataIndex/vctk/train/speech.lst",
        },
        "bass": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/train/bass.lst"
        },
        "drums": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/train/drums.lst"
        },
        "other": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/train/other.lst"
        },
        "acc": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/train/acc.lst"
        },
    }

    test_data = {
        "vocals": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/test/vocals.lst",
            "vctk": "arnold_workspace/dataIndex/vctk/test/speech.lst",
        },
        "bass": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/test/bass.lst"
        },
        "drums": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/test/drums.lst"
        },
        "other": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/test/other.lst"
        },
        "acc": {
            "musdb18hq": "arnold_workspace/dataIndex/musdb18hq/test/acc.lst"
        },
    }

    for k in train_data.keys():
        for k2 in train_data[k].keys():
            train_data[k][k2] = op.join(ROOT, train_data[k][k2])

    for k in test_data.keys():
        for k2 in test_data[k].keys():
            test_data[k][k2] = op.join(ROOT, test_data[k][k2])

    # AUGMENTATION
    aug_sources = ["vocals"]
    aug_effects = ["low_pass", "clip", "reverb_rir"]

    aug_conf = {
        "rir_root": os.path.join(ROOT, "voicefixer_main/datasets/se/RIR_44k/train"),
        # clean
        'tempo': {
            'prob': [0.0, 0.0],
            'speed_up_range': [1.1, 1.6],
            'speed_down_range': [0.7, 0.95]
        },
        'speed': {
            'prob': [0.0, 0.0],
            'speed_up_range': [1.1, 1.6],
            'speed_down_range': [0.7, 0.95]
        },
        'fade': {
            'prob': [0.1],
            'fade_in_portion': [0.1, 0.3],
            'fade_out_portion': [0.1, 0.3]
        },
        'pitch': {
            'prob': [0.0, 0.0],
            'pitch_up_range': [100, 350],
            'pitch_down_range': [-350, -100]
        },
        'treble': {
            'prob': [0.05],
            'level': [3, 20]
        },
        'bass': {
            'prob': [0.15],
            'level': [3, 35]
        },
        'tremolo': {
            'prob': [0.0],
            'level': [5, 50]
        },
        'reverb_freeverb': {
            'prob': [0.20],
            'reverb_level': [0, 50],  # 0-100
            'dumping_factor': [0, 100],  # 0-100
            'room_size': [0, 100],  # 0-100
        },
        'reverb_rir': {
            'prob': [0.30],
            'rir_file_name': None
        },
        'low_pass': {
            'prob': [0.25],
            'low_pass_range': [4000, 22000],  # frequency, not sample rate
        },
        'high_pass': {
            'prob': [0.25],
            'high_pass_range': [500, 2000]
        },
        'clip': {
            'prob': [0.20],  # todo
            'louder_time': [1.0, 11.0]
        },
        'reverse': {
            'prob': [0.0]
        },
        'time_dropout': {
            'prob': [0.0],
            'max_segment': 0.1,
            'drop_range': [0.0, 1.0]
        },
        'empty_c': {
            'prob': [0.0]
        },
        # noise
        'empty_n': {
            'prob': [0]
        },
        'beep': {
            'prob': [0]
        },
        'quant': {
            'prob': [0.4],
            'bins': [4, 10]
        },
    }

    mel_weight_44k_128 = torch.tensor([19.40951426, 19.94047336, 20.4859038, 21.04629067,
                                       21.62194148, 22.21335214, 22.8210215, 23.44529231,
                                       24.08660962, 24.74541882, 25.42234287, 26.11770576,
                                       26.83212784, 27.56615283, 28.32007747, 29.0947679,
                                       29.89060111, 30.70832636, 31.54828121, 32.41121487,
                                       33.29780773, 34.20865341, 35.14437675, 36.1056621,
                                       37.09332763, 38.10795802, 39.15039691, 40.22119881,
                                       41.32154931, 42.45172373, 43.61293329, 44.80609379,
                                       46.031602, 47.29070223, 48.58427549, 49.91327905,
                                       51.27863232, 52.68119708, 54.1222372, 55.60274206,
                                       57.12364703, 58.68617876, 60.29148652, 61.94081306,
                                       63.63501986, 65.37562658, 67.16408954, 69.00109084,
                                       70.88850318, 72.82736101, 74.81985537, 76.86654792,
                                       78.96885475, 81.12900906, 83.34840929, 85.62810662,
                                       87.97005418, 90.37689804, 92.84887686, 95.38872881,
                                       97.99777002, 100.67862715, 103.43232942, 106.26140638,
                                       109.16827015, 112.15470471, 115.22184756, 118.37439245,
                                       121.6122689, 124.93877158, 128.35661454, 131.86761321,
                                       135.47417938, 139.18059494, 142.98713744, 146.89771854,
                                       150.91684347, 155.0446638, 159.28614648, 163.64270198,
                                       168.12035831, 172.71749158, 177.44220154, 182.29556933,
                                       187.28286676, 192.40502126, 197.6682721, 203.07516896,
                                       208.63088733, 214.33770931, 220.19910108, 226.22363072,
                                       232.41087124, 238.76803591, 245.30079083, 252.01064464,
                                       258.90261676, 265.98474, 273.26010248, 280.73496362,
                                       288.41440094, 296.30489752, 304.41180337, 312.7377183,
                                       321.28877878, 330.07870237, 339.10812951, 348.38276173,
                                       357.91393924, 367.70513992, 377.76413924, 388.09467408,
                                       398.70920178, 409.61813793, 420.81980127, 432.33215467,
                                       444.16083117, 456.30919947, 468.78589276, 481.61325588,
                                       494.78824596, 508.31969844, 522.2238331, 536.51163441,
                                       551.18859414, 566.26142988, 581.75006061, 597.66210737]) / 19.40951426
    mel_weight_44k_128 = mel_weight_44k_128[None, None, None, ...]
