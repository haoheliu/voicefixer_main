import sys

sys.path.append("/Users/admin/Documents/projects/arnold_workspace/src")
sys.path.append("/opt/tiger/lhh_arnold_base/arnold_workspace/src")

import concurrent.futures
from multiprocessing import set_start_method
from evaluation import Config
from progressbar import *
import time
from evaluation import AudioMetrics
from math import ceil
# from general_speech_restoration.all_stft_hard_only.unet.handler import refresh_model
import logging
import random
from tools.file.hdfs import *
from tools.file.io import *
from pynvml import nvmlInit
import glob
from evaluation.util import *

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


def aggregate_score(
        save_dir: str,
        testsets: list,
        limit_number=None):
    """
        Gather score of the inference results.
        This process usually take longer time than inferencing on GPU. So it's better to have more sub-processes.
        Thus the evaluation process and inferencing process are writed separately.
    """
    result = {}
    score_part_1, score_part_2 = None, None
    for i, testset in enumerate(testsets):
        print("No:", i, len(testsets))
        has_target = True
        result[testset] = {}
        res_dir = os.path.join(save_dir, testset)
        os.makedirs(res_dir, exist_ok=True)
        meta = Config.get_meta(testset)
        rate, lst = meta['rate'], read_list(meta['list'])  # todo here I limit the numbers
        if (limit_number is not None): lst = lst[:limit_number]
        judger = AudioMetrics(rate=rate)
        # print("Generating Output",end=)
        widgets = [
            "Calculate Score on: ",
            testset,
            ' [', Timer(), '] ',
            Bar(),
            testset,
            ' (', ETA(), ') ',
        ]
        pbar = ProgressBar(widgets=widgets).start()
        for i, pair in enumerate(lst):
            line = pair.split(" ")
            if (len(line) == 1):
                source, target, has_target = line[0], None, False
            else:
                source, target = line[0], line[1]
            fname = os.path.basename(source)
            wav_file_dir = os.path.join(res_dir, fname)

            score_part_1 = load_json(wav_file_dir[:-4] + ".json")

            if (has_target):
                try:
                    score_part_2 = judger.evaluation(est=wav_file_dir, target=target)
                    score_part_2.update(score_part_1)

                    write_json(score_part_2, wav_file_dir[:-4] + ".json")
                except Exception as e:
                    print(wav_file_dir, target)
                    logging.exception(e)

                result[testset][os.path.basename(target)] = score_part_2
            pbar.update(int((i / (len(lst) + 1)) * 100))
        pbar.finish()
        if (has_target):
            df = pd.DataFrame.from_dict(result[testset]).transpose()
            df.loc['mean'] = df.mean()
            print(testset)
            print(df.mean())
            df.to_csv(os.path.join(res_dir, testset + ".csv"))
            write_json(df.mean().to_dict(), os.path.join(res_dir, "result.json"))


def inference(handler: callable,
              ckpt: str,
              save_dir: str,
              testsets: list,
              device: torch.device,
              limit_number=None):
    """
    :param handler: function,
    :param save_dir: place to save results
    :param testsets: list,
    :param device: torch.decive,
    :param limit_number: if None, evaluate on full dataset
    :return:
    """
    for i, testset in enumerate(testsets):
        print("No:", i, len(testsets))
        res_dir = os.path.join(save_dir, testset)
        os.makedirs(res_dir, exist_ok=True)
        meta = Config.get_meta(testset)
        rate, lst = meta['rate'], read_list(meta['list'])  # todo here I limit the numbers
        if (limit_number is not None): lst = lst[:limit_number]
        widgets = [
            "Inferencing: ",
            testset,
            ' [', Timer(), '] ',
            Bar(),
            testset,
            ' (', ETA(), ') ',
        ]
        pbar = ProgressBar(widgets=widgets).start()
        for j, pair in enumerate(lst):
            line = pair.split(" ")
            if (len(line) == 1):
                source, target = line[0], None
            else:
                source, target = line[0], line[1]
            fname = os.path.basename(source)
            wav_file_dir = os.path.join(res_dir, fname)
            if (j == 0 and i == 0):
                score_part_1 = handler(input=source, output=wav_file_dir, target=target, device=device, ckpt=ckpt,
                                       needrefresh=True, meta=meta)
            else:
                score_part_1 = handler(input=source, output=wav_file_dir, target=target, device=device, ckpt=ckpt,
                                       needrefresh=False, meta=meta)
            write_json(score_part_1, wav_file_dir[:-4] + ".json")
            pbar.update(int((j / (len(lst)+1)) * 100))
        pbar.finish()

def split_list_average_n(origin_list, n):
    for i in range(0, len(origin_list), n):
        yield origin_list[i:i + n]


def evaluation(
        output_path: str,
        handler: callable,
        ckpt: str,
        description: str,
        limit_testset_to=None,
        limit_phrase_number=None):
    """
    Perform evaluation on multiple testsets concurrently
    :param output_path: The path to store your evaluation results, default Config.EVAL_RESULT
    :param handler: The function you provide to do processing
    :param ckpt: Checkpoint path, will be passed to your handler function.
    :param description: Any description you wanna add to this evaluation
    :param limit_testset_to: list, If you not intend to evaluate on all dataset, you can set this parameter.
    :param limit_phrase_number: If you do not want to evaluate on full dataset, you can set this parameter.
    :return:
    """
    set_start_method('spawn')
    id = int(random.random() * 1000)
    id = time.strftime("%Y-%m-%d", time.localtime()) + "_" + description + "_" + str(id)
    output_path = os.path.join(output_path, id)
    # Inference
    if(torch.cuda.is_available()): nvmlInit()
    inference(handler,
              ckpt,
              output_path,
              limit_testset_to,
              torch.device("cuda:0") if (torch.cuda.is_available()) else torch.device("cpu"),
              limit_phrase_number)

    # os.system("python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy_all.py &")
    # os.system("python3 /opt/tiger/lhh_arnold_base/arnold_workspace/env/occupy_all.py &")

    # Evaluation
    MAX_PROC = 12
    test_set_per_process = ceil(len(limit_testset_to) / MAX_PROC)
    partitions = [each for each in split_list_average_n(limit_testset_to, test_set_per_process)]
    print("Partition numbers", partitions)
    # for partition in [limit_testset_to]:
    params = []
    for i in range(len(partitions)):
        params.append((
            output_path,
            partitions[i],
            limit_phrase_number
        ))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(partitions)) as excutor:
            for each in params:
                excutor.submit(aggregate_score, *each)
    except Exception as e:
        logging.exception(e)

    gather_score(output_path, limit_testset_to)

    # print("Do you want to back up this result to hdfs? ")
    # while (True):
    #     in_content = input("Type in yes/no/part: ")
    #     if (in_content == "yes"):
    #         remote_path = os.path.join(Config.HDFS_RESULT_ROOT, id)
    #         convert_wav_to_flac(output_path)
    #         hdfs_mkdir(remote_path)
    #
    #         for file in glob.glob(os.path.join(output_path, "*.csv")) + \
    #                     glob.glob(os.path.join(output_path, "*/*.csv")) + \
    #                     glob.glob(os.path.join(output_path, "*/*/*.csv")):
    #             print(file)
    #             hdfs_put(file, hdfs_path=remote_path)
    #
    #         os.system("tar -zcf " + output_path + ".tar " + output_path)
    #         print("Putting", output_path + ".tar", "to", remote_path)
    #         hdfs_put(output_path + ".tar", hdfs_path=remote_path)
    #
    #         print("Done")
    #         break
    #     elif (in_content == "no"):
    #         break
    #     else:
    #         continue
    return output_path


def gather_score(output_path: str, testset=Config.SETTING1):
    final_result = {}
    for each in testset:
        if (os.path.exists(os.path.join(output_path, each, "result.json"))):
            final_result[each] = load_json(os.path.join(output_path, each, "result.json"))
            df = pd.DataFrame.from_dict(final_result).transpose()
            df.to_csv(os.path.join(output_path, "result.csv"))

def handler_copy(input, output, target,ckpt, device, needrefresh=False,meta={}) -> dict:
    """
    :param input: Input path of a .wav file
    :param output: Save path of your result. (.wav)
    :param device: Torch.device
    :return:
    """
    os.system("cp "+input+" "+output)
    return {}


if __name__ == '__main__':
    # Config.init()
    evaluation(output_path=Config.EVAL_RESULT,
               handler=handler_copy,
               ckpt="",
               description="sspade",
               limit_testset_to=Config.get_testsets("declipping"),
               # limit_testset_to=Config.get_all_testsets(),
               limit_phrase_number=None)
