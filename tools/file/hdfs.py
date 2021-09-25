
from contextlib import contextmanager
from pathlib import Path

from typing import IO, Any, AnyStr, List

import io
import os
import numpy as np
import pandas as pd
import torch
import yaml
import shutil
import subprocess

HDFS_BIN = 'hdfs'

@contextmanager
def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen("{} dfs -text {}".format(HDFS_BIN, hdfs_path),
                                shell=True,
                                stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen("{} dfs -appendToFile - {}".format(
            HDFS_BIN, hdfs_path),
                                shell=True,
                                stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen("{} dfs -put -f - {}".format(
            HDFS_BIN, hdfs_path),
                                shell=True,
                                stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hdfs_torch_load(filepath: str, **kwargs):
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    filepath += '/model_best.pth'
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict


def hdfs_torch_save(obj, filepath: str, **kwargs):
    if not filepath.startswith("hdfs://"):
        return torch.save(obj, filepath, **kwargs)
    with hopen(filepath, "wb") as writer:
        torch.save(obj, writer, **kwargs)


def hdfs_mkdir(hdfs_path: str):
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -mkdir -p {}".format(HDFS_BIN, hdfs_path),
                        shell=True)
    else:
        os.makedirs(hdfs_path, exist_ok=True)

def hdfs_remove_dir(hdfs_path: str):
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -rm -r {}".format(HDFS_BIN, hdfs_path),
                        shell=True)
    else:
        shutil.rmtree(hdfs_path)

def hdfs_put(local_path: str, hdfs_path: str):
    assert not local_path.startswith("hdfs://")
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -put -f {} {} ".format(HDFS_BIN, local_path,
                                                      hdfs_path),
                        shell=True)
    else:
        shutil.move(local_path, hdfs_path)

def hdfs_copy(local_path: str, hdfs_path: str):
    assert not local_path.startswith("hdfs://")
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -put -f {} {}".format(HDFS_BIN, local_path,
                                                      hdfs_path),
                        shell=True)
    else:
        shutil.copy(local_path, hdfs_path)

def hdfs_get(hdfs_path: str, local_path: str):
    assert not local_path.startswith("hdfs://")
    if hdfs_path.startswith("hdfs://"):
        subprocess.call("{} dfs -get {} {}".format(HDFS_BIN, hdfs_path,
                                                   local_path),
                        shell=True)
    else:
        shutil.move(hdfs_path, local_path)


def data_preparation(data, device):
    for i in range(len(data)):
        data[i] = np.frombuffer(data[i], dtype='float32').reshape((5, -1))
    return torch.from_numpy(np.array(data)).to(device)


class MetricTracker:

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys,
                                  columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
