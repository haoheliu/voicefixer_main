import os
import re
import numpy as np
import matplotlib.pyplot as plt

dir = "train"
all_files = os.listdir(dir)

def distance(pos1, pos2):
    return np.sum(np.abs(pos1-pos2)**2) ** 0.5

dist = []
exception = 0
threshold = 5

for i, file in enumerate(all_files):
    if("temp" in file): continue
    try:
        rt60 = float(re.findall("rt60_\d+\.\d+",file)[0][5:])
        room = np.array([float(x) for x in re.findall("room_[-+]?\d+\.\d+_[-+]?\d+\.\d+_[-+]?\d+\.\d+", file)[0][5:].split("_")])
        mic = np.array([float(x) for x in re.findall("mic_[-+]?\d+\.\d+_[-+]?\d+\.\d+_[-+]?\d+\.\d+", file)[0][4:].split("_")])
        source = np.array([float(x) for x in re.findall("source_[-+]?\d+\.\d+_[-+]?\d+\.\d+_[-+]?\d+\.\d+", file)[0][7:].split("_")])
        dist.append(distance(mic,source))
        if(dist[-1] > 5):
            cmd = "rm "+os.path.join(dir,file)
            os.system(cmd)
            print(cmd)
    except Exception as e:
        print(file)
        print(e)
        break

print(exception)
plt.hist(dist)
plt.show()

