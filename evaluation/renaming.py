import os
import re
import glob

dir = "/Volumes/Haohe_SSD/ICLR2022/MOS_Tests/Super_Resolution_24k"

def contain(num):
    return int(num) < 6

cnt = 0
for file in glob.glob(os.path.join(dir,"*/*")) + glob.glob(os.path.join(dir,"*/*/*"))+glob.glob(os.path.join(dir,"*/*/*/*")):
    if("flac" == file[-4:] or "wav" == file[-3:]):
        basename = os.path.basename(file)
        name = re.findall("[sp]\d+_\d+", file)[0]
        if(name[0] != "p" and name[0] != "s"):
            raise ValueError(file,name)
        type = basename.split(".")[-1]
        target_name = name+"."+type
        # print(file,os.path.join(os.path.dirname(file),target_name))
        os.rename(file,os.path.join(os.path.dirname(file),target_name))