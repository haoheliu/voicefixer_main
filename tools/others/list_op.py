import numpy as np


def divide_list(li,thread_num):
    range_ = np.linspace(0,len(li),thread_num+1)
    res = []
    start,end = None,None
    for each in range(range_.shape[0]):
        if(each + 1 == range_.shape[0]):
            break
        start,end = int(range_[each]),int(range_[each+1])
        res.append(li[start:end])
    return res