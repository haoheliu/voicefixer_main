import sys
sys.path.append("/Users/admin/Documents/projects/arnold_workspace/src")
sys.path.append("/opt/tiger/lhh_arnold_base/arnold_workspace/src")


import logging as lg

def setup_logging(save_dir):
    lg.basicConfig(level=lg.INFO,
                   format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S',
                   filename=save_dir,
                   filemode='w')