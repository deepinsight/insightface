from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

def loginfo(log_dir):
    log_time_name = time.strftime('train_log_%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = os.path.join(log_dir, '%s.txt' % log_time_name)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)