import os, time
from config import *


# log file.
g_timestamp = time.time()
g_log_file_path = 'experiment/{}/log_train_nodelist_{}.txt'.format(opt.exp_name, g_timestamp)
if not os.path.exists(g_log_file_path):
    file = open(g_log_file_path, 'w')
    file.close()
file = open(g_log_file_path, 'w')
file.write('train.\n')
file.close()

