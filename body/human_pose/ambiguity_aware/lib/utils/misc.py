import os 
import os.path as osp
import h5py
import math
import time
import logging 
import pickle as pkl 
import torch.nn as nn
from pathlib import Path

def create_logger(cfg, cfg_name):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
    cfg_name = osp.basename(cfg_name).split('.')[0]
    final_output_dir = root_output_dir / cfg_name
    print(f'=> creating {final_output_dir}')
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(cfg_name, time_str)
    final_log_file = final_output_dir / log_file 
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / (cfg_name + "_" + time_str)
    print(f"=> creating {tensorboard_log_dir}")
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    return logger, str(final_output_dir), str(tensorboard_log_dir)

def init_weights(model):
    for m in model.modules(): 
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
    
def save_pickle(data, save_path):
    with open(save_path, "wb") as f:
        pkl.dump(data, f)
    print(f"=> saved to {save_path}")

def load_pickle(load_path):
    with open(load_path, "rb") as f:
        data = pkl.load(f)
    print(f"<= loaded from {load_path}")
    return data 

def process_dataset_for_video(path, is_mpi=False):
    # add some content for specified dataset(h5)
    f = h5py.File(path, "a")
    imagenames = [name.decode() for name in f['imagename'][:]]
    seqnames = ['/'.join(name.split('/')[:-1]) for name in imagenames]
    if is_mpi: 
        indices_in_seq_ref = [int(name.split('/')[-1].split('.')[0].split('_')[1]) for name in imagenames]
        # reset indices 
        indices_in_seq = []
        i = 0 
        last_seqname = None
        for index, seqname in zip(indices_in_seq_ref, seqnames): 
            if last_seqname is not None and seqname != last_seqname: 
                i = 0 
            last_seqname = seqname 
            indices_in_seq.append(i)
            i += 1
        # indices_in_seq = [i for i, index in enumerate(indices_in_seq)]
    else: 
        indices_in_seq = [int(name.split('/')[-1]) for name in imagenames]
    f['index_in_seq'] = indices_in_seq
    f['seqname'] = [name.encode() for name in seqnames]
    seq_lens = {}
    for seqname in seqnames: 
        if seqname not in seq_lens: 
            seq_lens[seqname] = 0 
        seq_lens[seqname] += 1

    f['seqlen'] = [seq_lens[seqname] for seqname in seqnames]
    f.close()

