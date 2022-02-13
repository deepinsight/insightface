import _init_paths
import os, sys, datetime, shutil
import os.path as osp
import random
import h5py 
import pickle as pkl
import numpy as np
import argparse
import subprocess
import warnings 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import lr_scheduler

from lib.dataloader.mpiinf import MPIINFDataset as mpiinf
from lib.dataloader.h36m import Human36MDataset as h36m
from lib.dataloader.surreal import SurrealDataset as surreal
from lib.models.model import get_pose_model, get_discriminator
from lib.core.config import  config, update_config, update_dir
from lib.utils.misc import save_pickle, create_logger, load_pickle
from lib.utils.utils import load_checkpoint, save_checkpoint, get_optimizer
from lib.utils.vis import plot_scalemid_dist, plot_scalemid_seq_dist
from tensorboardX import SummaryWriter

def set_cudnn(config): 
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

def parse_args():
    parser = argparse.ArgumentParser("Train the unsupervised human pose estimation network")    
    parser.add_argument('--cfg', help="Specify the path of the path of the config(*.yaml)", default='../cfg/default.yaml')
    parser.add_argument('--use_gt', action='store_true', help='Specify whether to use 2d gt / predictions as inputs')
    parser.add_argument('--model_dir', help='Specify the directory of pretrained model', default='')
    parser.add_argument('--data_dir', help="Specify the directory of data", default=config.DATA_DIR)
    parser.add_argument('--log_dir', help='Specify the directory of output', default=config.LOG_DIR)
    
    parser.add_argument('--dataset_name', help="Specify which dataset to use", choices=["h36m", "mpi"], default="h36m")
    parser.add_argument('--workers', help="Specify the number of workers for data loadering", default=config.NUM_WORKERS)
    parser.add_argument('--gpu', help="Specify the gpu to use for training", default='')
    parser.add_argument('--debug', action='store_true', help="Turn on the debug mode")
    parser.add_argument('--print_info', action='store_true', help="Whether to print detailed information in tqdm processing")
    parser.add_argument('--eval', action='store_true', help="Evaluate the model on the dataset(i.e. generate: joint_3d_pre)")
    parser.add_argument('--eval_suffix', default=None, help="Specify the suffix to save predictions on 3D in evaluation mode")
    parser.add_argument('--pretrain', default='', help="Whether to use pretrain model")
    parser.add_argument('--finetune_rotater', action='store_true', help="Load pretrained model and finetune rotater")
    parser.add_argument('--print_interval', type=int, default=50)
    args = parser.parse_args()
    if args.cfg:
        update_config(args.cfg)
    else:
        print("Using default config...")
    update_dir(args.model_dir, args.log_dir, args.data_dir, args.debug)
    return args

def reset_config(config, args):
    if not config.USE_GT: 
        config.USE_GT = True if args.use_gt else False
    if args.gpu:
        config.GPU = args.gpu
    config.NUM_WORKERS = args.workers
    config.DEBUG = args.debug
    if args.print_info: 
        config.PRINT_INFO = args.print_info
    if args.pretrain: 
        config.TRAIN.PRETRAIN_LIFTER = True 
        config.TRAIN.LIFTER_PRETRAIN_PATH = args.pretrain
    if args.finetune_rotater:
        assert config.TRAIN.PRETRAIN_LIFTER
        config.TRAIN.FINETUNE_ROTATER = args.finetune_rotater

def main():
    args = parse_args()
    reset_config(config, args)
    set_cudnn(config)
    seed = config.RANDOM_SEED
    np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed) ; torch.cuda.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU.strip()
    gpus = list(range(len(config.GPU.strip().split(','))))

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg)
    summary_writer = SummaryWriter(log_dir=tb_log_dir)

    this_dir = osp.dirname(__file__)
    # backup the source code and the yaml config
    if args.cfg:
        shutil.copy(args.cfg, osp.join(final_output_dir, osp.basename(args.cfg)))
    if not osp.exists(osp.join(final_output_dir, "lib")):
        shutil.copytree(osp.join(this_dir, "../lib/"), osp.join(final_output_dir, "lib"))
    for k, v in config.items():
        logger.info(f"{k}: {v}")

    # conditional import 
    if config.TRAIN.FINETUNE_ROTATER: 
        from lib.core.function3 import train, validate, evaluate
    elif config.TRAIN.USE_CYCLE: 
        from lib.core.function2 import train, validate, evaluate
    else: 
        from lib.core.function1 import train, validate, evaluate

    # build model
    logger.info('start building model.')
    if len(gpus) > 1: 
        pose_model = torch.nn.DataParallel(get_pose_model(config)).cuda(gpus[0])
        discriminator = torch.nn.DataParallel(get_discriminator(config)).cuda(gpus[0])
        temp_discriminator = torch.nn.DataParallel(get_discriminator(config)).cuda(gpus[0])
    else:
        pose_model = get_pose_model(config).cuda()
        discriminator = get_discriminator(config, is_temp=False).cuda()
        temp_discriminator = get_discriminator(config, is_temp=True).cuda()
    optimizer_g = get_optimizer(config, pose_model, is_dis=False)
    optimizer_d = get_optimizer(config, discriminator, is_dis=True)
    optimizer_d_temp = get_optimizer(config, temp_discriminator, is_dis=True, is_temp=True)
    step_size, gamma = config.TRAIN.SCHEDULER_STEP_SIZE, config.TRAIN.SCHEDULER_GAMMA
    scheduler_g = lr_scheduler.StepLR(optimizer_g, step_size=step_size, gamma=gamma)
    scheduler_d = lr_scheduler.StepLR(optimizer_d, step_size=step_size, gamma=gamma)
    scheduler_temp = lr_scheduler.StepLR(optimizer_d_temp, step_size=step_size, gamma=gamma)
    logger.info('finished building model.')
    # print out the model arch 
    if config.TRAIN.PRETRAIN_LIFTER: 
        print("Load pretrained lifter...")
        state_dict = torch.load(config.TRAIN.LIFTER_PRETRAIN_PATH)['pose_model_state_dict']
        # state_dict = {k[7:]:v for k, v in state_dict.items()}
        pose_model.load_state_dict(state_dict, strict=False)

    if config.DATA.DATASET_NAME == 'surreal': 
        loader_func = surreal
    else:
        loader_func = h36m if config.DATA.DATASET_NAME == "h36m" else mpiinf
    dataset_train = loader_func(config, is_train=True)
    dataset_test = loader_func(config, is_train=False)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=config.NUM_WORKERS
    )
    
    if args.eval:
        prefix = config.DATA.DATASET_NAME
        # for mode in ['train', 'valid']:
        for mode in ['valid']:
            is_train = True if mode == 'train' else False
            v3d_to_ours = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 0, 7, 9, 10] if prefix == "h36m" else np.arange(config.DATA.NUM_JOINTS)
            mpi2h36m = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 6, 7, 1, 14, 15, 16, 0]
            if prefix == 'surreal': 
                indices = np.arange(config.DATA.NUM_JOINTS)
            else:
                indices = v3d_to_ours if prefix == "h36m" else mpi2h36m 
            mode = "train" if is_train else "valid"
            read_name = f"../data/{prefix}_{mode}_pred3.h5"
            # read_name = f"../../unsupervised_mesh/data/h36m_{mode}_pred_3d_mesh.h5"
            save_name = f"../data/{prefix}_{mode}_pred_3d.h5"
            if args.eval_suffix is not None: 
                save_name = save_name[:-3] + "_" + args.eval_suffix + ".h5"

            # eval mode, load the pretrained model and generate the 3d prediction of all 3ds 
            if not config.TRAIN.PRETRAIN_LIFTER: 
                raise Warning("You are not using a pretrain model... may be you can specify --pretrain flag")
            dataloader = DataLoader(dataset_train if mode == "train" else dataset_test, batch_size=config.BATCH_SIZE, \
                shuffle=False, drop_last=False, pin_memory=True, num_workers=config.NUM_WORKERS)
            all_out_data = evaluate(dataloader, pose_model, config, is_train=(mode == "train"))
            p1_mpjpe, p2_mpjpe = all_out_data['p1_mpjpe'], all_out_data['p2_mpjpe']
            # read out imagenames
            print("Reading imagenames and joints 2d...")
            fin = h5py.File(read_name, "r")
            fout = h5py.File(save_name, "w")
            imagenames = fin['imagename'][:].copy()
            joints_2d_gt = np.array(fin['joint_2d_gt'])
            fout['imagename'] = imagenames
            fout['joint_2d_gt'] = joints_2d_gt[:, indices]
            fout['joint_3d_gt'] = all_out_data['joint_3d_gt']
            fout['joint_3d_pre'] = all_out_data['joint_3d_pre']
            possible_same_keys = ['shape', 'pose', 'original_joint_2d_gt', 'joint_2d_pre', 'seqlen']

            for key in possible_same_keys: 
                if key in fin.keys(): 
                    if 'joint' in key: 
                        fout[key] = np.array(fin[key])[:, indices]
                    else: 
                        fout[key] = np.array(fin[key])
            if 'seqname' in fin.keys(): 
                fout['seqname'] = fin['seqname'][:].copy()

            if 'auc' in all_out_data.keys(): 
                fout['auc'] = all_out_data['auc']
                fout['pckh5'] = all_out_data['pckh5']
                fout['auc_p2'] = all_out_data['auc_p2']
                fout['pckh5_p2'] = all_out_data['pckh5_p2']
            if 'scales' in all_out_data.keys():
                fout['scale_pre'] = all_out_data['scales'] 
            if 'scale_mids' in all_out_data.keys():
                fout['scale_mid_pre'] = all_out_data['scale_mids']

            fin.close()
            fout.close()
            print("Evaluation on the {} set finished. P1 Mpjpe: {:.3f}, P2 Mpjpe: {:.3f}, saved to {}".format(
                "training" if is_train else "test", p1_mpjpe, p2_mpjpe, save_name
            ))
            if prefix == "mpi":
                print("PCKh@0.5: {:.3f}, AUC: {:.3f}".format(all_out_data['pckh5'], all_out_data['auc']))
                print("P2: PCKh@0.5: {:.3f}, AUC: {:.3f}".format(all_out_data['pckh5_p2'], all_out_data['auc_p2']))
        # uncomment this if you need to plot images
        # print("Rendering sequences...")
        # subprocess.call(f'python render.py --seq_num 10 --in_filename ../data/{prefix}_valid_pred_3d.h5 --save_dir ../vis', shell=True)
        return 

    # preparation for visualization & perseq optimization(optional)
    if config.USE_GT: 
        # note that the gt here is not the gt above(config.USE_GT)
        train_path = f"../data/{config.DATA.DATASET_NAME}_train_scales.pkl"
        valid_path = f"../data/{config.DATA.DATASET_NAME}_valid_scales.pkl"
    else:
        train_path = f"../data/{config.DATA.DATASET_NAME}_train_scales_pre.pkl"
        valid_path = f"../data/{config.DATA.DATASET_NAME}_valid_scales_pre.pkl"

    train_scale_mids_gt = load_pickle(train_path)['scale_mid'] if osp.exists(train_path) else None
    valid_scale_mids_gt = load_pickle(valid_path)['scale_mid'] if osp.exists(valid_path) else None
    train_seqnames, valid_seqnames = dataset_train.get_seqnames(), dataset_test.get_seqnames()
    best_p1_mpjpe = best_p2_mpjpe = cur_p1_mpjpe = 10000.0
    best_auc_val = best_pckh5 = 0.0
    best_auc_val_p2 = best_pckh5_p2 = 0.0

    for epoch in range(config.TRAIN.NUM_EPOCHS):
        scheduler_d.step(); scheduler_g.step();scheduler_temp.step(); # scheduler_s.step()
        avg_d_loss, avg_g_loss, avg_t_loss, train_scale_mids_pre = train(train_loader, pose_model, discriminator, temp_discriminator, optimizer_g,
                optimizer_d, optimizer_d_temp, epoch, config, summary_writer=summary_writer, print_interval=config.PRINT_INTERVAL)
        logger.info("***** Epoch: {}, Avg G Loss: {:.3f}, Avg D Loss: {:.3f} Avg T Loss: {:.3f} *****".format(
            epoch, avg_g_loss, avg_d_loss, avg_t_loss))
        p1_mpjpe, p2_mpjpe, vis_image, valid_scale_mids_pre, extra_dict = validate(test_loader, pose_model, epoch, config)
        logger.info("Epoch: {}, P1 Mpjpe/Best P1: {:.3f}/{:.3f}, P2 Mpjpe/Best P2/Cur P1: {:.3f}/{:.3f}/{:.3f}".format(epoch, p1_mpjpe, best_p1_mpjpe, p2_mpjpe, best_p2_mpjpe, cur_p1_mpjpe))
        if p2_mpjpe < best_p2_mpjpe: 
            best_p2_mpjpe = p2_mpjpe
            cur_p1_mpjpe = p1_mpjpe
            is_best = True 
        else: 
            is_best = False

        if p1_mpjpe < best_p1_mpjpe: 
            best_p1_mpjpe = p1_mpjpe

        if extra_dict is not None: 
            auc_val, pckh5 = extra_dict['auc'], extra_dict['pckh5']
            auc_val_p2, pckh5_p2 = extra_dict['auc_p2'], extra_dict['pckh5_p2']
            if auc_val_p2 > best_auc_val_p2: 
                best_auc_val_p2 = auc_val_p2
                best_pckh5_p2 = pckh5_p2
                is_best = True 
            else: 
                is_best = False

            if auc_val > best_auc_val: 
                best_auc_val = auc_val 
                best_pckh5 = pckh5
            logger.info("PCKh@0.5(Best): {:.3f}({:.3f}), AUC value(Best): {:.3f}({:.3f})".format(pckh5, best_pckh5, auc_val, best_auc_val))
            logger.info("P2: PCKh@0.5(Best): {:.3f}({:.3f}), AUC value(Best): {:.3f}({:.3f})".format(pckh5_p2, best_pckh5_p2, auc_val_p2, best_auc_val_p2))

        save_checkpoint({
                "epoch": epoch, 
                "auc": best_auc_val, 
                "pckh5": best_pckh5, 
                "auc_p2": best_auc_val_p2, 
                "pckh5_p2": best_pckh5_p2, 
                "p1_mpjpe": p1_mpjpe, 
                "p2_mpjpe": p2_mpjpe, 
                "pose_model_state_dict": pose_model.state_dict(), 
                "discriminator_state_dict": discriminator.state_dict(), 
                "temp_discriminator_state_dict": temp_discriminator.state_dict(), 
                "optimizer_d": optimizer_d.state_dict(), 
                "optimizer_g": optimizer_g.state_dict(), 
                "optimizer_d_temp": optimizer_d_temp.state_dict()
            }, is_best, final_output_dir)
        summary_writer.add_scalar("p1_mpjpe_3d_test/epoch", p1_mpjpe, epoch)
        summary_writer.add_scalar("p2_mpjpe_3d_test/epoch", p2_mpjpe, epoch)
        summary_writer.add_image("test_joints/epoch", vis_image, epoch)
        if extra_dict is not None: 
            summary_writer.add_scalar("PCKh0.5/epoch", pckh5, epoch)
            summary_writer.add_scalar("AUC/epoch", auc_val, epoch)

        if train_scale_mids_gt is not None and train_scale_mids_pre is not None and len(train_scale_mids_pre) > 0: 
            num_seq = config.VIS.SCALE_MID_NUM_SEQ
            vis_image_scale_mid1 = plot_scalemid_dist(train_scale_mids_pre, train_scale_mids_gt.tolist())
            vis_image_scale_mid1 = torch.from_numpy(vis_image_scale_mid1).type(torch.float32).permute(2, 0, 1) / 255
            vis_image_scale_mid2 = plot_scalemid_seq_dist(train_scale_mids_pre, train_scale_mids_gt.tolist(), train_seqnames, num_seq=num_seq)
            vis_image_scale_mid2 = torch.from_numpy(vis_image_scale_mid2).type(torch.float32).permute(2, 0, 1) / 255
            summary_writer.add_image("train_scalemid_distribution/epoch", vis_image_scale_mid1, epoch)
            summary_writer.add_image("train_scalemid_seq_distribution/epoch", vis_image_scale_mid2, epoch)
        if valid_scale_mids_gt is not None and valid_scale_mids_pre is not None and len(valid_scale_mids_pre) > 0: 
            num_seq = config.VIS.SCALE_MID_NUM_SEQ
            vis_image_scale_mid1 = plot_scalemid_dist(valid_scale_mids_pre, valid_scale_mids_gt.tolist())
            vis_image_scale_mid1 = torch.from_numpy(vis_image_scale_mid1).type(torch.float32).permute(2, 0, 1) / 255
            vis_image_scale_mid2 = plot_scalemid_seq_dist(valid_scale_mids_pre, valid_scale_mids_gt.tolist(), valid_seqnames, num_seq=num_seq)
            vis_image_scale_mid2 = torch.from_numpy(vis_image_scale_mid2).type(torch.float32).permute(2, 0, 1) / 255
            summary_writer.add_image("valid_scalemid_distribution/epoch", vis_image_scale_mid1, epoch)
            summary_writer.add_image("valid_scalemid_seq_distribution/epoch", vis_image_scale_mid2, epoch)

    summary_writer.close()

if __name__ == '__main__':
    main()
