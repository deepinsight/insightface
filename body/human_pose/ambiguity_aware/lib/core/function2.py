import time
import pickle as pkl
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm 
from itertools import permutations
from lib.core.loss import *
from lib.utils.vis import vis_joints
from lib.utils.utils import p_mpjpe, rotate, calc_auc, calc_auc_aligned

def train(train_loader, pose_model, discriminator, temp_discriminator, \
    optimizer_g, optimizer_d, optimizer_d_temp, epoch, config, summary_writer=None, print_interval=10, scale_mid_info=None, seqnames=None):  
    # parse the config
    print_info = config.PRINT_INFO 
    num_frames = config.DATA.NUM_FRAMES
    num_critics = config.TRAIN.NUM_CRITICS 
    num_critics_temp = config.TRAIN.NUM_CRITICS_TEMP 
    subnet_critics = config.TRAIN.SUBNET_CRITICS
    mainnet_critics = config.TRAIN.MAINNET_CRITICS
    loss_weights = config.TRAIN.LOSS_WEIGHTS
    scale_loss_weights = config.TRAIN.SCALE_LOSS_WEIGHTS
    rotate_loss_weights = config.TRAIN.ROTATE_LOSS_WEIGHTS
    use_scaler = config.TRAIN.USE_SCALER
    use_dis = config.TRAIN.USE_DIS; use_temp = config.TRAIN.USE_TEMP; use_lstm = config.TRAIN.USE_LSTM
    use_new_temp = config.TRAIN.USE_NEW_TEMP
    learn_symmetry = config.TRAIN.LEARN_SYMMETRY
    multi_new_temp = config.TRAIN.MULTI_NEW_TEMP
    cat_current = config.TRAIN.CAT_CURRENT
    scale_mid_mean, scale_mid_std = config.DATA.SCALE_MID_MEAN, config.DATA.SCALE_MID_STD
    scaler_mean_only = config.TRAIN.SCALER_MEAN_ONLY
    fix_traj = config.FIX.FIX_TRAJ; fix_traj_by_rot = config.FIX.FIX_TRAJ_BY_ROT; fix_bone_loss = config.FIX.FIX_BONE_LOSS
    w1, w2, w3, w4 = loss_weights
    if len(scale_loss_weights) == 2:
        w_s_neighbour = 1.0
        w_s, w_s_bone = scale_loss_weights
    else: 
        w_s, w_s_bone, w_s_neighbour = scale_loss_weights
    w_rot_reg = rotate_loss_weights[0]
    scaler_criterion = torch.nn.KLDivLoss()
    assert subnet_critics >= 1 or (not use_scaler)
    subnet_string_format = "[Phase: training subnet] | [Kl Loss: {:.6f}] | [Bone Loss: {:.3f}] | [Weighted Loss: {:.3f}]"
    mainnet_string_format = "[Phase: training mainnet] | [Est Loss: {:.4f}] | [3d Loss: {:.4f}] | [2d Loss: {:.4f}] | [D_G loss: {:.4f}] | [D_D Loss: {:.4f}] | [T Loss: {:.4f}]"
    mainnet_string = mainnet_string_format.format(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    subnet_string = subnet_string_format.format(0.0, 0.0, 0.0, 0.0)
    factor = 0.6451607 if config.DATA.DATASET_NAME == "h36m" else 0.7286965902
    is_h36m = config.DATA.DATASET_NAME == 'h36m'
    use_2d_gt_supervision = config.DATA.USE_2D_GT_SUPERVISION

    pose_model.train()
    discriminator.train(); temp_discriminator.train()
    t = tqdm(train_loader, desc=f"Epoch: {epoch}, training...")
    all_scale_mids = []
    train_cnt = subnet_cnt = mainnet_cnt = 0
    running_dis_loss, running_pose_loss, running_temp_loss = 0.0, 0.0, 0.0
    running_pose_subnet_loss, running_pose_mainnet_loss = 0.0, 0.0
    # here neighbours is expected to be 2d, N * F * J * 2
    for kp2ds, kp3ds, rot, diff1, diff2, scales_gt in t:
        bs = kp2ds.size(0)
        kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
        rot = rot.cuda()
        diff1 = diff1.cuda(); diff2 = diff2.cuda()
        scales_gt = scales_gt.cuda()
        l_t = torch.tensor(0.0).type_as(kp2ds)

        rem = train_cnt % (subnet_critics + mainnet_critics)
        flag = rem > subnet_critics 
        for param in pose_model.lifter_before.parameters():
            param.requires_grad = flag
        for param in pose_model.lifter_after.parameters():
            param.requires_grad = flag
        try: 
            for param in pose_model.scaler.parameters():
                param.requires_grad = not flag
            for param in pose_model.rotater.parameters():
                param.requires_grad = flag
        except: 
            pass

        if rem < subnet_critics: 
            optimizer_g.zero_grad()
            (first_half_feature, second_half_feature), est_3d, tran_3d, proj_2d, \
                recn_3d, recovered_3d,  recn_2d, scale_mids, scales = pose_model(kp2ds, is_train=True, rot=rot)
            (first_half_feature, second_half_feature), est_3d_diff, tran_3d_diff, proj_2d_diff, \
                recn_3d_diff, recovered_3d_diff, recn_2d_diff, scale_mids_diff, scales_diff = pose_model(diff2, is_train=True, rot=rot)
            # for later use of visualization
            all_scale_mids.extend(scale_mids.detach().cpu().numpy().tolist())
            # for later calculation of KL loss 
            scale_mids = torch.cat([scale_mids, scale_mids_diff])

            all_scales = []

            assert scale_mids is not None
            # add kl loss between scale_mids and expected distribution(mean: 0.707, std: 0.025)
            scales_dist = (scales_gt - scales).abs().mean()
            loss_s_bone = loss_bone(est_3d, est_3d_diff)
            if learn_symmetry:
                mean, std = scale_mids.mean(), scale_mids.std()
                loss_s = kl_criterion(mean, std, scale_mid_mean, scale_mid_std, mean_only=scaler_mean_only)
                loss_pose_subnet = w_s * loss_s + w_s_bone * loss_s_bone
            else:
                loss_s = torch.tensor(0.0).type_as(kp2ds)
                loss_pose_subnet = w_s_bone * loss_s_bone
            loss_pose_subnet.backward()
            optimizer_g.step()
            subnet_cnt += 1
            if summary_writer is not None: 
                iters = len(train_loader) * epoch * subnet_critics // (subnet_critics + mainnet_critics) + subnet_cnt
                summary_writer.add_scalar("scale_pre_gt_dist/iters", scales_dist.item(), iters)
                summary_writer.add_scalar("loss_pose_subnet", loss_pose_subnet.item(), iters)
                summary_writer.add_scalar("loss_s_bone", loss_s_bone.item(), iters)
                summary_writer.add_scalar("loss_kl", loss_s.item(), iters)
        else:
            optimizer_g.zero_grad()
            (first_half_feature, second_half_feature), est_3d, tran_3d, proj_2d, \
                recn_3d, recovered_3d, recn_2d, scale_mids, scales = pose_model(kp2ds, is_train=True, rot=rot)
            all_scale_mids.extend(scale_mids.detach().cpu().numpy().tolist())

            # minus the root 
            if not config.TRAIN.GENERIC_BASELINE:
                kp3ds, est_3d_centered = kp3ds - kp3ds[:, 13:14], (est_3d - est_3d[:, 13:14]) * factor
            else:
                kp3ds, est_3d_centered = kp3ds - kp3ds[:, 13:14], (est_3d - est_3d[:, 13:14])
            est_gt_loss_3d = loss_3d(est_3d_centered, kp3ds)
            est_dist_2d = loss_2d(kp2ds, diff1)
            l3d = loss_3d(tran_3d, recn_3d)
            l2d = loss_2d(recn_2d, kp2ds)
            loss_pose = w1 * l3d + w2 * l2d

            if use_dis: 
                g_fake_logits = discriminator(proj_2d)
                l_g = (g_fake_logits - 1).pow(2).mean()
                loss_pose += w3 * l_g

            if use_temp: 
                (first_half_feature, second_half_feature), est_3d_diff, tran_3d_diff, proj_2d_diff, \
                    recn_3d_diff, recovered_3d_diff, recn_2d_diff, _, _ = pose_model(diff1, is_train=True, rot=rot)
                fake_diff = (proj_2d - proj_2d_diff)
                if not cat_current: 
                    fake_diff_cat = torch.cat([proj_2d_diff, fake_diff], -1)
                else: 
                    fake_diff_cat = torch.cat([proj_2d, fake_diff], -1)
                g_diff_fake_logits = temp_discriminator(fake_diff_cat)
                l_t = (g_diff_fake_logits - 1).pow(2).mean()
                loss_pose += w4 * l_t

            if use_new_temp:
                (first_half_feature, second_half_feature), est_3d_diff, tran_3d_diff, proj_2d_diff, \
                    recn_3d_diff, recovered_3d_diff, recn_2d_diff, _, _ = pose_model(diff1, is_train=True, rot=rot)
                if fix_traj:
                    if fix_traj_by_rot:
                        l_t = loss_3d(est_3d - est_3d_diff, recovered_3d - recovered_3d_diff)
                    else:
                        l_t = loss_3d((est_3d - est_3d_diff).norm(dim=-1), (recn_3d - recn_3d_diff).norm(dim=-1))
                else:
                    l_t = loss_3d(est_3d - est_3d_diff, recn_3d - recn_3d_diff)

                loss_pose += w4 * l_t

            loss_pose.backward()
            optimizer_g.step()
            
            # train the discriminator
            # if use_dis and train_cnt % num_critics == 0:
            if use_dis and mainnet_cnt % num_critics == 0:
                optimizer_d.zero_grad()
                _, _, _, proj_2d, _, _, _, _, _ = pose_model(kp2ds, is_train=True, rot=rot)
                d_real_logits = discriminator(kp2ds)
                d_fake_logits = discriminator(proj_2d.detach())
                loss_dis1 = (d_real_logits - 1).pow(2).mean() + d_fake_logits.pow(2).mean()
                loss_dis1.backward()
                optimizer_d.step()

            # train the temporal discriminator
            # if use_temp and train_cnt % num_critics_temp == 0:
            if use_temp and mainnet_cnt % num_critics_temp == 0:
                optimizer_d_temp.zero_grad()
                _, _, _, proj_2d, _, _, _, _, _ = pose_model(kp2ds, is_train=True, rot=rot)
                _, _, _, proj_2d_diff, _, _, _, _, _ = pose_model(diff1, is_train=True, rot=rot)
                real_diff = (kp2ds - diff1)
                fake_diff = (proj_2d - proj_2d_diff)
                if not cat_current: 
                    real_diff_cat = torch.cat([diff1, real_diff], -1)
                    fake_diff_cat = torch.cat([proj_2d_diff, fake_diff], -1)
                else: 
                    real_diff_cat = torch.cat([kp2ds, real_diff], -1)
                    fake_diff_cat = torch.cat([proj_2d, fake_diff], -1)
                d_diff_real_logits = temp_discriminator(real_diff_cat)
                d_diff_fake_logits = temp_discriminator(fake_diff_cat.detach())
                loss_dis2 = (d_diff_real_logits - 1).pow(2).mean() + d_diff_fake_logits.pow(2).mean()
                loss_dis2.backward()
                optimizer_d_temp.step()

            if summary_writer is not None: 
                iters = epoch * len(train_loader) + train_cnt
                summary_writer.add_scalar("pose_loss/iters", loss_pose.item(), iters)
                summary_writer.add_scalar("3d_loss/iters", l3d.item(), iters)
                summary_writer.add_scalar("2d_loss/iters", l2d.item(), iters)
                summary_writer.add_scalar("estimated_3d_loss/iters", est_gt_loss_3d.item(), iters)
                if use_dis: 
                    summary_writer.add_scalar("dis_d_loss/iters", loss_dis1.item(), iters)
                    summary_writer.add_scalar("dis_g_loss/iters", l_g.item(), iters)
                if use_temp: 
                    summary_writer.add_scalar("temp_d_loss/iters", loss_dis2.item(), iters)
                    summary_writer.add_scalar("temp_g_loss/iters", l_t.item(), iters)
                if use_new_temp: 
                    summary_writer.add_scalar("temp_loss/iters", l_t.item(), iters)

            mainnet_cnt += 1
            running_pose_loss += loss_pose.item() * kp2ds.size(0)
            if use_dis: 
                running_dis_loss += loss_dis1.item() * kp2ds.size(0)
            if use_temp: 
                running_temp_loss += loss_dis2.item() * kp2ds.size(0)

        if print_info:
            if train_cnt % (print_interval * (subnet_critics + mainnet_critics)) in [0, subnet_critics]:
                if rem < subnet_critics: 
                    subnet_string = subnet_string_format.format(loss_s.item(), loss_s_bone.item(), loss_pose_subnet.item())
                    tqdm.write(subnet_string)
                else:
                    mainnet_string = mainnet_string_format.format(est_gt_loss_3d.item(), l3d.item(), l2d.item(), l_g.item(), loss_dis1.item(), l_t.item())
                    tqdm.write(mainnet_string)
        train_cnt += 1

    # finally do some visualization
    if summary_writer is not None: 
        with torch.no_grad():
            kp2ds, kp3ds, rot, _, _, _ = next(iter(train_loader))
            kp2ds = kp2ds.cuda()
            kp3ds = kp3ds.cuda()
            rot = rot.cuda()
            _, est_3d, tran_3d, proj_2d, recn_3d, recovered_3d, recn_2d, _, _ = pose_model(kp2ds, is_train=True, rot=rot) 
            joints = {"input 2d": kp2ds.cpu().numpy(), "recn_2d": recn_2d.cpu().numpy(), "gt 3d": kp3ds.cpu().numpy(), "est 3d": est_3d.cpu().numpy(), "tran 3d": tran_3d.cpu().numpy(),  "recn 3d": recn_3d.cpu().numpy()}
            vis_image = vis_joints(joints)
            vis_image = torch.from_numpy(vis_image).type(torch.float32).permute(2, 0, 1) / 255
            summary_writer.add_image("train_joints/epoch", vis_image, epoch)
    
    return running_dis_loss / len(train_loader.dataset), running_pose_loss / len(train_loader.dataset), running_temp_loss / len(train_loader.dataset), all_scale_mids


def validate(val_loader, pose_model, epoch, config):
    # parse the config 
    num_frames = config.DATA.NUM_FRAMES 
    use_same_norm_3d = config.DATA.USE_SAME_NORM_3D
    prefix = "mpi" if config.DATA.DATASET_NAME == "mpi" else "h36m"
    pose_model.eval()
    # for 10 w
    # factor = 0.680019 if prefix == "h36m" else 0.7577316 
    factor = 0.680019 if prefix == "h36m" else 0.7577316
    if not use_same_norm_3d:
        with open(f"../data/{prefix}_test_factor_3d.pkl", "rb") as f:
            factor = torch.from_numpy(pkl.load(f)).cuda()
    valid_cnt = 0 
    p1_mpjpe, p2_mpjpe = 0.0, 0.0
    all_scale_mids_pre, all_scale_mids_gt = [], []
    pred_3ds, gt_3ds = [], []
    with torch.no_grad():
        t = tqdm(val_loader, desc=f"Epoch: {epoch}, testing...")
        for kp2ds, kp3ds, _, _, _, _ in t:
            if valid_cnt == 0:
                bs = kp2ds.size(0)
            kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
            pred_3d, _, scale_mids = pose_model(kp2ds, is_train=False, rot=None)
            root3d, root3d_pred = kp3ds[:, 13:14].clone(), pred_3d[:, 13:14].clone()
            kp3ds, pred_3d = kp3ds - root3d, pred_3d - root3d_pred
            # rescale back to original scale from 1 unit
            if not config.TRAIN.GENERIC_BASELINE:
                if not use_same_norm_3d:
                    pred_3d = pred_3d * factor[int(bs * valid_cnt):int(bs * (valid_cnt+1))]
                else:
                    pred_3d = pred_3d * factor 
            all_scale_mids_pre.extend(scale_mids.cpu().numpy().tolist())
            p1_err = (pred_3d - kp3ds).pow(2).sum(dim=-1).sqrt().mean().item() * 1000.0
            p2_err = p_mpjpe(pred_3d.cpu().numpy(), kp3ds.cpu().numpy()) * 1000.0
            p1_mpjpe += p1_err
            p2_mpjpe += p2_err
            if prefix == "mpi":
                pred_3ds.append(pred_3d.cpu().numpy()); gt_3ds.append(kp3ds.cpu().numpy())
            valid_cnt += 1
        # do some visualization 
        val_iter = iter(val_loader)
        num = np.random.randint(0, len(val_loader) - 1)
        for i in range(num):
            _, _, _, _, _, _ = next(val_iter)
        kp2ds, kp3ds, _, _, _, _ = next(val_iter)
        kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
        kp3ds = kp3ds.cuda()
        pred_3d, _, _ = pose_model(kp2ds, is_train=False, rot=None)
        joints = {"2d input": kp2ds.cpu().numpy(), "lift 3d": pred_3d.cpu().numpy(), "gt 3d": kp3ds.cpu().numpy()}
        vis_image = vis_joints(joints)
        vis_image = torch.from_numpy(vis_image).type(torch.float32).permute(2, 0, 1) / 255
    p1_mpjpe /= valid_cnt 
    p2_mpjpe /= valid_cnt
    if prefix == "mpi":
        auc_val, pckh5 = calc_auc(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        auc_val_p2, pckh5_p2 = calc_auc_aligned(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        extra_dict = {"auc": auc_val, "pckh5": pckh5, "auc_p2": auc_val_p2, "pckh5_p2": pckh5_p2}
        del pred_3ds, gt_3ds
    else: 
        extra_dict = None
    return p1_mpjpe, p2_mpjpe, vis_image, all_scale_mids_pre, extra_dict

def evaluate(val_loader, pose_model, config, is_train=False): 
    # is_train flag for generating the 3d predictions for training set 
    num_frames = config.DATA.NUM_FRAMES 
    use_same_norm_3d = config.DATA.USE_SAME_NORM_3D 
    prefix = "mpi" if config.DATA.DATASET_NAME == "mpi" else "h36m"
    pose_model.eval()
    valid_cnt = 0
    test3d_loss_total_p1, test3d_loss_total_p2 = 0.0, 0.0
    pred_3ds, gt_3ds, all_scales, all_scale_mids, Rs = [], [], [], [], []
    if is_train: 
        factor = 0.6451607 if prefix == "h36m" else 0.729043041
    else:  
        factor = 0.680019 if prefix == "h36m" else 0.7577316
    if not use_same_norm_3d: 
        factor_filename = f"../data/{prefix}_test_factor_3d.pkl" if not is_train else f"../data/{prefix}_train_factor_3d.pkl"
        with open(factor_filename, "rb") as f:
            factor = torch.from_numpy(pkl.load(f)).cuda()
    # print('eval with 15 joints')
    # joint_indices = list(range(14)) + [16]
    with torch.no_grad():
        t = tqdm(val_loader, desc=f"Evaluating...")
        for kp2ds_, kp3ds, _, _, _, _ in t: 
            if valid_cnt == 0: 
                bs = kp2ds_.size(0)
            kp2ds = kp2ds_.cuda(); kp3ds = kp3ds.cuda()
            pred_3d, scales, scale_mids = pose_model(kp2ds, is_train=False, rot=None)
            root3d, root3d_pred = kp3ds[:, 13:14].clone(), pred_3d[:, 13:14].clone()
            kp3ds, pred_3d = kp3ds - root3d, pred_3d - root3d_pred
            # rescale back to original scale from 1 unit
            if not use_same_norm_3d:
                pred_3d = pred_3d * factor[int(bs * valid_cnt):int(bs * (valid_cnt+1))]
            else:
                pred_3d = pred_3d * factor 
            pred_numpy, gt_numpy = pred_3d.cpu().numpy(), kp3ds.cpu().numpy()
            scales = scales.cpu().numpy(); scale_mids = scale_mids.cpu().numpy()
            pred_3ds.append(pred_numpy); gt_3ds.append(gt_numpy); 
            all_scales.append(scales); all_scale_mids.append(scale_mids)
            # calculate the p1 and p2 mpjpe 
            err_p1 = np.sqrt(((pred_numpy - gt_numpy) ** 2).sum(axis=-1)).mean() * 1000.0
            # err_p1 = p_mpjpe(pred_numpy, gt_numpy, rot=False) * 1000.0
            # err_p2 = p_mpjpe(pred_numpy[:, joint_indices], gt_numpy[:, joint_indices], rot=True, trans=True) * 1000.0
            err_p2 = p_mpjpe(pred_numpy, gt_numpy, rot=True, trans=True) * 1000.0
            # err_p2, R = p_mpjpe(pred_numpy, gt_numpy, rot=True, trans=False, scale=False)
            # err_p2 *= 1000.0
            # Rs.append(R)
            test3d_loss_total_p1 += err_p1
            test3d_loss_total_p2 += err_p2
            valid_cnt += 1
    # Rs = np.concatenate(Rs, axis=0)
    # with open("../data/rotations_{}.pkl".format("train" if is_train else "valid"), "wb") as f: 
    #     pkl.dump(Rs, f)
    all_out_data = {
        "p1_mpjpe": test3d_loss_total_p1 / valid_cnt, 
        "p2_mpjpe": test3d_loss_total_p2 / valid_cnt, 
        "joint_3d_gt": np.concatenate(gt_3ds, axis=0), 
        "joint_3d_pre": np.concatenate(pred_3ds, axis=0), 
        "scales": np.concatenate(all_scales, axis=0), 
        "scale_mids": np.concatenate(all_scale_mids, axis=0)
    }
    if prefix == "mpi": 
        auc_val, pckh5 = calc_auc(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        auc_val_p2, pckh5_p2 = calc_auc_aligned(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        all_out_data['auc'] = auc_val 
        all_out_data['pckh5'] = pckh5
        all_out_data['auc_p2'] = auc_val_p2
        all_out_data['pckh5_p2'] = pckh5_p2
    del pred_3ds, gt_3ds

    return all_out_data
