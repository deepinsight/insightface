import time
import pickle as pkl
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm 
from lib.core.loss import *
from lib.utils.vis import vis_joints
from lib.utils.utils import p_mpjpe, get_pck3d, calc_auc, calc_auc_aligned

def train(train_loader, pose_model, discriminator, temp_discriminator, \
    optimizer_g, optimizer_d, optimizer_d_temp, epoch, config, summary_writer=None, print_interval=30):  
    # parse the config
    print_info = config.PRINT_INFO 
    num_frames = config.DATA.NUM_FRAMES
    num_critics = config.TRAIN.NUM_CRITICS 
    num_critics_temp = config.TRAIN.NUM_CRITICS_TEMP 
    loss_weights = config.TRAIN.LOSS_WEIGHTS
    scale_loss_weights = config.TRAIN.SCALE_LOSS_WEIGHTS
    use_scaler = config.TRAIN.USE_SCALER
    use_dis = config.TRAIN.USE_DIS; use_temp = config.TRAIN.USE_TEMP
    use_bone_noscale = config.TRAIN.USE_BONE_NOSCALE
    use_new_temp = config.TRAIN.USE_NEW_TEMP
    multi_new_temp = config.TRAIN.MULTI_NEW_TEMP
    cat_current = config.TRAIN.CAT_CURRENT
    scale_mid_mean, scale_mid_std = config.DATA.SCALE_MID_MEAN, config.DATA.SCALE_MID_STD
    w1, w2, w3, w4 = loss_weights
    use_2d_gt_supervision = config.DATA.USE_2D_GT_SUPERVISION
    fix_traj = config.FIX.FIX_TRAJ; fix_traj_by_rot = config.FIX.FIX_TRAJ_BY_ROT; fix_bone_loss = config.FIX.FIX_BONE_LOSS
    if len(scale_loss_weights) == 2: 
        w_s, w_s_bone = scale_loss_weights
    else: 
        w_s, w_s_bone, w_s_neighbour = scale_loss_weights
    scaler_criterion = torch.nn.KLDivLoss()
    string_format1 = "[Kl Loss: {:.4f}] | [Bone Loss: {:.4f}] | [Scale Error: {:.4f}]"
    string_format2 = "[Est Loss: {:.4f}] | [3d Loss: {:.4f}] | [2d Loss: {:.4f}] | [D_G loss: {:.4f}] | [D_D Loss: {:.4f}] | [T Loss: {:.4f}]"
    prefix = config.DATA.DATASET_NAME
    # for 10w data 
    # factor = 0.651607 if config.DATA.DATASET_NAME == "h36m" else 0.7349340771
    # for 60w data
    is_h36m = config.DATA.DATASET_NAME == 'h36m'
    root_idx = 13 if config.DATA.NUM_JOINTS == 17 else 12   
    if config.DATA.DATASET_NAME == 'surreal': 
        factor = 0.563642
    else: 
        factor = 0.6451607 if config.DATA.DATASET_NAME == "h36m" else 0.7286965902

    # pose_model.train()
    # discriminator.train(); temp_discriminator.train()
    pose_model.train()
    discriminator.train(); temp_discriminator.train()
    t = tqdm(train_loader, desc=f"Epoch: {epoch}, training...")
    all_scale_mids = []
    train_cnt = 0
    running_dis_loss, running_pose_loss, running_temp_loss = 0.0, 0.0, 0.0
    for kp2ds, kp3ds, rot, diff1, diff2, scales_gt in t:
        bs = kp2ds.size(0)
        kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
        rot = rot.cuda(); scales_gt = scales_gt.cuda()
        diff1 = diff1.cuda(); diff2 = diff2.cuda()
        l_t = torch.tensor(0.0).type_as(kp2ds)
        l_g = torch.tensor(0.0).type_as(kp2ds)
        loss_dis1 = torch.tensor(0.0).type_as(kp2ds)
        
        # if (not use_dis and not use_temp) or (use_dis and use_temp and train_cnt % num_critics == 0):
        optimizer_g.zero_grad()
        (first_half_feature, second_half_feature), est_3d, tran_3d, proj_2d, \
            recn_3d, recovered_3d, recn_2d, scale_mids, scales = pose_model(kp2ds, is_train=True, rot=rot)
        if use_scaler: 
            (first_half_feature, second_half_feature), est_3d_diff, tran_3d_diff, proj_2d_diff, \
                recn_3d_diff, recovered_3d_diff, recn_2d_diff, scale_mids_diff, scales_diff = pose_model(diff2, is_train=True, rot=rot)
            all_scale_mids.extend(scale_mids.detach().cpu().numpy().tolist())
            scaled_mids = torch.cat([scale_mids, scale_mids_diff])

        # minus the root 
        if not config.TRAIN.GENERIC_BASELINE:
            kp3ds, est_3d_centered = kp3ds - kp3ds[:, root_idx:root_idx+1], (est_3d - est_3d[:, root_idx:root_idx+1]) * factor
        else:
            kp3ds, est_3d_centered = kp3ds - kp3ds[:, root_idx:root_idx+1], (est_3d - est_3d[:, root_idx:root_idx+1])
        est_gt_loss_3d = loss_3d(est_3d_centered, kp3ds)
        est_dist_2d = loss_2d(kp2ds, diff1)
        l3d = loss_3d(tran_3d, recn_3d)
        l2d = loss_2d(recn_2d, kp2ds)
        loss_pose = w1 * l3d + w2 * l2d

        if use_dis: 
            g_fake_logits = discriminator(proj_2d)
            l_g = (g_fake_logits - 1).pow(2).mean()
            # l_g = loss_gadv(g_fake_logits)
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

            l_t = loss_3d(est_3d - est_3d_diff, recn_3d - recn_3d_diff)
            loss_pose += w4 * l_t

        if use_bone_noscale: 
            (_, _), est_3d_diff2, _, _, _, _, _, _, _ = pose_model(diff2, is_train=True, rot=rot)
            l_b = loss_bone(est_3d, est_3d_diff2)
            loss_pose += 1.0 * l_b

        if use_scaler: 
            assert scale_mids is not None
            # add kl loss between scale_mids and expected distribution(mean: 0.707, std: 0.025)
            scales_dist = (scales_gt - scales).abs().mean()
            mean, std = scale_mids.mean(), scale_mids.std()
            loss_s = kl_criterion(mean, std, scale_mid_mean, scale_mid_std)

            # besides, add bone loss
            loss_s_bone = loss_bone(est_3d, est_3d_diff)

            all_scales = []
            loss_pose = loss_pose + w_s * loss_s + w_s_bone * loss_s_bone

        loss_pose.backward()
        optimizer_g.step()
        
        # train the discriminator
        if use_dis and train_cnt % num_critics == 0:
            optimizer_d.zero_grad()
            _, _, _, proj_2d, _, _, _, _, _ = pose_model(kp2ds, is_train=True, rot=rot)
            d_real_logits = discriminator(kp2ds)
            d_fake_logits = discriminator(proj_2d.detach())
            # loss_dis1 = loss_dadv(d_real_logits, d_fake_logits)
            loss_dis1 = (d_real_logits - 1).pow(2).mean() + d_fake_logits.pow(2).mean()
            loss_dis1.backward()
            optimizer_d.step()

        # train the temporal discriminator
        if use_temp and train_cnt % num_critics_temp == 0:
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
            if use_scaler: 
                summary_writer.add_scalar("scale_pre_gt_dist/iters", scales_dist.item(), iters)
                summary_writer.add_scalar("scale_loss/iters", loss_s.item(), iters)
                summary_writer.add_scalar("scale_bone_loss/iters", loss_s_bone.item(), iters)

        running_pose_loss += loss_pose.item() * kp2ds.size(0)
        if use_dis: 
            running_dis_loss += loss_dis1.item() * kp2ds.size(0)
        if use_temp: 
            running_temp_loss += loss_dis2.item() * kp2ds.size(0)
        if print_info and train_cnt % print_interval == 0:
            if use_scaler: 
                string1 = string_format1.format(loss_s.item(), loss_s_bone.item(), scales_dist.item())
            string2 = string_format2.format(est_gt_loss_3d.item(), l3d.item(), l2d.item(), l_g.item(), loss_dis1.item(), l_t.item())
            string = string1 + "\n" + string2 if use_scaler else string2
            tqdm.write(string)
            # t.set_postfix(losses)
        train_cnt += 1

    # finally do some visualization
    if summary_writer is not None: 
        # visualize the grad norm and weight norm
        for param in pose_model.parameters(): 
            weight_norm = param.data.norm()
            grad_norm = param.grad.data.norm()
            summary_writer.add_scalar("weight_norm", weight_norm, epoch)
            summary_writer.add_scalar("grad_norm", grad_norm, epoch)
            # only visualize one epoch, so exit 
            break
        with torch.no_grad():
            kp2ds, kp3ds, rot, _, _, _ = next(iter(train_loader))
            kp2ds = kp2ds.cuda()
            kp3ds = kp3ds.cuda()
            rot = rot.cuda()
            _, est_3d, tran_3d, proj_2d, recn_3d, recovered_3d, recn_2d, _, _ = pose_model(kp2ds, is_train=True, rot=rot) 
            joints = {"input 2d": kp2ds.cpu().numpy(), "recn_2d": recn_2d.cpu().numpy(), "gt 3d": kp3ds.cpu().numpy(), "est 3d": est_3d.cpu().numpy(), "tran 3d": tran_3d.cpu().numpy(),  "recn 3d": recn_3d.cpu().numpy()}
            if config.DATA.NUM_JOINTS > 15:
                vis_image = vis_joints(joints)
                vis_image = torch.from_numpy(vis_image).type(torch.float32).permute(2, 0, 1) / 255
                summary_writer.add_image("train_joints/epoch", vis_image, epoch)
    
    return running_dis_loss / len(train_loader.dataset), running_pose_loss / len(train_loader.dataset), running_temp_loss / len(train_loader.dataset), all_scale_mids


def validate(val_loader, pose_model, epoch, config, factor=0.6451607):
    # parse the config 
    num_frames = config.DATA.NUM_FRAMES 
    use_same_norm_3d = config.DATA.USE_SAME_NORM_3D
    prefix = "mpi" if config.DATA.DATASET_NAME == "mpi" else "h36m"
    pose_model.eval()
    prefix = config.DATA.DATASET_NAME
    # for 10 w
    # factor = 0.680019 if prefix == "h36m" else 0.7577316 
    if prefix == 'surreal': 
        factor = 0.56071004
    else:
        factor = 0.680019 if prefix == "h36m" else 0.7577316
    root_idx = 13 if config.DATA.NUM_JOINTS == 17 else 12   
    if not use_same_norm_3d:
        with open(f"../data/{prefix}_test_factor_3d.pkl", "rb") as f:
            factor = torch.from_numpy(pkl.load(f)).cuda()
    center_idx = int((num_frames - 1) / 2)
    valid_cnt = 0 
    p1_mpjpe, p2_mpjpe = 0.0, 0.0
    pck_3d = 0.0
    all_scale_mids_pre = []
    pred_3ds, gt_3ds = [], []
    with torch.no_grad():
        t = tqdm(val_loader, desc=f"Epoch: {epoch}, testing...")
        for kp2ds, kp3ds, rot, _, _, _ in t:
            if valid_cnt == 0:
                bs = kp2ds.size(0)
            kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
            rot = rot.cuda()
            pred_3d, _, scale_mids = pose_model(kp2ds, is_train=False, rot=rot)
            root3d, root3d_pred = kp3ds[:, root_idx:root_idx+1].clone(), pred_3d[:, root_idx:root_idx+1].clone()
            kp3ds, pred_3d = kp3ds - root3d, pred_3d - root3d_pred
            # rescale back to original scale from 1 unit
            if not config.TRAIN.GENERIC_BASELINE:
                if not use_same_norm_3d:
                    pred_3d = pred_3d * factor[int(bs * valid_cnt):int(bs * (valid_cnt+1))]
                else:
                    pred_3d = pred_3d * factor 
            # all_scale_mids_pre.extend(scale_mids.cpu().numpy().tolist())
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
        kp2ds, kp3ds, rot, _, _, _ = next(val_iter)
        kp2ds = kp2ds.cuda(); kp3ds = kp3ds.cuda()
        kp3ds = kp3ds.cuda()
        pred_3d, _, _ = pose_model(kp2ds, is_train=False, rot=rot)
        joints = {"2d input": kp2ds.cpu().numpy(), "lift 3d": pred_3d.cpu().numpy(), "gt 3d": kp3ds.cpu().numpy()}
        vis_image = vis_joints(joints) if config.DATA.NUM_JOINTS > 15 else np.zeros((200, 200, 3), dtype=np.uint8)
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
    pose_model.eval()
    center_idx = int((num_frames - 1) / 2)
    valid_cnt = 0
    test3d_loss_total_p1, test3d_loss_total_p2 = 0.0, 0.0
    root_idx = 13 if config.DATA.NUM_JOINTS == 17 else 12   
    pred_3ds, gt_3ds, all_scales, all_scale_mids = [], [], [], []
    prefix = config.DATA.DATASET_NAME
    if is_train: 
        if prefix == 'surreal': 
            factor = 0.563642
        else:
            factor = 0.6451607 if prefix == "h36m" else 0.728695902
    else:  
        if prefix == 'surreal': 
            factor = 0.56071004
        else:
            factor = 0.680019 if prefix == "h36m" else 0.7577316
    if not use_same_norm_3d: 
        factor_filename = f"../data/{prefix}_test_factor_3d.pkl" if not is_train else f"../data/{prefix}_train_factor_3d.pkl"
        with open(factor_filename, "rb") as f:
            factor = torch.from_numpy(pkl.load(f)).cuda()
    # print('eval with 15 joints')
    # exclude spine: 14, neck: 15
    # joint_indices = list(range(14)) + [16]
    with torch.no_grad():
        t = tqdm(val_loader, desc=f"Evaluating...")
        for kp2ds_, kp3ds, rot, _, _, _ in t: 
            if valid_cnt == 0: 
                bs = kp2ds_.size(0)
            kp2ds = kp2ds_.cuda(); kp3ds = kp3ds.cuda()
            rot = rot.cuda()
            pred_3d, scales, scale_mids = pose_model(kp2ds, is_train=False, rot=rot)
            root3d, root3d_pred = kp3ds[:, root_idx:root_idx+1].clone(), pred_3d[:, root_idx:root_idx+1].clone()
            kp3ds, pred_3d = kp3ds - root3d, pred_3d - root3d_pred
            # rescale back to original scale from 1 unit
            if not use_same_norm_3d:
                pred_3d = pred_3d * factor[int(bs * valid_cnt):int(bs * (valid_cnt+1))]
            else:
                pred_3d = pred_3d * factor 
            pred_numpy, gt_numpy = pred_3d.cpu().numpy(), kp3ds.cpu().numpy()
            pred_3ds.append(pred_numpy); gt_3ds.append(gt_numpy); 
            if scales is not None: 
                scales = scales.cpu().numpy()
                all_scales.append(scales)
            if scale_mids is not None:
                scale_mids = scale_mids.cpu().numpy()
                all_scale_mids.append(scale_mids)
            # calculate the p1 and p2 mpjpe 
            err_p1 = np.sqrt(((pred_numpy - gt_numpy) ** 2).sum(axis=-1)).mean() * 1000.0
            if err_p1 > 10000.0: 
                import ipdb 
                ipdb.set_trace()
            # err_p2 = p_mpjpe(pred_numpy[:, joint_indices], gt_numpy[:, joint_indices]) * 1000.0
            err_p2 = p_mpjpe(pred_numpy, gt_numpy) * 1000.0
            test3d_loss_total_p1 += err_p1
            test3d_loss_total_p2 += err_p2
            valid_cnt += 1
    all_out_data = {
        "p1_mpjpe": test3d_loss_total_p1 / valid_cnt, 
        "p2_mpjpe": test3d_loss_total_p2 / valid_cnt, 
        "joint_3d_gt": np.concatenate(gt_3ds, axis=0), 
        "joint_3d_pre": np.concatenate(pred_3ds, axis=0), 
    }
    if len(all_scales) > 0: 
        all_out_data['scales'] = np.concatenate(all_scales, axis=0)
    if len(all_scale_mids) > 0: 
        all_out_data['scale_mids'] = np.concatenate(all_scale_mids, axis=0)
        
    if prefix == "mpi": 
        auc_val, pckh5 = calc_auc(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        auc_val_p2, pckh5_p2 = calc_auc_aligned(np.concatenate(pred_3ds, axis=0), np.concatenate(gt_3ds, axis=0))
        all_out_data['auc'] = auc_val 
        all_out_data['pckh5'] = pckh5
        all_out_data['auc_p2'] = auc_val_p2
        all_out_data['pckh5_p2'] = pckh5_p2

    del pred_3ds, gt_3ds

    return all_out_data
