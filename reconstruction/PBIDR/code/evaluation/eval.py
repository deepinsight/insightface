import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_rendering = kwargs['eval_rendering']
    eval_animation = kwargs['eval_animation']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'),\
                                                                  id=scan_id, datadir=dataset_conf['data_dir'])
    if torch.cuda.is_available():
        model.cuda()


    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(False, **dataset_conf)

    if eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=eval_dataset.collate_fn
                                                      )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    ####################################################################################################################
    print("evaluating...")

    model.eval()

    detail_3dmm, detail_3dmm_subdivision_full = plt.get_displacement_mesh(model)
    detail_3dmm.export('{0}/Detailed_3dmm_{1}.obj'.format(evaldir, epoch), 'obj')
    detail_3dmm_subdivision_full.export('{0}/Subdivide_full_{1}.obj'.format(evaldir, epoch), 'obj')

    if eval_animation:
        sdf_np0, sdf_np1 = plt.get_displacement_animation(model)
        np.save('{0}/Cropped_Detailed_sdf_{1}.npy'.format(evaldir, epoch), sdf_np0)
        np.save('{0}/Cropped_Subdivide_full_{1}.npy'.format(evaldir, epoch), sdf_np1)

    if eval_rendering:
        images_dir = '{0}/rendering'.format(evaldir)
        utils.mkdir_ifnotexists(images_dir)

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, total_pixels)
            res = []
            for s in split:
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                    'diffuse_values': out['diffuse_values'].detach(),
                    'specular_values': out['specular_values'].detach(),
                    'albedo_values': out['albedo_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = (rgb_eval + 1.) / 2.
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0]))

            diffuse_eval = model_outputs['diffuse_values']
            diffuse_eval = diffuse_eval.reshape(batch_size, total_pixels, 3)
            diffuse_eval = (diffuse_eval + 1.) / 2.
            diffuse_eval = plt.lin2img(diffuse_eval, img_res).detach().cpu().numpy()[0]
            diffuse_eval = diffuse_eval.transpose(1, 2, 0)
            img = Image.fromarray((diffuse_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}_diffuse.png'.format(images_dir, '%03d' % indices[0]))

            specular_eval = model_outputs['specular_values']
            specular_eval = specular_eval.reshape(batch_size, total_pixels, 3)
            specular_eval = (specular_eval + 1.) / 2.
            specular_eval = plt.lin2img(specular_eval, img_res).detach().cpu().numpy()[0]
            specular_eval = specular_eval.transpose(1, 2, 0)
            img = Image.fromarray((specular_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}_specular.png'.format(images_dir, '%03d' % indices[0]))

            albedo_eval = model_outputs['albedo_values']
            albedo_eval = albedo_eval.reshape(batch_size, total_pixels, 3)
            albedo_eval = (albedo_eval + 1.) / 2.
            albedo_eval = plt.lin2img(albedo_eval, img_res).detach().cpu().numpy()[0]
            albedo_eval = albedo_eval.transpose(1, 2, 0)
            img = Image.fromarray((albedo_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}_albedo.png'.format(images_dir, '%03d' % indices[0]))

            rgb_gt = ground_truth['rgb']
            rgb_gt = (rgb_gt + 1.) / 2.
            rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0]
            rgb_gt = rgb_gt.transpose(1, 2, 0)

            mask = model_input['object_mask']
            mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
            mask = mask.transpose(1, 2, 0)

            rgb_eval_masked = rgb_eval * mask
            rgb_gt_masked = rgb_gt * mask

            psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, mask)
            psnrs.append(psnr)

        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))



def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/test.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=0, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_rendering', default=False, action="store_true",help='If set, evaluate rendering quality.')
    parser.add_argument('--eval_animation', default=False, action="store_true",help='If set, evaluate rendering quality.')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             eval_animation=opt.eval_animation
             )
