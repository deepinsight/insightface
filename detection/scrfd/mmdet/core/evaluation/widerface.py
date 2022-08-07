"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

from __future__ import absolute_import
import os
import tqdm
import pickle
import datetime
import argparse
import numpy as np
from scipy.io import loadmat
#from facedet.evaluation.box_utils import jaccard
#from facedet.evaluation.bbox import bbox_overlaps
#import torch
#from mmdet.core.bbox import bbox_overlaps

#def intersect(box_a, box_b):
#    A = box_a.size(0)
#    B = box_b.size(0)
#    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
#                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
#    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
#                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
#    inter = torch.clamp((max_xy - min_xy), min=0)
#    return inter[:, :, 0] * inter[:, :, 1]
#
#def jaccard(box_a, box_b):
#    inter = intersect(box_a, box_b)
#    #torch.cuda.empty_cache()
#    if not inter.is_cuda:
#        box_a_cpu = box_a.cpu()
#        box_b_cpu = box_b.cpu()
#        area_a_cpu = ((box_a_cpu[:, 2]-box_a_cpu[:, 0]) *
#              (box_a_cpu[:, 3]-box_a_cpu[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#        area_b_cpu = ((box_b_cpu[:, 2]-box_b_cpu[:, 0]) *
#              (box_b_cpu[:, 3]-box_b_cpu[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#        union_cpu = area_a_cpu + area_b_cpu - inter.cpu()
#        return inter / union_cpu
#    else:
#        area_a = ((box_a[:, 2]-box_a[:, 0]) *
#              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
#        area_b = ((box_b[:, 2]-box_b[:, 0]) *
#              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
#        union = area_a + area_b - inter
#
#        return inter / union  # [A,B]
#
def bbox_overlaps(boxes, query_boxes):
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] +
                          1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] +
                                1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps

def bbox_overlap(a, b):
    x1 = np.maximum(a[:,0], b[0])
    y1 = np.maximum(a[:,1], b[1])
    x2 = np.minimum(a[:,2], b[2])
    y2 = np.minimum(a[:,3], b[3])
    w = x2-x1+1
    h = y2-y1+1
    inter = w*h
    aarea = (a[:,2]-a[:,0]+1) * (a[:,3]-a[:,1]+1)
    barea = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    o = inter / (aarea+barea-inter)
    o[w<=0] = 0
    o[h<=0] = 0
    return o

def __bbox_overlap(a, b):
    x1 = torch.max(a[:,0], b[0])
    y1 = torch.max(a[:,1], b[1])
    x2 = torch.min(a[:,2], b[2])
    y2 = torch.min(a[:,3], b[3])
    w = x2-x1+1
    h = y2-y1+1
    inter = w*h
    aarea = (a[:,2]-a[:,0]+1) * (a[:,3]-a[:,1]+1)
    barea = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    o = inter / (aarea+barea-inter)
    o[w<=0] = 0
    o[h<=0] = 0
    return o

def np_around(array, num_decimals=0):
    #return array
    return np.around(array, decimals=num_decimals)

#def compute_iou(box_a, box_b):
#    x0 = np.maximum(box_a[:,0], box_b[0])
#    y0 = np.maximum(box_a[:,1], box_b[1])
#    x1 = np.minimum(box_a[:,2], box_b[2])
#    y1 = np.minimum(box_a[:,3], box_b[3])
#    #print ('x0', x0[0], x1[0], y0[0], y1[0], box_a[0], box_b[:])
#    #w = np.maximum(x1 - x0 + 1, 0) 
#    w = np_around(x1 - x0 + 1) 
#    #h = np.maximum(y1 - y0 + 1, 0)
#    h = np_around(y1 - y0 + 1)
#    inter = np_around(w * h)
#    area_a = (box_a[:,2] - box_a[:,0] + 1) * (box_a[:,3] - box_a[:,1] + 1)
#    area_a = np_around(area_a)
#    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
#    area_b = np_around(area_b)
#    iou = inter / (area_a + area_b - inter)
#    iou[w <= 0] = 0
#    iou[h <=0] = 0
#    return iou

def np_round(val, decimals=4):
    return val
    #if isinstance(val, np.ndarray):
    #    val = np.around(val, decimals=decimals)
    #return val


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            boxes = pickle.load(f)
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    #print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    with open(cache_file, 'wb') as f:
        pickle.dump(boxes, f)
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = -1 
    min_score = 2

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score).astype(np.float64)/diff
    return pred


def image_eval(pred, gt, ignore, iou_thresh, mpp):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    
    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    gt_overlap_list = mpp.starmap(bbox_overlap, zip([_gt]*_pred.shape[0],[_pred[h] for h in range(_pred.shape[0])]))

    #use_cuda = True
    #if use_cuda:
    #    _pred = torch.cuda.FloatTensor(_pred[:,:4])
    #    _gt = torch.cuda.FloatTensor(_gt)
    #else:
    #    _pred = torch.FloatTensor(_pred[:,:4])
    #    _gt = torch.FloatTensor(_gt)

    #overlaps = jaccard(_pred, _gt).cpu().numpy()
    #overlaps = compute_iou((_pred[:, :4]), (_gt))

    #overlaps = bbox_overlaps(_pred, _gt)

    #if use_cuda:
    #    overlaps = overlaps.cpu().numpy()
    #else:
    #    overlaps = overlaps.numpy()

    for h in range(_pred.shape[0]):

        #gt_overlap = overlaps[h]
        #gt_overlap = bbox_overlap(_gt, _pred[h])
        gt_overlap = gt_overlap_list[h]
        #if use_cuda:
        #    gt_overlap = gt_overlap.cpu().numpy()
        #else:
        #    gt_overlap = gt_overlap.numpy()

        #max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        #gt_overlap = compute_iou(_gt, _pred[h, :4])
        #exit()
        #exit()
        #print ('overlap', gt_overlap)
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)

    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    fp = np.zeros((pred_info.shape[0],), dtype=np.int)
    last_info = [-1, -1]
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index) #valid pred number
            pr_info[t, 1] = pred_recall[r_index] # valid gt number

            if t>0 and pr_info[t, 0] > pr_info[t-1,0] and pr_info[t, 1]==pr_info[t-1,1]:
                fp[r_index] = 1
                #if thresh>=0.85:
                #    print(thresh, t, pr_info[t])
    #print(pr_info[:10,0])
    #print(pr_info[:10,1])
    return pr_info, fp


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        #_pr_curve[i, 0] = round(pr_curve[i, 1] / pr_curve[i, 0], 4)
        #_pr_curve[i, 1] = round(pr_curve[i, 1] / count_face, 4)
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    #print ('rec:', rec)
    #print ('pre:', prec)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np_round(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


def wider_evaluation(pred, gt_path, iou_thresh=0.5, debug=False):
    #pred = get_preds(pred)
    pred = norm_score(pred)
    thresh_num = 1000
    #thresh_num = 2000
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    from multiprocessing import Pool
    #from multiprocessing.pool import ThreadPool
    mpp = Pool(8)
    aps = [-1.0, -1.0, -1.0]
    meta = {}
    #setting_id = 2
    print('')
    for setting_id in range(3):
    #for setting_id in range(1):
        ta = datetime.datetime.now()
        # different setting
        #iou_th = 0.5 #+ 0.05 * idx
        iou_th = iou_thresh
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        #pbar = tqdm.tqdm(range(event_num))
        #for i in pbar:
        high_score_count = 0
        high_score_fp_count = 0
        for i in range(event_num):
            #pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                img_name = str(img_list[j][0][0])
                pred_info = pred_list[img_name]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                #print ('keep_index', keep_index)
                count_face += len(keep_index)
                

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                #ignore = np.zeros(gt_boxes.shape[0])
                #if len(keep_index) != 0:
                #    ignore[keep_index-1] = 1
                #assert len(keep_index)>0
                ignore = np.zeros(gt_boxes.shape[0], dtype=np.int)
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_info = np_round(pred_info,1)
                #print('ignore:', len(ignore), len(np.where(ignore==1)[0]))
                #pred_sort_idx= np.argsort(pred_info[:,4])
                #pred_info = pred_info[pred_sort_idx][::-1]
                #print ('pred_info', pred_info[:20, 4])
                #exit()


                gt_boxes = np_round(gt_boxes)
                #ignore = np_round(ignore)
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_th, mpp)
                #print(pred_recall[:10], proposal_list[:10])
                #print('1 stage', pred_recall, proposal_list)
                #print(pred_info.shape, pred_recall.shape)

                _img_pr_info, fp = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                #for f in range(pred_info.shape[0]):
                #    _score = pred_info[f,4]
                #    if _score<0.929:
                #        break
                #    high_score_count+=1
                #    if fp[f]==1:
                #        w = pred_info[f, 2]
                #        h = pred_info[f, 3]
                #        print('fp:', event_name, img_name, _score, w, h)
                #        high_score_fp_count+=1
                pr_curve += _img_pr_info
        #print ('pr_curve', pr_curve, count_face)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        #print(pr_curve.shape)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        #for f in range(thresh_num):
        #    print('R-P:', recall[f], propose[f])
        for srecall in np.arange(0.1, 1.0001, 0.1):
            rindex = len(np.where(recall<=srecall)[0])-1
            rthresh = 1.0 - float(rindex)/thresh_num
            print('Recall-Precision-Thresh:', recall[rindex], propose[rindex], rthresh)

        ap = voc_ap(recall, propose)
        aps[setting_id] = ap
        tb = datetime.datetime.now()
        #print('high score count:', high_score_count)
        #print('high score fp count:', high_score_fp_count)
        print('%s cost %.4f seconds, ap: %.5f'%(settings[setting_id], (tb-ta).total_seconds(), ap))

    return aps

def get_widerface_gts(gt_path):
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)

    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    all_results = []
    for setting_id in range(3):
        results = {}
        gt_list = setting_gts[setting_id]
        count_face = 0
        # [hard, medium, easy]
        #pbar = tqdm.tqdm(range(event_num))
        #for i in pbar:
        for i in range(event_num):
            #pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]
            results[event_name] = {}

            for j in range(len(img_list)):

                gt_boxes = gt_bbx_list[j][0].astype('float').copy()
                gt_boxes[:,2] += gt_boxes[:,0]
                gt_boxes[:,3] += gt_boxes[:,1]
                keep_index = sub_gt_list[j][0].copy()
                #print ('keep_index', keep_index.shape)
                count_face += len(keep_index)
                

                if len(gt_boxes) == 0:
                    results[event_name][str(img_list[j][0][0])] = np.empty( (0,4) )
                    continue
                keep_index -= 1
                keep_index = keep_index.flatten()
                #ignore = np.zeros(gt_boxes.shape[0])
                #if len(keep_index) != 0:
                #    ignore[keep_index-1] = 1
                #assert len(keep_index)>0
                #ignore = np.zeros(gt_boxes.shape[0], dtype=np.int)
                #if len(keep_index) != 0:
                #    ignore[keep_index-1] = 1
                #print('ignore:', len(ignore), len(np.where(ignore==1)[0]))
                #pred_sort_idx= np.argsort(pred_info[:,4])
                #pred_info = pred_info[pred_sort_idx][::-1]
                #print ('pred_info', pred_info[:20, 4])
                #exit()
                #if setting_id==2 and len(keep_index)<gt_boxes.shape[0]:
                #    print(gt_boxes.shape, keep_index.shape)
                
                gt_boxes = np_round(gt_boxes)[keep_index,:]

                results[event_name][str(img_list[j][0][0])] = gt_boxes
        all_results.append(results)
    return all_results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default='')
    parser.add_argument('-g', '--gt', default='./ground_truth/')

    args = parser.parse_args()
    evaluation(args.pred, args.gt)














