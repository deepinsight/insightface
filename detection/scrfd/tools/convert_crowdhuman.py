from __future__ import print_function
import cv2
import argparse
import os
import os.path as osp
import shutil
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='convert crowdhuman dataset to scrfd format')
    parser.add_argument('--raw', help='raw dataset dir')
    parser.add_argument('--save', default='data/crowdhuman', help='save path')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    raw_image_dir = osp.join(args.raw, 'Images')
    for subset in ['train', 'val']:
        save_image_dir = osp.join(args.save, subset, 'images')
        if not osp.exists(save_image_dir):
            os.makedirs(save_image_dir)
        anno_file = osp.join(args.raw, 'annotation_%s.odgt'%subset)
        fullbody_anno_file = osp.join(osp.join(args.save, subset, "label_fullbody.txt"))
        head_anno_file = osp.join(osp.join(args.save, subset, "label_head.txt"))
        fullbody_f = open(fullbody_anno_file, 'w')
        head_f = open(head_anno_file, 'w')
        for line in open(anno_file, 'r'):
            data = json.loads(line)
            img_id = data['ID']
            img_name = "%s.jpg"%img_id
            raw_image_file = osp.join(raw_image_dir, img_name)
            target_image_file = osp.join(save_image_dir, img_name)
            img = cv2.imread(raw_image_file)
            print(raw_image_file, img.shape)
            fullbody_f.write("# %s %d %d\n"%(img_name,img.shape[1],img.shape[0]))
            head_f.write("# %s %d %d\n"%(img_name,img.shape[1],img.shape[0]))
            shutil.copyfile(raw_image_file, target_image_file)
            items = data['gtboxes']
            for item in items:
                fbox = item['fbox']
                is_ignore = False
                extra = item['extra']
                if 'ignore' in extra:
                    is_ignore = extra['ignore']==1
                bbox = np.array(fbox, dtype=np.float32)
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                if is_ignore:
                    fullbody_f.write("%.5f %.5f %.5f %.5f %d\n"%(bbox[0], bbox[1], bbox[2], bbox[3], is_ignore))
                else:
                    vbox = item['vbox']
                    vbox = np.array(vbox, dtype=np.float32)
                    vbox[2] += vbox[0]
                    vbox[3] += vbox[1]
                    x1, y1, x2, y2 = vbox[0], vbox[1], vbox[2], vbox[3]
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    kps = np.ones( (5,3), dtype=np.float32)
                    kps[0,0] = x1
                    kps[0,1] = y1
                    kps[1,0] = x2
                    kps[1,1] = y1
                    kps[2,0] = cx
                    kps[2,1] = cy
                    kps[3,0] = x1
                    kps[3,1] = y2
                    kps[4,0] = x2
                    kps[4,1] = y2
                    kps_str = " ".join(["%.5f"%x for x in kps.flatten()])
                    fullbody_f.write("%.5f %.5f %.5f %.5f %s\n"%(bbox[0], bbox[1], bbox[2], bbox[3], kps_str))


                hbox = item['hbox']
                is_ignore = False
                extra = item['head_attr']
                if 'ignore' in extra:
                    is_ignore = extra['ignore']==1
                bbox = np.array(hbox, dtype=np.float32)
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                head_f.write("%.5f %.5f %.5f %.5f %d\n"%(bbox[0], bbox[1], bbox[2], bbox[3], is_ignore))
        fullbody_f.close()
        head_f.close()


if __name__ == '__main__':
    main()

