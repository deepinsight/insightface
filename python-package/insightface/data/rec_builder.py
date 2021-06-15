import pickle
import numpy as np
import os
import os.path as osp
import sys
import mxnet as mx


class RecBuilder():
    def __init__(self, path, image_size=(112, 112)):
        self.path = path
        self.image_size = image_size
        self.widx = 0
        self.wlabel = 0
        self.max_label = -1
        assert not osp.exists(path), '%s exists' % path
        os.makedirs(path)
        self.writer = mx.recordio.MXIndexedRecordIO(os.path.join(path, 'train.idx'), 
                                                    os.path.join(path, 'train.rec'),
                                                    'w')
        self.meta = []

    def add(self, imgs):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        assert len(imgs) > 0
        label = self.wlabel
        for img in imgs:
            idx = self.widx
            image_meta = {'image_index': idx, 'image_classes': [label]}
            header = mx.recordio.IRHeader(0, label, idx, 0)
            if isinstance(img, np.ndarray):
                s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
            else:
                s = mx.recordio.pack(header, img)
            self.writer.write_idx(idx, s)
            self.meta.append(image_meta)
            self.widx += 1
        self.max_label = label
        self.wlabel += 1


    def add_image(self, img, label):
        #!!! img should be BGR!!!!
        #assert label >= 0
        #assert label > self.last_label
        idx = self.widx
        header = mx.recordio.IRHeader(0, label, idx, 0)
        if isinstance(label, list):
            idlabel = label[0]
        else:
            idlabel = label
        image_meta = {'image_index': idx, 'image_classes': [idlabel]}
        if isinstance(img, np.ndarray):
            s = mx.recordio.pack_img(header,img,quality=95,img_fmt='.jpg')
        else:
            s = mx.recordio.pack(header, img)
        self.writer.write_idx(idx, s)
        self.meta.append(image_meta)
        self.widx += 1
        self.max_label = max(self.max_label, idlabel)

    def close(self):
        with open(osp.join(self.path, 'train.meta'), 'wb') as pfile:
            pickle.dump(self.meta, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        print('stat:', self.widx, self.wlabel)
        with open(os.path.join(self.path, 'property'), 'w') as f:
            f.write("%d,%d,%d\n" % (self.max_label+1, self.image_size[0], self.image_size[1]))
            f.write("%d\n" % (self.widx))

