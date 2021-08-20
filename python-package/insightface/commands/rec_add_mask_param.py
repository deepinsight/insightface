
from argparse import ArgumentParser, Namespace

import os
import mxnet as mx
import numpy as np
import numbers
from . import BaseInsightFaceCLICommand
from ..app import MaskRenderer
from ..data import RecBuilder


def rec_add_mask_param_command_factory(args: Namespace):

    return RecAddMaskParamCommand(
        args.input, args.output
    )


class RecAddMaskParamCommand(BaseInsightFaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        _parser = parser.add_parser("rec.addmaskparam")
        _parser.add_argument("input", type=str, help="input rec")
        _parser.add_argument("output", type=str, help="output rec, with mask param")
        _parser.set_defaults(func=rec_add_mask_param_command_factory)

    def __init__(
        self,
        input: str,
        output: str,
    ):
        self._input = input
        self._output = output


    def run(self):
        tool = MaskRenderer()
        tool.prepare(ctx_id=0, det_size=(128,128))
        root_dir = self._input
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        save_path = self._output
        wrec=RecBuilder(path=save_path)
        s = imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            if len(header.label)==2:
                imgidx = np.array(range(1, int(header.label[0])))
            else:
                imgidx = np.array(list(self.imgrec.keys))
        else:
            imgidx = np.array(list(self.imgrec.keys))
        stat = [0, 0]
        print('total:', len(imgidx))
        for iid, idx in enumerate(imgidx):
            #if iid==500000:
            #    break
            if iid%1000==0:
                print('processing:', iid)
            s = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            sample = mx.image.imdecode(img).asnumpy()
            bgr = sample[:,:,::-1]
            params = tool.build_params(bgr)
            #if iid<10:
            #    mask_out = tool.render_mask(bgr, 'mask_blue', params)
            #    cv2.imwrite('maskout_%d.jpg'%iid, mask_out)
            stat[1] += 1
            if params is None:
                wlabel = [label] + [-1.0]*236
                stat[0] += 1
            else:
                #print(0, params[0].shape, params[0].dtype)
                #print(1, params[1].shape, params[1].dtype)
                #print(2, params[2])
                #print(3, len(params[3]), params[3][0].__class__)
                #print(4, params[4].shape, params[4].dtype)
                mask_label = tool.encode_params(params)
                wlabel = [label, 0.0]+mask_label # 237 including idlabel, total mask params size is 235
                if iid==0:
                    print('param size:', len(mask_label), len(wlabel), label)
            assert len(wlabel)==237
            wrec.add_image(img, wlabel)
            #print(len(params))

        wrec.close()
        print('finished on', self._output, ', failed:', stat[0])

