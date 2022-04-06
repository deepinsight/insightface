# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import os
import paddle
import logging
import numpy as np
import shutil
import json
from utils.rearrange_weight import rearrange_weight


class Checkpoint(object):
    def __init__(self,
                 rank,
                 world_size,
                 embedding_size,
                 num_classes,
                 model_save_dir="./",
                 checkpoint_dir=None,
                 max_num_last_checkpoint=3):

        self.rank: int = rank
        self.world_size: int = world_size
        self.embedding_size: int = embedding_size
        self.num_classes: int = num_classes
        self.model_save_dir: str = model_save_dir
        self.checkpoint_dir: str = checkpoint_dir
        self.max_num_last_checkpoint: int = max_num_last_checkpoint

    def save(self,
             backbone: paddle.nn.Layer,
             classifier: paddle.nn.Layer=None,
             optimizer=None,
             epoch=0,
             for_train=True):

        model_save_dir = os.path.join(self.model_save_dir, str(epoch))
        if not os.path.exists(model_save_dir):
            # may be more than one processes trying
            # to create the directory
            try:
                os.makedirs(model_save_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

        if self.rank == 0:
            # for non dist param, we only save their at rank 0.
            for name, param in backbone.state_dict().items():
                paddle.save(
                    param,
                    os.path.join(model_save_dir, param.name + '.pdparam'))

        if classifier is not None:
            # for dist param, we need to save their at all ranks.
            for name, param in classifier.state_dict().items():
                paddle.save(
                    param,
                    os.path.join(model_save_dir, param.name + '.pdparam'))

        if for_train:
            assert optimizer is not None
            opt_state_dict = optimizer.state_dict()
            lr_state_dict = opt_state_dict['LR_Scheduler']
            for name, opt in opt_state_dict.items():
                if '@GRAD' in name:
                    continue
                # for non dist opt var, we only save their at rank 0,
                # but for dist opt var, we need to save their at all ranks.
                if 'dist@' in name and '@rank@' in name or self.rank == 0:
                    paddle.save(opt,
                                os.path.join(model_save_dir, name + '.pdopt'))

            if self.rank == 0:
                # save some extra info for resume
                # pretrain_world_size, embedding_size, num_classes are used for
                # re-split fc weight when gpu setting changed.
                # epoch use to restart.
                config_file = os.path.join(model_save_dir, 'meta.json')
                extra_info = dict()
                extra_info["pretrain_world_size"] = self.world_size
                extra_info["embedding_size"] = self.embedding_size
                extra_info['num_classes'] = self.num_classes
                extra_info['epoch'] = epoch
                extra_info['lr_state'] = lr_state_dict
                with open(config_file, 'w') as f:
                    json.dump(extra_info, f)

        logging.info("Save model to {}.".format(model_save_dir))
        if self.rank == 0 and self.max_num_last_checkpoint > 0:
            for idx in range(-1, epoch - self.max_num_last_checkpoint + 1):
                path = os.path.join(self.model_save_dir, str(idx))
                if os.path.exists(path):
                    logging.info("Remove checkpoint {}.".format(path))
                    shutil.rmtree(path)

    def load(self,
             backbone: paddle.nn.Layer,
             classifier: paddle.nn.Layer=None,
             optimizer=None,
             for_train=True,
             dtype=None):

        assert os.path.exists(self.checkpoint_dir)
        checkpoint_dir = os.path.abspath(self.checkpoint_dir)

        param_state_dict = {}
        opt_state_dict = {}
        dist_param_state_dict = {}

        dist_weight_state_dict = {}
        dist_weight_velocity_state_dict = {}
        dist_bias_state_dict = {}
        dist_bias_velocity_state_dict = {}
        for path in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, path)
            if not os.path.isfile(path):
                continue

            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)

            if ext not in ['.pdopt', '.pdparam']:
                continue

            if not for_train and ext == '.pdopt':
                continue

            tensor = paddle.load(path, return_numpy=True)
            if dtype:
                assert dtype in ['float32', 'float16']
                tensor = tensor.astype(dtype)

            if 'dist@' in name and '@rank@' in name:
                if '.w' in name and 'velocity' not in name:
                    dist_weight_state_dict[name] = tensor
                elif '.w' in name and 'velocity' in name:
                    dist_weight_velocity_state_dict[name] = tensor
                elif '.b' in name and 'velocity' not in name:
                    dist_bias_state_dict[name] = tensor
                elif '.b' in name and 'velocity' in name:
                    dist_bias_velocity_state_dict[name] = tensor

            else:
                if ext == '.pdparam':
                    param_state_dict[name] = tensor
                else:
                    opt_state_dict[name] = tensor

        if classifier is not None and for_train:
            meta_file = os.path.join(checkpoint_dir, 'meta.json')
            if not os.path.exists(meta_file):
                logging.error(
                    "Please make sure the checkpoint dir {} exists, and "
                    "parameters in that dir are validating.".format(
                        checkpoint_dir))
                exit()

            with open(meta_file, 'r') as handle:
                extra_info = json.load(handle)

            # Preporcess distributed parameters.
            pretrain_world_size = extra_info['pretrain_world_size']
            assert pretrain_world_size > 0
            embedding_size = extra_info['embedding_size']
            assert embedding_size == self.embedding_size
            num_classes = extra_info['num_classes']
            assert num_classes == self.num_classes

            logging.info(
                "Parameters for pre-training: pretrain_world_size ({}), "
                "embedding_size ({}), and num_classes ({}).".format(
                    pretrain_world_size, embedding_size, num_classes))
            logging.info("Parameters for inference or fine-tuning: "
                         "world_size ({}).".format(self.world_size))

            rank_str = '%05d' % self.rank

            dist_weight_state_dict = rearrange_weight(
                dist_weight_state_dict, pretrain_world_size, self.world_size)
            dist_bias_state_dict = rearrange_weight(
                dist_bias_state_dict, pretrain_world_size, self.world_size)
            for name, value in dist_weight_state_dict.items():
                if rank_str in name:
                    dist_param_state_dict[name] = value
            for name, value in dist_bias_state_dict.items():
                if rank_str in name:
                    dist_param_state_dict[name] = value

            if for_train:
                dist_weight_velocity_state_dict = rearrange_weight(
                    dist_weight_velocity_state_dict, pretrain_world_size,
                    self.world_size)
                dist_bias_velocity_state_dict = rearrange_weight(
                    dist_bias_velocity_state_dict, pretrain_world_size,
                    self.world_size)
                for name, value in dist_weight_velocity_state_dict.items():
                    if rank_str in name:
                        opt_state_dict[name] = value
                for name, value in dist_bias_velocity_state_dict.items():
                    if rank_str in name:
                        opt_state_dict[name] = value

        def map_actual_param_name(state_dict, load_state_dict):
            for name, param in state_dict.items():
                state_dict[name] = load_state_dict[param.name]
            return state_dict

        logging.info("Load checkpoint from '{}'. ".format(checkpoint_dir))
        param_state_dict = map_actual_param_name(backbone.state_dict(),
                                                 param_state_dict)
        backbone.set_state_dict(param_state_dict)
        if classifier is not None:
            dist_param_state_dict = map_actual_param_name(
                classifier.state_dict(), dist_param_state_dict)
            classifier.set_state_dict(dist_param_state_dict)
        if for_train:
            assert optimizer is not None
            optimizer.set_state_dict(opt_state_dict)

        if classifier is not None and for_train:
            return extra_info
        else:
            return {}
