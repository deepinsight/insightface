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


def rearrange_weight(weight_dict, init_num_rank, new_num_rank):
    """
    A help function to convert pre-trained distributed fc parameters for
    inference or fine-tuning. Note that the number of ranks or GPUs for
    inference or fine-tuning can be different from that for pre-training.

    Args:
        weight_dict(dict): the dict store distributed parameters,
            key: eg. dist@fc@rank@00000.w_0
            value: numpy.ndarray
        init_num_rank(int) : pre-trained weight at init_num_rank gpu device.
        new_num_rank(int) : want to rearrange weight to new_num_rank gpu device.

    Returns:
        dict: rearranged weight for new_num_rank gpu device.
    """

    ret_dict = {}
    if init_num_rank == new_num_rank:
        return weight_dict

    if len(weight_dict) == 0:
        return weight_dict

    # generate name format
    name_format = list(weight_dict.keys())[0]
    name_format = name_format.split('.')
    name_format[0] = name_format[0].split('@')
    name_format[0][-1] = '%05d'
    name_format[0] = '@'.join(name_format[0])
    name_format = '.'.join(name_format)

    # calculate num class of pretrain shard
    # num class of new shard
    num_class = sum([
        w.shape[1] if len(w.shape) == 2 else len(w)
        for _, w in weight_dict.items()
    ])
    init_nshard = (num_class + init_num_rank - 1) // init_num_rank
    new_nshard = (num_class + new_num_rank - 1) // new_num_rank

    if new_nshard * (new_num_rank - 1) >= num_class:
        raise ValueError(
            "num class {} cann't be rationally splited by num rank {}".format(
                num_class, new_num_rank))

    if init_num_rank > new_num_rank:
        for new_idx in range(new_num_rank):
            start = new_idx * new_nshard
            end = min((new_idx + 1) * new_nshard - 1, num_class - 1)
            init_shard_idx_start = start // init_nshard
            init_shard_idx_end = end // init_nshard

            weight_list = []
            for init_idx in range(init_shard_idx_start,
                                  init_shard_idx_end + 1):
                name = name_format % init_idx
                init_weight = weight_dict[name]
                s = max(start - init_idx * init_nshard, 0)
                if init_idx == init_shard_idx_end:
                    e = min(end - init_idx * init_nshard + 1, init_nshard)
                else:
                    e = init_nshard
                if len(init_weight.shape) == 2:
                    weight_list.append(init_weight[:, s:e])
                else:
                    weight_list.append(init_weight[s:e])

            name = name_format % new_idx
            # for 2-dimention, we concat at axis=1,
            # else for 1-dimention, we concat at axis=0
            ret_dict[name] = np.concatenate(
                weight_list, axis=len(weight_list[0].shape) - 1)
    else:
        for new_idx in range(new_num_rank):
            start = new_idx * new_nshard
            end = min((new_idx + 1) * new_nshard - 1, num_class - 1)
            init_shard_idx_start = start // init_nshard
            init_shard_idx_end = end // init_nshard

            if init_shard_idx_start == init_shard_idx_end:
                name = name_format % init_shard_idx_start
                init_weight = weight_dict[name]
                init_start = init_shard_idx_start * init_nshard
                s = max(start - init_start, 0)
                e = min((init_shard_idx_start + 1) * init_nshard,
                        end) - init_start + 1
                if len(init_weight.shape) == 2:
                    new_weight = init_weight[:, s:e]
                else:
                    new_weight = init_weight[s:e]
            else:
                # init_shard_idx_start + 1 == init_shard_idx_end
                name = name_format % init_shard_idx_start
                init_weight = weight_dict[name]
                init_start = init_shard_idx_start * init_nshard
                s = max(start - init_start, 0)
                if len(init_weight.shape) == 2:
                    new_weight = init_weight[:, s:]
                else:
                    new_weight = init_weight[s:]

                e = end - (init_shard_idx_end * init_nshard) + 1
                if e > 0:
                    name = name_format % init_shard_idx_end
                    init_weight = weight_dict[name]
                    if len(init_weight.shape) == 2:
                        new_weight2 = init_weight[:, :e]
                    else:
                        new_weight2 = init_weight[:e]

                    new_weight = np.concatenate(
                        [new_weight, new_weight2],
                        axis=len(new_weight.shape) - 1)
            name = name_format % new_idx
            ret_dict[name] = new_weight

    return ret_dict
