import oneflow as flow
import oneflow.nn as nn
import os
from typing import List, Union


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        ofrecord_root: str = "./ofrecord",
        mode: str = "train",  # "val"
        dataset_size: int = 9469,
        batch_size: int = 1,
        total_batch_size: int = 1,
        data_part_num: int = 8,
        placement: flow.placement = None,
        sbp: Union[flow.sbp.sbp, List[flow.sbp.sbp]] = None,
    ):
        super().__init__()
        channel_last = False
        output_layout = "NHWC" if channel_last else "NCHW"
        assert (ofrecord_root, mode)
        self.train_record_reader = flow.nn.OfrecordReader(
            os.path.join(ofrecord_root, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=5,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
            placement=placement,
            sbp=sbp,
        )
        self.record_label_decoder = flow.nn.OfrecordRawDecoder(
            "label", shape=(), dtype=flow.int32
        )

        color_space = "RGB"
        height = 112
        width = 112

        self.record_image_decoder = flow.nn.OFRecordImageDecoder(
            "encoded", color_space=color_space
        )
        self.resize = (
            flow.nn.image.Resize(target_size=[height, width])
            if mode == "train"
            else flow.nn.image.Resize(
                resize_side="shorter", keep_aspect_ratio=True, target_size=112
            )
        )

        self.flip = (
            flow.nn.CoinFlip(batch_size=batch_size, placement=placement, sbp=sbp)
            if mode == "train"
            else None
        )

        rgb_mean = [127.5, 127.5, 127.5]
        rgb_std = [127.5, 127.5, 127.5]
        self.crop_mirror_norm = (
            flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
            if mode == "train"
            else flow.nn.CropMirrorNormalize(
                color_space=color_space,
                output_layout=output_layout,
                crop_h=0,
                crop_w=0,
                crop_pos_y=0.5,
                crop_pos_x=0.5,
                mean=rgb_mean,
                std=rgb_std,
                output_dtype=flow.float,
            )
        )

        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.total_batch_size

    def forward(self):
        train_record = self.train_record_reader()
        label = self.record_label_decoder(train_record)
        image_raw_buffer = self.record_image_decoder(train_record)

        image = self.resize(image_raw_buffer)[0]

        rng = self.flip() if self.flip != None else None
        image = self.crop_mirror_norm(image, rng)

        return image, label


class SyntheticDataLoader(flow.nn.Module):
    def __init__(
        self, batch_size, image_size=112, num_classes=10000, placement=None, sbp=None,
    ):
        super().__init__()

        self.image_shape = (batch_size, 3, image_size, image_size)
        self.label_shape = (batch_size,)
        self.num_classes = num_classes
        self.placement = placement
        self.sbp = sbp

        if self.placement is not None and self.sbp is not None:
            self.image = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=255,
                    size=self.image_shape,
                    dtype=flow.float32,
                    placement=self.placement,
                    sbp=self.sbp,
                ),
                requires_grad=False,
            )
            self.label = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=self.num_classes,
                    size=self.label_shape,
                    placement=self.placement,
                    sbp=self.sbp,
                ).to(dtype=flow.int32),
                requires_grad=False,
            )
        else:
            self.image = flow.randint(
                0, high=255, size=self.image_shape, dtype=flow.float32, device="cuda"
            )
            self.label = flow.randint(
                0, high=self.num_classes, size=self.label_shape, device="cuda",
            ).to(dtype=flow.int32)

    def __len__(self):
        return 10000

    def forward(self):
        return self.image, self.label
