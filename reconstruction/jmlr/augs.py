import numpy as np
import cv2
import os
import os.path as osp
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

class RectangleBorderAugmentation(ImageOnlyTransform):

    def __init__(
            self,
            fill_value = 0,
            fg_limit = (0.7, 0.9),
            always_apply=False,
            p=1.0,
            ):
        super(RectangleBorderAugmentation, self).__init__(always_apply, p)
        #assert limit>0.0 and limit<1.0
        assert isinstance(fg_limit, tuple)
        assert fg_limit[1]>fg_limit[0]
        self.fill_value = 0
        self.fg_limit = fg_limit
        #self.output_size = output_size


    def apply(self, image, fg, top, left, **params):
        assert image.shape[0]==image.shape[1]
        oimage = np.ones_like(image) * self.fill_value
        f = int(fg*image.shape[0])
        t = int(top*image.shape[0])
        l = int(left*image.shape[1])
        oimage[t:t+f,l:l+f,:] = image[t:t+f,l:l+f,:]
        return oimage

    def get_params(self):
        fg = np.random.uniform(self.fg_limit[0], self.fg_limit[1])
        top = np.random.uniform(0.0, 1.0-fg)
        left = np.random.uniform(0.0, 1.0-fg)
        return {'fg': fg, 'top': top, 'left': left}

    def get_transform_init_args_names(self):
        return ('fill_value','fg_limit')

class SunGlassAugmentation(ImageOnlyTransform):

    def __init__(
            self,
            fill_value = 0,
            loc = [ (38, 52), (73, 52) ],
            rad_limit = (10, 20),
            always_apply=False,
            p=1.0,
            ):
        super(SunGlassAugmentation, self).__init__(always_apply, p)
        #assert limit>0.0 and limit<1.0
        assert isinstance(rad_limit, tuple)
        self.fill_value = 0
        self.loc = loc
        self.rad_limit = rad_limit


    def apply(self, image, rad, **params):
        for i in range(2):
            cv2.circle(image, self.loc[i], rad, self.fill_value, -1)
        return image

    def get_params(self):
        rad = np.random.randint(self.rad_limit[0], self.rad_limit[1])
        return {'rad':rad}

    def get_transform_init_args_names(self):
        return ('fill_value', 'loc', 'rad_limit')

class ForeHeadAugmentation(ImageOnlyTransform):

    def __init__(
            self,
            height_min = 0.2,
            height_max = 0.4,
            width_min = 0.5,
            always_apply=False,
            p=1.0,
            ):
        super(ForeHeadAugmentation, self).__init__(always_apply, p)
        assert height_max > height_min
        #assert limit>0.0 and limit<1.0
        self.height_min = height_min
        self.height_max = height_max
        self.width_min = width_min


    def apply(self, image, height, width, left, **params):
        mask_value = np.random.randint(0, 255, size=(int(image.shape[0]*height), int(image.shape[1]*width), 3), dtype=image.dtype)
        l = int(image.shape[1]*left)
        image[:mask_value.shape[0], l:l+mask_value.shape[1], :] = mask_value
        return image

    def get_params(self):
        height = np.random.uniform(self.height_min, self.height_max)
        width = np.random.uniform(self.width_min, 1.0)
        left = np.random.uniform(0.0, 1.0 - width)
        return {'height': height, 'width': width, 'left': left}

    def get_transform_init_args_names(self):
        return ('height_min', 'height_max','width_min')


def get_aug_transform(cfg):
    aug_modes = cfg.aug_modes
    input_size = cfg.input_size
    task = cfg.task
    transform_list = []
    is_test = False
    if 'test-aug' in aug_modes:
        #transform_list.append(
        #    A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
        #    )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0, always_apply=True)
            )
        is_test = True

    if '1' in aug_modes:
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
            )
    if '1A' in aug_modes:
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.03, rotate_limit=6, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3)
            )
    if '2' in aug_modes:
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4)
            )
    if '3' in aug_modes:
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.6)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6)
            )
    if 'nist1' in aug_modes:
        transform_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.06, rotate_limit=6, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4)
            )
    if 'nist2' in aug_modes:
        transform_list.append(
            #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.3)
            A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.05, p=0.2)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=6, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4)
            )
        transform_list.append(
                A.OneOf([
                    RectangleBorderAugmentation(p=0.5),
                    ForeHeadAugmentation(p=0.5),
                    #SunGlassAugmentation(p=0.2),
                    ], p=0.06)
                )
        transform_list.append(
            A.ToGray(p=0.05)
            )
        transform_list.append(
            A.geometric.resize.RandomScale(scale_limit=(0.7, 0.9), interpolation=cv2.INTER_LINEAR, p=0.05)
            )
        transform_list.append(
            A.ISONoise(p=0.06)
            )
        transform_list.append(
            A.MedianBlur(blur_limit=(1,7), p=0.05)
            )
        transform_list.append(
            A.MotionBlur(blur_limit=(5,12), p=0.05)
            )
        transform_list.append(
            A.ImageCompression(quality_lower=50, quality_upper=80, p=0.05)
            )
    if 'prod' in aug_modes:
        transform_list.append(
            #A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.125, p=0.2)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.3)
            )
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=10, interpolation=cv2.INTER_LINEAR, 
                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.4)
            )
        transform_list.append(
                A.OneOf([
                    RectangleBorderAugmentation(p=0.5),
                    ForeHeadAugmentation(p=0.5),
                    MaskAugmentation(mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'], mask_probs=[0.4, 0.4, 0.1, 0.1], h_low=0.33, h_high=0.4, p=0.2),
                    SunGlassAugmentation(p=0.2),
                    ], p=0.2)
                )
        transform_list.append(
            A.ToGray(p=0.05)
            )
        transform_list.append(
            A.geometric.resize.RandomScale(scale_limit=(0.6, 0.9), interpolation=cv2.INTER_LINEAR, p=0.2)
            )
        transform_list.append(
            A.ISONoise(p=0.1)
            )
        transform_list.append(
            A.MedianBlur(blur_limit=(1,7), p=0.1)
            )
        transform_list.append(
            A.MotionBlur(blur_limit=(5,12), p=0.1)
            )
        transform_list.append(
            A.ImageCompression(quality_lower=30, quality_upper=80, p=0.1)
            )
    #if input_size!=112: # TODO!!
    #    transform_list.append(
    #        A.geometric.resize.Resize(input_size, input_size, interpolation=cv2.INTER_LINEAR, always_apply=True)
    #        )
    transform_list += \
        [
            #A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
    #here, the input for A transform is rgb cv2 img
    if is_test:
        transform = A.ReplayCompose(
            transform_list ,
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
        )
    else:
        transform = A.Compose(
            transform_list,
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
        )
    return transform


if __name__ == "__main__":
    tool = MaskRenderer()
    tool.prepare(ctx_id=0, det_size=(128,128))
    image = cv2.imread("./test1.png")[:,:,::-1]
    mask_image  = "mask_blue"
    #params = tool.build_params(image)
    label = np.load('assets/mask_label.npy')
    params = tool.decode_params(label)
    #print(params[0][:20])
    mask_out = tool.render_mask(image, mask_image, params, input_is_rgb=True, auto_blend=False)[:,:,::-1]
    #print(uv_out.dtype, uv_out.shape)
    cv2.imwrite('output_mask.jpg', mask_out)
    transform = A.Compose([
        MaskAugmentation(mask_names=['mask_white', 'mask_blue', 'mask_black', 'mask_green'], mask_probs=[0.4, 0.4, 0.1, 0.1], h_low=0.33, h_high=0.4, p=1.0),
        #MaskAugmentation(p=1.0),
        ])
    mask_out = transform(image=image, hlabel=label)["image"][:,:,::-1]
    cv2.imwrite('output_mask2.jpg', mask_out)
