import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class RectangleBorderAugmentation(ImageOnlyTransform):

    def __init__(
            self,
            fill_value = 0,
            limit = 0.3,
            always_apply=False,
            p=1.0,
            ):
        super(RectangleBorderAugmentation, self).__init__(always_apply, p)
        assert limit>0.0 and limit<1.0
        self.fill_value = 0
        self.limit = limit


    def apply(self, image, border_size_limit, **params):
        assert len(border_size_limit)==4
        border_size = border_size_limit.copy()
        border_size[0] *= image.shape[1]
        border_size[2] *= image.shape[1]
        border_size[1] *= image.shape[0]
        border_size[3] *= image.shape[0]
        border_size = border_size.astype(np.int)
        image[:,:border_size[0],:] = self.fill_value
        image[:border_size[1],:,:] = self.fill_value
        image[:,-border_size[2]:,:] = self.fill_value
        image[-border_size[3]:,:,:] = self.fill_value
        return image

    def get_params(self):
        border_size_limit = np.random.uniform(0.0, self.limit, size=4)
        return {'border_size_limit': border_size_limit}

    def get_transform_init_args_names(self):
        return ('fill_value', 'limit')

