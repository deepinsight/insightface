from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d, ResNetV1e
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet

from .mobilenet import MobileNetV1

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNetV1e', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'ResNeSt', 'TridentResNet', 'MobileNetV1'
]
