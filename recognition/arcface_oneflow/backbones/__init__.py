from .ir_resnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .fmobilefacenet import mobilefacenet


def get_model(name, input_blob, cfg):
    if name == "r18":
        return iresnet18(input_blob, cfg)
    elif name == "r34":
        return iresnet34(input_blob, cfg)
    elif name == "r50":
        return iresnet50(input_blob, cfg)
    elif name == "r100":
        return iresnet100(input_blob, cfg)
    elif name == "r200":
        return iresnet200(input_blob, cfg)
    elif name == "mbf":
        return mobilefacenet(input_blob, cfg)
    else:
        raise ValueError()
