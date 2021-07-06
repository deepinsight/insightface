from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r1024":
        from .iresnet1024 import iresnet1024
        return iresnet1024(False, **kwargs)
    else:
        raise ValueError()
