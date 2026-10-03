from .face_analysis import *
try:
    import os

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from .mask_renderer import *
except ImportError:
    # The mask renderer depends on optional compiled face3d extensions. Keep the
    # main InsightFace API importable in source-tree and GUI-safe environments.
    pass


def __getattr__(name):
    if name == "PersonAnalysis":
        from .person_analysis import PersonAnalysis
        return PersonAnalysis
    raise AttributeError(name)
