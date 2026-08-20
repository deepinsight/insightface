from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    norm_output = kwargs.get("norm_output", False)
    # resnet
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
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features, norm_output=norm_output)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features, norm_output=norm_output)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, norm_output=norm_output)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0, norm_output=norm_output)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, norm_output=norm_output)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0, norm_output=norm_output)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True, norm_output=norm_output)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True, norm_output=norm_output)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True, norm_output=norm_output)
        
    elif name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True, norm_output=norm_output)

    elif name in ("mambavision_t", "mambavision_s", "mambavision_b", "mambavision_l"):
        from . import mamba_vision
        return getattr(mamba_vision, name)(
            num_features=kwargs.get("num_features", 512),
            fp16=kwargs.get("fp16", False),
            norm_output=norm_output)

    elif name in ("transface_pp_vit_s", "transface_pp_vit_b"):
        from . import transface_pp
        byte_format = kwargs.get("byte_format", "tiff")
        return getattr(transface_pp, name)(
            num_bytes=transface_pp.num_bytes_for(byte_format),
            num_classes=kwargs.get("num_features", 512),
            use_topology=kwargs.get("use_topology", True),
            tibc_prob=kwargs.get("tibc_prob", 0.3),
            fp16=kwargs.get("fp16", False),
            norm_output=norm_output)

    elif name == "transface_vit_b":
        num_features = kwargs.get("num_features", 512)
        from .transface_vit import TransFaceViT
        return TransFaceViT(
            img_size=112, patch_size=9, num_classes=num_features,
            embed_dim=512, depth=24, num_heads=8,
            drop_path_rate=0.05, norm_layer="ln",
            mask_ratio=0.05, using_checkpoint=True, norm_output=norm_output)

    elif name == "transface_vit_l":
        num_features = kwargs.get("num_features", 512)
        from .transface_vit import TransFaceViT
        return TransFaceViT(
            img_size=112, patch_size=9, num_classes=num_features,
            embed_dim=768, depth=24, num_heads=8,
            drop_path_rate=0.05, norm_layer="ln",
            mask_ratio=0.05, using_checkpoint=True, norm_output=norm_output)

    else:
        raise ValueError()
