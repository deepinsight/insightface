import oneflow as flow
import oneflow.nn as nn


def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )


def meter(self, mkey, *args):
    assert mkey in self.m
    self.m[mkey]["meter"].record(*args)


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        cfg,
        combine_margin,
        cross_entropy,
        data_loader,
        optimizer,
        lr_scheduler=None,
    ):
        super().__init__()

        if cfg.fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(make_grad_scaler())
        elif cfg.scale_grad:
            self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.model = model

        self.cross_entropy = cross_entropy
        self.combine_margin = combine_margin
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        image, label = self.data_loader()

        image = image.to("cuda")
        label = label.to("cuda")

        logits, label = self.model(image, label)
        logits = self.combine_margin(logits, label) * 64
        loss = self.cross_entropy(logits, label)

        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, cfg):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)
        self.model = model
        if cfg.fp16:
            self.config.enable_amp(True)

    def build(self, image):
        logits = self.model(image)
        return logits
