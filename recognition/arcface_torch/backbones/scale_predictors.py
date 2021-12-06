import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, num_feats=(512, 1), batch_norm=True, exponent=True, fp16=False, **kwargs):
        super(MLPHead, self).__init__()
        assert len(num_feats) >= 2

        self.fp16 = fp16
        self.exponent = exponent

        in_feats = num_feats[0]

        self.mlp = []
        for out_feats in num_feats[1:-1]:
            self.mlp.append(nn.Linear(in_feats, out_feats, bias=False))
            nn.init.kaiming_normal_(self.mlp[-1].weight)

            if batch_norm:
                self.mlp.append(nn.BatchNorm1d(out_feats, affine=False))

            self.mlp.append(nn.ReLU())
            in_feats = out_feats

        out_feats = num_feats[-1]
        self.mlp.append(nn.Linear(in_feats, out_feats, bias=False))

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        with torch.cuda.amp.autocast(self.fp16):
            x = self.mlp(x)
            if self.exponent == "exp":
                x = torch.exp(x)
            if self.exponent == "sigm":
                x = torch.sigmoid(x)
            if self.exponent == "sigm_mul":
                x = 64 * torch.sigmoid(x)
            if self.exponent == "sigm_sum":
                x = 64 + torch.sigmoid(x)
            if self.exponent == "sigm_sum_mul":
                x = 32 + 32 * torch.sigmoid(x)
            if self.exponent == "lin":
                x = x
        return {"scale": x}


class DummyHead(nn.Module):
    def __init__(self, fp16=False,  **kwargs):
        super(DummyHead, self).__init__()

        self.lin = nn.Linear(25088, 1)
        self.constant = 64.
        self.fp16 = fp16

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        with torch.cuda.amp.autocast(self.fp16):
            # x = self.mlp(x)
            # if self == "exp":
            #     x = torch.exp(x)
            # if self == "sigm":
            #     x = torch.sigmoid(x)
            # if self == "lin":
            #     x = x

            x = self.lin(x)
            x = 0 * x + torch.ones_like(x) * self.constant
        return {"scale": x}
