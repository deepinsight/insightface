import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, num_feats=(512, 1), batch_norm=True, activation=True, fp16=False, coefficient=64., **kwargs):
        super(MLPHead, self).__init__()
        assert len(num_feats) >= 2

        self.fp16 = fp16
        self.activation = activation
        self.coefficient = coefficient

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
            if self.activation == "exp":
                x = torch.exp(x)
            if self.activation == "sigm":
                x = torch.sigmoid(x)
            if self.activation == "sigm_mul":
                x = self.coefficient * torch.sigmoid(x)
            if self.activation == "sigm_sum":
                x = self.coefficient + torch.sigmoid(x)
            if self.activation == "sigm_sum_mul":
                x = self.coefficient + self.coefficient * torch.sigmoid(x)
            if self.activation == "relu":
                x = self.coefficient * F.relu(x)
            if self.activation == "lin":
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
            x = self.lin(x)
            x = 0 * x + torch.ones_like(x) * self.constant
        return {"scale": x}
