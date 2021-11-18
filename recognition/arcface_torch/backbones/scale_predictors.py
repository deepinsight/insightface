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
            if self == "exp":
                x = torch.exp(x)
            if self == "sigm":
                x = torch.sigmoid(x)
            if self == "lin":
                x = x
        return {"scale": x}
