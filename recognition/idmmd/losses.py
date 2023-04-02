import torch
from torch import nn
import torch.nn.functional as F


class IDMMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(IDMMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def get_centers_by_id(self, x_rgb, x_ir, targets):
        centers_rgb = []
        centers_ir = []

        batch_y_set = set(targets.data.cpu().numpy())

        for _, l in enumerate(batch_y_set):
            feat1 = x_rgb[targets==l]
            feat2 = x_ir[targets==l]

            centers_rgb.append(feat1.mean(dim=0).unsqueeze(0))
            centers_ir.append(feat2.mean(dim=0).unsqueeze(0))

        centers_rgb = torch.cat(centers_rgb, 0).cuda()
        centers_ir = torch.cat(centers_ir, 0).cuda()

        return centers_rgb, centers_ir
    
    def forward(self, x_rgb, x_ir, targets):

        centers_rgb, centers_ir = self.get_centers_by_id(x_rgb, x_ir, targets)

        if self.kernel_type == 'linear':
            loss = self.linear_mmd(centers_rgb, centers_ir)         # domain-level loss

        elif self.kernel_type == 'rbf':
            B = centers_rgb.size(0)
            kernels = self.guassian_kernel(centers_rgb, centers_ir)

            XX = kernels[:B, :B]
            YY = kernels[B:, B:]
            XY = kernels[:B, B:]
            YX = kernels[B:, :B]

            loss = (XX + YY - XY - YX).mean()

        return loss
        

    def linear_mmd(self, center_rgb, center_ir):
        def compute_dist_(x_rgb, x_ir):
            n = x_rgb.size(0)
            dist1 = torch.pow(x_rgb, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist2 = torch.pow(x_ir, 2).sum(dim=1, keepdim=True).expand(n, n)
            
            dist = dist1 + dist2.t()
            dist.addmm_(mat1=x_rgb, mat2=x_ir.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12)  # for numerical stability
            return dist

        matrix = compute_dist_(center_rgb, center_ir)
        loss = matrix.diag()
        
        return loss.mean()


    def guassian_kernel(self, x_rgb, x_ir):
        total = torch.cat([x_rgb, x_ir], dim=0)
        N = total.size(0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        dists = ((total0-total1)**2).sum(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(dists.data) / (N**2-N)
        
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i)
                          for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-dists / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)



class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        one_hot = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), 1.0).cuda()
        phi = logits - self.m
        output = torch.where(one_hot==1, phi, logits)
        output *= self.s

        return output