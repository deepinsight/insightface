import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class AdaFaceLoss(torch.nn.Module):
    """Adaptive margin loss from AdaFace (https://arxiv.org/abs/2204.09949).

    Adjusts angular and additive margins per sample based on the feature norm,
    which serves as a proxy for image quality. High-norm samples receive a
    tighter margin; low-norm samples are penalised less.
    """

    def __init__(self, m: float = 0.4, h: float = 0.333,
                 s: float = 64.0, t_alpha: float = 0.01):
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = 1e-3
        self.register_buffer('batch_mean', torch.ones(1) * 20.0)
        self.register_buffer('batch_std',  torch.ones(1) * 100.0)

    def forward(self, logits: torch.Tensor, norms: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: cosine similarities, shape (N, C), values in (-1, 1)
            norms:  feature L2 norms, shape (N, 1)
            labels: class indices, shape (N, 1); -1 means no local positive class
        Returns:
            scaled logits with adaptive margin applied, shape (N, C)
        """
        index_positive = torch.where(labels.view(-1) != -1)[0]

        safe_norms = torch.clip(norms.view(-1), min=0.001, max=100).detach()

        with torch.no_grad():
            mean = safe_norms.mean()
            std  = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std  = std  * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = torch.clip(margin_scaler * self.h, -1, 1)

        # Angular margin
        g_angular = self.m * margin_scaler[index_positive] * -1
        m_arc = torch.zeros_like(logits)
        m_arc[index_positive, labels[index_positive].view(-1)] = g_angular
        theta   = logits.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        logits  = theta_m.cos()

        # Additive margin
        g_add = self.m + self.m * margin_scaler[index_positive]
        m_cos = torch.zeros_like(logits)
        m_cos[index_positive, labels[index_positive].view(-1)] = g_add
        logits = logits - m_cos

        return logits * self.s


def ehsm_sample_weight(patch_entropy: torch.Tensor, gamma: float = 1.0, topk: int = None) -> torch.Tensor:
    """Entropy-guided Hard Sample Mining (TransFace / TransFace++) sample weights.

    eta(x) = 1 + exp(-gamma * mean_topk(E(x_i))) : samples whose patch features carry
    little information (low entropy / std) are hard and get weights close to 2,
    easy ones close to 1.  patch_entropy: (B, P) -> (B, 1).
    """
    if topk is not None and topk < patch_entropy.shape[1]:
        patch_entropy = torch.topk(patch_entropy, k=topk, dim=1).values
    entropy = gamma * patch_entropy.mean(dim=1, keepdim=True)
    return (1 + torch.exp(-entropy)).detach()
