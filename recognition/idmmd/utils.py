import os
import numpy as np
import torch
import torch.nn.functional as F


def ort_loss(x, y):
    loss = torch.abs((x * y).sum(dim=1)).sum()
    loss = loss / float(x.size(0))
    return loss


def ang_loss(x, y):
    loss = (x * y).sum(dim=1).sum()
    loss = loss / float(x.size(0))
    return loss

def MMD_Loss(fc_nir, fc_vis):
    mean_fc_nir = torch.mean(fc_nir, 0)
    mean_fc_vis = torch.mean(fc_vis, 0)
    loss_mmd = F.mse_loss(mean_fc_nir, mean_fc_vis)
    return loss_mmd


def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)


def save_checkpoint(model, epoch, name="", dataset=''):
    if not os.path.exists("model/{}/".format(dataset)):
        os.makedirs("model/{}/".format(dataset))
    model_path = "model/{}/".format(dataset) + name + "_e{}.pth.tar".format(epoch)
    state = {"epoch": epoch, "state_dict": model.state_dict()}
    torch.save(state, model_path)
    print("checkpoint saved to {}".format(model_path))


def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()

    # print("to here")
    # print(model_dict.keys())
    # print('\n')

    # print(pretrained_dict.keys())

    # import pdb;pdb.set_trace()

    if 'LightCNN' in pretrained:
        tmp = [k for k in pretrained_dict]
        if "module." in tmp[0]:
            pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k!='module.weight'}
    
    print("len of params to be loaded: ",len(pretrained_dict))
    model.load_state_dict(pretrained_dict, strict=False)

    return weights['epoch']


def load_model_train_lightcnn(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()

    # print("to here")
    # print(model_dict.keys())
    # print('\n')

    # print(pretrained_dict.keys())

    # import pdb;pdb.set_trace()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'module.weight' not in k}
    
    print("len of params to be loaded: ",len(pretrained_dict))
    model.load_state_dict(pretrained_dict, strict=False)

    return weights['epoch']


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# assign adain_params to AdaIN layers
def assign_adain_params(adain_params, model):
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    correct = pred.eq(target.unsqueeze(0).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def adjust_learning_rate(lr, step, optimizer, epoch):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

