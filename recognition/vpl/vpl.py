import logging
import os

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class VPL(Module):
    """
    Modified from Partial-FC
    """

    @torch.no_grad()
    def __init__(self, rank, local_rank, world_size, batch_size, resume,
                 margin_softmax, num_classes, sample_rate=1.0, embedding_size=512, prefix="./", cfg=None):
        super(VPL, self).__init__()
        #
        assert sample_rate==1.0
        assert not resume
        self.num_classes: int = num_classes
        self.rank: int = rank
        self.local_rank: int = local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = world_size
        self.batch_size: int = batch_size
        self.margin_softmax: callable = margin_softmax
        self.sample_rate: float = sample_rate
        self.embedding_size: int = embedding_size
        self.prefix: str = prefix
        self.num_local: int = num_classes // world_size + int(rank < num_classes % world_size)
        self.class_start: int = num_classes // world_size * rank + min(rank, num_classes % world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, "rank_{}_softmax_weight.pt".format(self.rank))
        self.weight_mom_name = os.path.join(self.prefix, "rank_{}_softmax_weight_mom.pt".format(self.rank))

        self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
        self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
        logging.info("softmax weight init successfully!")
        logging.info("softmax weight mom init successfully!")
        self.stream: torch.cuda.Stream = torch.cuda.Stream(local_rank)

        self.index = None
        self.update = lambda: 0
        self.sub_weight = Parameter(self.weight)
        self.sub_weight_mom = self.weight_mom

        #vpl variables

        self._iters = 0
        self.cfg = cfg
        self.vpl_mode = -1
        if self.cfg is not None:
            self.vpl_mode = self.cfg['mode']
            if self.vpl_mode>=0:
                self.register_buffer("queue", torch.randn(self.num_local, self.embedding_size, device=self.device))
                self.queue = normalize(self.queue)
                self.register_buffer("queue_iters", torch.zeros((self.num_local,), dtype=torch.long, device=self.device))
                self.register_buffer("queue_lambda", torch.zeros((self.num_local,), dtype=torch.float32, device=self.device))


    def save_params(self):
        pass
        #torch.save(self.weight.data, self.weight_name)
        #torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        return index_positive

    def forward(self, total_features, norm_weight):
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            index_positive = self.sample(total_label)
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            return total_label, norm_weight, index_positive

    @torch.no_grad()
    def prepare_queue_lambda(self, label, iters):
        self.queue_lambda[:] = 0.0
        if iters>self.cfg['start_iters']:
            allowed_delta = self.cfg['allowed_delta']
            if self.vpl_mode==0:
                past_iters = iters - self.queue_iters
                idx = torch.where(past_iters <= allowed_delta)[0]
                self.queue_lambda[idx] = self.cfg['lambda']

            if iters % 2000 == 0 and self.rank == 0:
                logging.info('[%d]use-lambda: %d/%d'%(iters,len(idx), self.num_local))

    @torch.no_grad()
    def set_queue(self, total_features, total_label, index_positive, iters):
        local_label = total_label[index_positive]
        sel_features = normalize(total_features[index_positive,:])
        self.queue[local_label,:] = sel_features
        self.queue_iters[local_label] = iters

    def forward_backward(self, label, features, optimizer, feature_w):
        self._iters += 1
        total_label, norm_weight, index_positive = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True

        if feature_w is not None:
            total_feature_w = torch.zeros(
                size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
            dist.all_gather(list(total_feature_w.chunk(self.world_size, dim=0)), feature_w.data)

        if self.vpl_mode>=0:
            self.prepare_queue_lambda(total_label, self._iters)
            _lambda = self.queue_lambda.view(self.num_local, 1)
            injected_weight = norm_weight*(1.0-_lambda) + self.queue*_lambda
            injected_norm_weight = normalize(injected_weight)
            logits = self.forward(total_features, injected_norm_weight)
        else:
            logits = self.forward(total_features, norm_weight)

        logits = self.margin_softmax(logits, total_label)

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size
        #vpl set queue
        if self.vpl_mode>=0:
            if feature_w is None:
                self.set_queue(total_features.detach(), total_label, index_positive, self._iters)
            else:
                self.set_queue(total_feature_w, total_label, index_positive, self._iters)
        # backward backbone
        return x_grad, loss_v

