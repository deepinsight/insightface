import mxnet.optimizer as optimizer
from mxnet import ndarray as nd

class NoiseSGD(optimizer.SGD):
    """Noise SGD.


    This optimizer accepts the same arguments as :class:`.SGD`.
    """
    def __init__(self, scale, **kwargs):
        super(NoiseSGD, self).__init__(**kwargs)
        print('init noise sgd with', scale)
        self.scale = scale

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)
        noise = nd.random.normal(scale = self.scale, shape = grad.shape, dtype=grad.dtype, ctx = grad.context)
        grad += noise

        if state is not None:
            mom = state
            mom[:] *= self.momentum
            grad += wd * weight
            mom[:] += grad
            grad[:] += self.momentum * mom
            weight[:] += -lr * grad
        else:
            assert self.momentum == 0.0
            weight[:] += -lr * (grad + wd * weight)

