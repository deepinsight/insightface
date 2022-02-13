import mxnet as mx
import mxnet.optimizer as optimizer
from mxnet.ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as
                           NDabs)
#from mxnet.ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
#                      mp_sgd_update, mp_sgd_mom_update, square, ftrl_update)


class ONadam(optimizer.Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 schedule_decay=0.004,
                 **kwargs):
        super(ONadam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.m_schedule = 1.

    def create_state(self, index, weight):
        return (
            zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
            zeros(weight.shape, weight.context,
                  dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        #grad = grad * self.rescale_grad + wd * weight
        grad *= self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # warming momentum schedule
        momentum_t = self.beta1 * (1. - 0.5 *
                                   (pow(0.96, t * self.schedule_decay)))
        momentum_t_1 = self.beta1 * (1. - 0.5 *
                                     (pow(0.96,
                                          (t + 1) * self.schedule_decay)))
        self.m_schedule = self.m_schedule * momentum_t
        m_schedule_next = self.m_schedule * momentum_t_1

        # update m_t and v_t
        m_t, v_t = state
        m_t[:] = self.beta1 * m_t + (1. - self.beta1) * grad
        v_t[:] = self.beta2 * v_t + (1. - self.beta2) * grad * grad

        grad_prime = grad / (1. - self.m_schedule)
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t_prime = v_t / (1. - pow(self.beta2, t))
        m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

        # update weight
        weight[:] -= lr * m_t_bar / (sqrt(v_t_prime) + self.epsilon)
