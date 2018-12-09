import gluonbook as gb
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import loss as gloss, nn

from matplotlib import pyplot as plt


x = nd.arange(-8, 8, 0.1)
x.attach_grad()
with autograd.record():
    y = x.tanh()
y.backward()

plt.plot(x.asnumpy(), y.asnumpy())
# plt.plot(x.asnumpy(), x.grad.asnumpy())
plt.show()
