import random

from matplotlib import pyplot as plt
from mxnet import nd, autograd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
labels = nd.expand_dims(labels, 1)


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


#
# set_figsize()
# plt.scatter(features[:, 0].asnumpy(), labels.asnumpy(), 1)
# plt.show()

def data_iter(batch_size, features, labels):
    nums = len(features)
    indices = list(range(nums))
    random.shuffle(indices)
    for i in range(0, nums, batch_size):
        j = nd.array(indices[i:min(i + batch_size, nums)])
        yield features.take(j), labels.take(j)


def linear_net(X, w, b):
    return nd.dot(X, w) + b


def squar_loss(y, true_y):
    return (y - true_y) ** 2 / 2


def batch_sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()
lr = 0.03
num_epochs = 3
net = linear_net
loss = squar_loss
batch_size = 10
sgd = batch_sgd

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print("epoch %s loss %f" % (epoch + 1, train_l.mean().asnumpy()))

print w, b
