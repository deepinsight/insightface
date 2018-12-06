import random

from mxnet import nd, autograd


def generate_data():
    nums = 1000
    w = [1, 2]
    b = 3
    features = nd.random.normal(scale=1, shape=(nums, 2))
    labels = nd.dot(features, nd.array(w)) + b
    labels += nd.random.normal(scale=0.01, shape=(nums,))
    labels = nd.expand_dims(labels, 1)
    return features, labels


def data_iter(features, labels, batch_size):
    nums = len(features)
    indices = list(range(nums))
    random.shuffle(indices)
    for i in range(0, nums, batch_size):
        j = nd.array(indices[i:min(i + batch_size, nums)])
        yield features.take(j), labels.take(j)


def linear_net(batch_features, w, b):
    return nd.dot(batch_features, w) + b


def squar_loss(batch_labels, labels):
    return (batch_labels - labels) ** 2 / 2


def batch_sgd(params, lr, batch_size):
    for param in params:
        param -= param.grad * lr / batch_size


w = nd.random.normal(scale=0.01, shape=(2, 1))
b = nd.zeros(shape=(1, 1))
w.attach_grad()
b.attach_grad()

batch_size = 10
lr = 0.03
net = linear_net
loss = squar_loss
sgd = batch_sgd
epoch_nums = 3

features, labels = generate_data()
for epoch in range(epoch_nums):
    for X, Y in data_iter(features, labels, batch_size):
        with autograd.record():
            l = loss(net(X, w, b), Y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print("epoch %s loss %s" % (epoch + 1, train_l.mean().asscalar()))

print w.asnumpy(), b.asnumpy()
