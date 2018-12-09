import gluonbook as gb
from mxnet import nd, autograd

num_input = 784
num_output = 10
batch_size = 256
lr = 0.1
epoches = 5

trains, tests = gb.load_data_fashion_mnist(batch_size)

W = nd.random.normal(scale=0.01, shape=(num_input, num_output))
b = nd.zeros(num_output)
W.attach_grad()
b.attach_grad()


def softmax(pre):
    exp = pre.exp()
    sum = exp.sum(axis=1, keepdims=True)
    return exp / sum


def net(x):
    pre = nd.dot(x.reshape((-1, num_input)), W) + b
    return softmax(pre)


def cross_entropy(softmax_pre, y):
    return -nd.pick(softmax_pre, y).log()


def accuracy(softmax_pre, y):
    return (softmax_pre.argmax(axis=1) == y.astype("float32")).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc_toatal = 0
    for x, y in data_iter:
        acc_toatal += accuracy(net(x), y)
    return acc_toatal / len(data_iter)


print evaluate_accuracy(tests, net)


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - param.grad * lr / batch_size


def train():
    for epoch in range(epoches):
        l_total = 0
        acc_total = 0
        for x, y in trains:
            with autograd.record():
                pre = net(x)
                l = cross_entropy(pre, y)
            l.backward()
            sgd([W, b], lr, batch_size)

            l_total = l_total + l.mean().asscalar()
            acc_total = acc_total + accuracy(pre, y)

        acc_test = evaluate_accuracy(tests, net)
        print("epoch %s l %s acc %s acc_test %s" % (epoch, l_total / len(trains), acc_total / len(trains), acc_test))


train()
