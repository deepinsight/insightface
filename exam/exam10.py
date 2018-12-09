import gluonbook as gb
from mxnet import nd, autograd

num_classes = 10
num_features = 784
batch_size = 256


def load_data(batch_size):
    return gb.load_data_fashion_mnist(batch_size)


trains, testes = load_data(batch_size)
W = nd.random.normal(scale=0.01, shape=(num_features, num_classes))
b = nd.zeros(num_classes)
W.attach_grad()
b.attach_grad()


def softmax(pre_y):
    exp = pre_y.exp()
    # TODO keep
    print exp, exp.sum(axis=1)
    return exp / exp.sum(axis=1, keepdims=True)


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_features)), W) + b)


def cross_entropy(pre_y, true_y):
    return -nd.pick(pre_y, true_y).log()


def accuracy(pre, y):
    return (pre.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    accuracyes = 0
    for x, y in data_iter:
        pre = net(x)
        accuracyes += accuracy(pre, y)
    mean = accuracyes / len(data_iter)
    return mean


print evaluate_accuracy(testes, net)


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def train():
    for epoch in range(5):
        train_l_sum = 0
        train_acc_sum = 0
        for x, y in trains:
            with autograd.record():
                pre = net(x)
                l = cross_entropy(pre, y)
            l.backward()
            sgd([W, b], 0.1, batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(pre, y)

        test_acc = evaluate_accuracy(testes, net)
        print(" epoch %s l_sum %s acc_sum %s test_acc %s" % (
            epoch, train_l_sum / len(trains), train_acc_sum / len(trains), test_acc))


train()
