import gluonbook as gb
from mxnet import nd, autograd

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


print evaluate_accuracy(test_iter, net)
num_epochs, lr = 5, 0.1


print len(train_iter)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
          batch_size, [W, b], lr)

# print evaluate_accuracy(test_iter, net)

# for X, y in test_iter:
#     true_labels = gb.get_fashion_mnist_labels(y.asnumpy())
#     pred_labels = gb.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
#     titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
#
#     gb.show_fashion_mnist(X[0:9], titles[0:9])
#     break
