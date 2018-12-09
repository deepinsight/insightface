import gluonbook as gb
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import loss as gloss, nn

from matplotlib import pyplot as plt

trains, tests = gb.load_data_fashion_mnist(256)

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(256, activation="relu"), gluon.nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {"learning_rate": 0.5})

gb.train_ch3
nd.concat()
for epoch in range(5):
    l_total = 0
    acc_total = 0
    for x, y in trains:
        with autograd.record():
            pre = net(x)
            l = loss(pre, y)
        l.backward()
        trainer.step(256)

        # print l.shape
        l_total = l_total + l.mean().asscalar()
        acc_total = acc_total + gb.accuracy(pre, y)

    acc = gb.evaluate_accuracy(tests, net)
    print("epoch %s l %s acc %s acc_test %s" % (epoch, l_total / len(trains), acc_total / len(trains), acc))
