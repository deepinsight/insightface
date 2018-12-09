from mxnet import autograd, nd
from mxnet import gluon
import mxnet as mx


def generate_data():
    nums = 1000
    w = nd.array([1, 2]).reshape((2, 1))
    b = nd.ones(shape=(1,))
    features = nd.random.normal(scale=1, shape=(nums, 2))
    labels = nd.dot(features, w) + b
    return features, labels


batch_size = 10
features, labels = generate_data()
dataset = gluon.data.ArrayDataset(features, labels)
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize(mx.init.Normal(0.01))

loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), "sgd", {'learning_rate': 0.03})

for epoch in range(3):
    for X, Y in dataloader:
        with autograd.record():
            l = loss(net(X), Y)
        l.backward()
        trainer.step(batch_size)
    train_l = loss(net(features), labels)
    print("epoch %s l %s" %(epoch, train_l.mean().asscalar()))

for param in net.collect_params().values():
    print param.data()

