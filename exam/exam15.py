from matplotlib import pyplot as plt
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# y = 1.2*x - 3.4*x**2 + 5.6*x**3 + 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = poly_features[:, 0] * true_w[0] + poly_features[:, 1] * true_w[1] + poly_features[:, 2] * true_w[2] + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)
print poly_features.shape, labels.shape


def semilogy(x_vals, y_vals, x2_vals, y2_values):
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.semilogy(x_vals, y_vals)
    plt.semilogy(x2_vals, y2_values)
    plt.legend(['train', 'test'])
    plt.show()


num_epochs, loss = 100, gloss.L2Loss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()

    batch_size = 10
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    # train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)

        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())

    semilogy(range(1, num_epochs + 1), train_ls, range(1, num_epochs + 1), test_ls)


# fit_and_plot(poly_features[:100, :], poly_features[100:, :], labels[:100], labels[100:])
fit_and_plot(features[:100, :], features[100:, :], labels[:100], labels[100:])
