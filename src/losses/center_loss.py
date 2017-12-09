import os

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'
import mxnet as mx


# define metric of accuracy
class Accuracy(mx.metric.EvalMetric):
    def __init__(self, num=None):
        super(Accuracy, self).__init__('accuracy', num)

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num is not None:
            assert len(labels) == self.num

        pred_label = mx.nd.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy().astype('int32')

        mx.metric.check_label_shapes(label, pred_label)

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


# define some metric of center_loss
class CenterLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(CenterLossMetric, self).__init__('center_loss')

    def update(self, labels, preds):
        self.sum_metric += preds[1].asnumpy()[0]
        self.num_inst += 1


# see details:
# <A Discriminative Feature Learning Approach for Deep Face Recogfnition>
class CenterLoss(mx.operator.CustomOp):
    def __init__(self, ctx, shapes, dtypes, num_class, alpha, scale=1.0):
        if not len(shapes[0]) == 2:
            raise ValueError('dim for input_data shoudl be 2 for CenterLoss')

        self.alpha = alpha
        self.batch_size = shapes[0][0]
        self.num_class = num_class
        self.scale = scale

    def forward(self, is_train, req, in_data, out_data, aux):
        labels = in_data[1].asnumpy()
        diff = aux[0]
        center = aux[1]

        # store x_i - c_yi
        for i in range(self.batch_size):
            diff[i] = in_data[0][i] - center[int(labels[i])]

        loss = mx.nd.sum(mx.nd.square(diff)) / self.batch_size / 2
        self.assign(out_data[0], req[0], loss)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        diff = aux[0]
        center = aux[1]
        sum_ = aux[2]

        # back grad is just scale * ( x_i - c_yi)
        grad_scale = float(self.scale/self.batch_size)
        self.assign(in_grad[0], req[0], diff * grad_scale)

        # update the center
        labels = in_data[1].asnumpy()
        label_occur = dict()
        for i, label in enumerate(labels):
            label_occur.setdefault(int(label), []).append(i)

        for label, sample_index in label_occur.items():
            sum_[:] = 0
            for i in sample_index:
                sum_ = sum_ + diff[i]
            delta_c = sum_ / (1 + len(sample_index))
            center[label] += self.alpha * delta_c


@mx.operator.register("centerloss")
class CenterLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_class, alpha, scale=1.0, batchsize=64):
        super(CenterLossProp, self).__init__(need_top_grad=False)

        # convert it to numbers
        self.num_class = int(num_class)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.batchsize = int(batchsize)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        # call them 'bias' for zero initialization
        return ['diff_bias', 'center_bias', 'sum_bias']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)

        # store diff , same shape as input batch
        diff_shape = [self.batchsize, data_shape[1]]

        # store the center of each class , should be ( num_class, d )
        center_shape = [self.num_class, diff_shape[1]]

        # computation buf
        sum_shape = [diff_shape[1],]

        output_shape = [1, ]
        return [data_shape, label_shape], [output_shape], [diff_shape, center_shape, sum_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CenterLoss(ctx, shapes, dtypes, self.num_class, self.alpha, self.scale)
