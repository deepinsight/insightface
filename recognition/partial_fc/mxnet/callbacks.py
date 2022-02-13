import logging
import os
import sys
import time

import horovod.mxnet as hvd
import mxnet as mx
from mxboard import SummaryWriter
from mxnet import nd

from default import config
from evaluation import verification


class MetricNdarray(object):
    def __init__(self):
        self.sum = None
        self.count = 0
        self.reset()

    def reset(self):
        self.sum = None
        self.count = 0

    def update(self, val, n=1):
        assert isinstance(val, mx.nd.NDArray), type(val)
        if self.sum is None:  # init sum
            self.sum = mx.nd.zeros_like(val)

        self.sum += val * n
        self.count += n

    def get(self):
        average = self.sum / self.count
        return average.asscalar()


class CallBackVertification(object):
    def __init__(self, symbol, model):
        self.verbose = config.verbose
        self.symbol = symbol
        self.highest_acc = 0.0
        self.highest_acc_list = [0.0] * len(config.val_targets)
        self.model = model
        self.ver_list = []
        self.ver_name_list = []
        self.init_dataset(val_targets=config.val_targets,
                          data_dir=os.path.dirname(config.rec),
                          image_size=(config.image_size, config.image_size))

    def ver_test(self, num_update):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], self.model, 10, 10, None, None)
            logging.info('[%s][%d]XNorm: %f' %
                         (self.ver_name_list[i], num_update, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' %
                         (self.ver_name_list[i], num_update, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' %
                (self.ver_name_list[i], num_update, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, param):
        #
        num_update = param.num_update
        #
        if num_update > 0 and num_update % self.verbose == 0:  # debug in mbatches in 100 and 200
            # accuracy list
            self.ver_test(num_update)


class CallBackCenterSave(object):
    def __init__(self, memory_bank, save_interval=10000):
        self.save_interval = save_interval
        self.memory_bank = memory_bank

    def __call__(self, param):
        if param.num_update % self.save_interval == 0:
            self.memory_bank.save()


class CallBackModelSave(object):
    def __init__(self, symbol, model, prefix, rank):
        self.symbol = symbol
        self.model = model
        self.prefix = prefix
        self.max_step = config.max_update
        self.rank = rank

    def __call__(self, param):
        num_update = param.num_update

        if num_update in [
            self.max_step - 10,
        ] or (num_update % 10000 == 0 and num_update > 0):

            # params
            arg, aux = self.model.get_export_params()
            # symbol
            _sym = self.symbol
            # save

            # average all aux
            new_arg, new_aux = {}, {}
            for key, tensor in aux.items():
                new_aux[key] = hvd.allreduce(tensor, average=True)
            for key, tensor in arg.items():
                new_arg[key] = hvd.allreduce(tensor, average=True)

            if self.rank == 0:
                mx.model.save_checkpoint(prefix=self.prefix + "_average",
                                         epoch=0,
                                         symbol=_sym,
                                         arg_params=new_arg,
                                         aux_params=new_aux)
                mx.model.save_checkpoint(prefix=self.prefix,
                                         epoch=0,
                                         symbol=_sym,
                                         arg_params=arg,
                                         aux_params=aux)

        # training is over
        if num_update > self.max_step > 0:
            logging.info('Training is over!')
            sys.exit(0)


class MetricCallBack(object):
    def __init__(self, batch_size, rank, size, prefix_dir, frequent):
        self.batch_size = batch_size
        self.rank = rank
        self.size = size
        self.prefix_dir = prefix_dir
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.loss_metric_list = MetricNdarray()
        t = time.localtime()

        self.summary_writer = SummaryWriter(
            logdir=os.path.join(self.prefix_dir, 'log_tensorboard', str(t.tm_mon) + '_' + str(t.tm_mday) \
                                + '_' + str(t.tm_hour)),
            verbose=False)
        pass


class CallBackLogging(object):
    def __init__(self, rank, size, prefix_dir):
        self.batch_size = config.batch_size
        self.rank = rank
        self.size = size
        self.prefix_dir = prefix_dir
        self.frequent = config.frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.loss_metric = MetricNdarray()
        t = time.localtime()

        if self.rank == 0:
            self.summary_writer = SummaryWriter(logdir=os.path.join(
                self.prefix_dir, "log_tensorboard",
                "%s_%s_%s" % (str(t.tm_mon), str(t.tm_mday), str(t.tm_hour))),
                verbose=False)
        else:
            time.sleep(2)

    def __call__(self, param):
        """Callback to Show speed
        """
        count = param.num_update

        if self.last_count > count:
            self.init = False
        self.last_count = count

        self.loss_metric.update(param.loss[0])

        if self.init:
            if count % self.frequent == 0:
                nd.waitall()
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.size
                except ZeroDivisionError:
                    speed = float('inf')
                    speed_total = float('inf')

                # summary loss
                loss_scalar = self.loss_metric.get()

                if self.rank == 0:
                    self.summary_writer.add_scalar(tag="loss", value=loss_scalar, global_step=param.num_update)
                loss_str_format = "[%d][%s]:%.2f " % (param.num_epoch, "loss",
                                                      loss_scalar)
                self.loss_metric.reset()

                if self.rank == 0:
                    self.summary_writer.add_scalar(tag="speed", value=speed, global_step=param.num_update)
                    self.summary_writer.flush()
                    logging.info(
                        "Iter:%d Rank:%.2f it/sec Total:%.2f it/sec %s",
                        param.num_update, speed, speed_total, loss_str_format)

                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
