from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


class TrainMetric(object):
    def __init__(
        self, desc="train", calculate_batches=1, batch_size=256,
    ):

        self.desc = desc
        self.calculate_batches = calculate_batches
        self.num_samples = calculate_batches * batch_size
        self.fmt = "{}: iter {}, loss {}, throughput: {:.3f}"

        self.timer = StopWatch()
        self.timer.start()

    def metric_cb(self, step):
        def callback(loss):

            if (step + 1) % self.calculate_batches == 0:
                throughput = self.num_samples / self.timer.split()

                print(
                    self.fmt.format(
                        self.desc, step, loss.mean(), throughput
                    )
                )

        return callback


class ValidationMetric(object):
    def __init__(self, desc="validation"):

        self.desc = desc
        self.fmt = "{}: time: {:.3f}"

        self.timer = StopWatch()
        self.timer.start()

    def metric_cb(self):
        def callback(metrics):

            time = self.timer.split()

            print(self.fmt.format(self.desc, time))

        return callback
