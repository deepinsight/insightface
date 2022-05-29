import tensorflow as tf
from external.landmark_detector import utils, models, data_provider

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

from external.landmark_detector.flags import FLAGS

# general framework
class DeepNetwork(object):
    def __init__(self):
        pass

    def _build_network(self, inputs, datas):
        pass

    def _build_losses(self, predictions, states, images, datas):
        pass

    def _build_summaries(self, predictions, states, images, datas):
        tf.summary.image('images', images[:, :, :, :3], max_outputs=min(FLAGS['batch_size'], 3))

    def _get_data(self):
        provider = data_provider.ProtobuffProvider(
            filename=FLAGS['dataset_dir'],
            batch_size=FLAGS['batch_size'],
            rescale=FLAGS['rescale'],
            augmentation=FLAGS['eval_dir']=='',
            )
        return provider.get()

    def _build_restore_fn(self, sess):
        init_fn = None

        if FLAGS['pretrained_model_checkpoint_path']:
            print('Loading whole model ...')
            variables_to_restore = slim.get_model_variables()
            init_fn =  slim.assign_from_checkpoint_fn(
                FLAGS['pretrained_model_checkpoint_path'],
                variables_to_restore,
                ignore_missing_vars=True)
        return init_fn


    def train(self):
        g = tf.Graph()
        logging.set_verbosity(10)

        with g.as_default():
            # Load datasets.

            images, *datas = self._get_data()
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                is_training=True):

                    predictions, states = self._build_network(images, datas)

                    # custom losses
                    self._build_losses(predictions, states, images, datas)

                    # total losses
                    total_loss = slim.losses.get_total_loss()
                    tf.summary.scalar('losses/total loss', total_loss)

                    # image summaries
                    self._build_summaries(predictions, states, images, datas)

                    # learning rate decay
                    global_step = slim.get_or_create_global_step()

                    learning_rate = tf.train.exponential_decay(
                        FLAGS['initial_learning_rate'],
                        global_step,
                        FLAGS['learning_rate_decay_step'] / FLAGS['batch_size'],
                        FLAGS['learning_rate_decay_factor'],
                        staircase=True)

                    tf.summary.scalar('learning rate', learning_rate)

                    optimizer = tf.train.AdamOptimizer(learning_rate)

        with tf.Session(graph=g) as sess:
            init_fn = self._build_restore_fn(sess)
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                summarize_gradients=True)

            logging.set_verbosity(1)

            slim.learning.train(train_op,
                FLAGS['train_dir'],
                save_summaries_secs=60,
                init_fn=init_fn,
                save_interval_secs=600)

class DNFaceMultiView(DeepNetwork):
    def __init__(self, n_lms=FLAGS['n_landmarks']):
        super(DNFaceMultiView, self).__init__()
        self.n_lms = n_lms


    def _get_data(self):
        provider = data_provider.ProtobuffProvider(
            filename=FLAGS['dataset_dir'],
            batch_size=FLAGS['batch_size'],
            rescale=FLAGS['rescale'],
            augmentation=FLAGS['eval_dir']=='',
            )
        return provider.get()


    def _build_network(self, inputs, datas=None, n_stacks=1, n_channels=FLAGS['n_landmarks'], is_training=True):
        # gt_heatmap, gt_lms, mask_index, gt_mask = datas

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        net = inputs

        # net = models.StackedHourglass(net, FLAGS.n_landmarks)
        # states.append(net)
        # net = tf.stop_gradient(net)
        # net *= gt_mask[:,None,None,:]
        # net = tf.concat([inputs,net], 3)
        # net = models.StackedHourglass(net, FLAGS.n_landmarks)
        # states.append(net)

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        states = []

        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=is_training):
            with slim.arg_scope(models.hourglass_arg_scope_tf()):
                net = None
                # stacked hourglass
                for i in range(n_stacks):
                    with tf.variable_scope('stack_%02d' % i):
                        if net is not None:
                            net = tf.concat((inputs, net), 3)
                        else:
                            net = inputs

                        net, _ = models.hourglass(
                            net,
                            regression_channels=n_channels,
                            classification_channels=0,
                            deconv='transpose',
                            bottleneck='bottleneck_inception')

                        states.append(net)

                prediction = net
                return prediction, states

    def _build_losses(self, predictions, states, images, datas):
        gt_heatmap, gt_lms, mask_index, gt_mask = datas

        weight_hm = utils.get_weight(gt_heatmap, tf.ones_like(gt_heatmap), ng_w=0.1, ps_w=1) * 500
        weight_hm *= gt_mask[:,None,None,:]

        l2norm = slim.losses.mean_squared_error(states[0], gt_heatmap, weights=weight_hm)

        tf.summary.scalar('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        super()._build_summaries(predictions, states, images, datas)

        gt_heatmap, gt_lms, mask_index, gt_mask = datas

        tf.summary.image('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_outputs=min(FLAGS['batch_size'],3))
