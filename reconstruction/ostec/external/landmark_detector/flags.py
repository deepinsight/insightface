import tensorflow as tf

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0, '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97, '''Learning rate decay factor.''')
tf.app.flags.DEFINE_float('learning_rate_decay_step', 30000,'''Learning rate decay factor.''')

tf.app.flags.DEFINE_integer('batch_size', 4, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('eval_size', 4, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_iterations', 2, '''The number of iterations to unfold the pose machine.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,'''How many preprocess threads to use.''')
tf.app.flags.DEFINE_integer('n_landmarks', 84,'''number of landmarks.''')
tf.app.flags.DEFINE_integer('rescale', 256,'''Image scale.''')

tf.app.flags.DEFINE_string('dataset_dir', './data', '''Directory where to load datas.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train', '''Directory where to write event logs and checkpoint.''')
tf.app.flags.DEFINE_string('eval_dir', '','''Directory where to write event logs and checkpoint.''')
tf.app.flags.DEFINE_string('graph_dir', 'model/weight.pkl','''If specified, restore this pretrained model.''')

tf.app.flags.DEFINE_integer('max_steps', 1000000,'''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0','''Device to train with.''')

tf.app.flags.DEFINE_integer('flip_pred', 0,'''db name.''')

tf.app.flags.DEFINE_string('train_model', '', '''training model.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', '''Restore pretrained model.''')
tf.app.flags.DEFINE_string('testset_name', '', '''test set name.''')
tf.app.flags.DEFINE_string('model_name', '', '''test model name.''')
tf.app.flags.DEFINE_string('savemat_name', '', '''save_mat_name''')