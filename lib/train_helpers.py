import numpy as np
import os
import collections
import tensorflow.compat.v1 as tf
import tf_slim as slim
import minibatch_generators as mbg


# Which parameters are used in the network generation?
net_params = ['DO_BATCH_NORM', 'NUM_FILTERS', 'NUM_DENSE_UNITS',
              'CONV_FILTER_WIDTH', 'WIGGLE_ROOM', 'HWW_X', 'LEARN_LOG']


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


class SpecSampler(object):

    def __init__(self, batch_size, hww_x, hww_y, do_aug, learn_log, randomise=False,
            seed=None, balanced=True):
        self.do_aug = do_aug
        self.learn_log = learn_log
        self.hww_x = hww_x
        self.hww_y = hww_y
        self.seed = seed
        self.randomise = randomise
        self.balanced = balanced
        self.batch_size = batch_size

    def __call__(self, X, y=None):

        # must pad X and Y the same amount
        pad_hww = max(self.hww_x, self.hww_y)

        blank_spec = np.zeros((X[0].shape[0], 2 * pad_hww))
        self.specs = np.hstack([blank_spec] + X + [blank_spec])[None, ...]

        blank_label = np.zeros(2 * pad_hww) - 1
        if y is not None:
            labels = [yy > 0 for yy in y]
        else:
            labels = [np.zeros(self.specs.shape[2] - 4 * pad_hww)]

        self.labels = np.hstack([blank_label] + labels + [blank_label])

        which_spec = [ii * np.ones(xx.shape[1]) for ii, xx in enumerate(X)]
        self.which_spec = np.hstack([blank_label] + which_spec + [blank_label]).astype(np.int32)

        self.medians = np.zeros((len(X), X[0].shape[0]))
        for idx, spec in enumerate(X):
            self.medians[idx] = np.median(spec, axis=1)

        assert self.labels.shape[0] == self.specs.shape[2]
        return self

    def __iter__(self): ##, num_per_class, seed=None
        #num_samples = num_per_class * 2
        channels = self.specs.shape[0]
        if not self.learn_log:
            channels += 3
        height = self.specs.shape[1]

        if self.seed is not None:
            np.random.seed(self.seed)

        idxs = np.where(self.labels >= 0)[0]
        for sampled_locs, y in mbg.minibatch_iterator(idxs, self.labels[idxs],
            self.batch_size, randomise=self.randomise, balanced=self.balanced,
                class_size='smallest'):

            # extract the specs
            bs = y.shape[0]  # avoid using self.batch_size as last batch may be smaller
            X = np.zeros((bs, channels, height, self.hww_x*2), np.float32)
            y = np.zeros(bs) * np.nan
            if self.learn_log:
                X_medians = np.zeros((bs, channels, height), np.float32)
            count = 0

            for loc in sampled_locs:
                which = self.which_spec[loc]

                X[count] = self.specs[:, :, (loc-self.hww_x):(loc+self.hww_x)]

                if not self.learn_log:
                    X[count, 1] = X[count, 0] - self.medians[which][:, None]
                    # X[count, 0] = (X[count, 0] - X[count, 0].mean()) / X[count, 0].std()
                    X[count, 0] = (X[count, 1] - X[count, 1].mean(1, keepdims=True)) / (X[count, 1].std(1, keepdims=True) + 0.001)

                    X[count, 2] = (X[count, 1] - X[count, 1].mean()) / X[count, 1].std()
                    X[count, 3] = X[count, 1] / X[count, 1].max()

                y[count] = self.labels[(loc-self.hww_y):(loc+self.hww_y)].max()
                if self.learn_log:
                    which = self.which_spec[loc]
                    X_medians[count] = self.medians[which]

                count += 1

            # doing augmentation
            if self.do_aug:
                if self.learn_log:
                    mult = (1.0 + np.random.randn(bs, 1, 1, 1) * 0.1)
                    mult = np.clip(mult, 0.1, 200)
                    X *= mult
                else:
                    X *= (1.0 + np.random.randn(bs, 1, 1, 1) * 0.1)
                    X += np.random.randn(bs, 1, 1, 1) * 0.1
                    if np.random.rand() > 0.9:
                        X += np.roll(X, 1, axis=0) * np.random.randn()

            if self.learn_log:
                xb = {'input': X.astype(np.float32), 'input_med': X_medians.astype(np.float32)}
                yield xb, y.astype(np.int32)

            else:
                yield X.astype(np.float32).transpose(0, 2, 3, 1), y.astype(np.int32)


def create_net(SPEC_HEIGHT, HWW_X, LEARN_LOG, NUM_FILTERS,
    WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM):

    tf.compat.v1.disable_eager_execution()

    channels = 4
    net = collections.OrderedDict()

    net['input'] = tf.placeholder(
        tf.float32, (None, SPEC_HEIGHT, HWW_X*2, channels), name='input')
    net['conv1_1'] = slim.conv2d(
        net['input'], NUM_FILTERS, (SPEC_HEIGHT - WIGGLE_ROOM, CONV_FILTER_WIDTH),
        padding='valid', activation_fn=None, biases_initializer=None)
    net['conv1_1'] = tf.nn.leaky_relu(net['conv1_1'], alpha=1/3)

    net['conv1_2'] = slim.conv2d(
        net['conv1_1'], NUM_FILTERS, (1, 3), padding='valid',
        activation_fn=None, biases_initializer=None)
    net['conv1_2'] = tf.nn.leaky_relu(net['conv1_2'], alpha=1/3)

    W = net['conv1_2'].shape[2]
    net['pool2'] = slim.max_pool2d(
        net['conv1_2'], kernel_size=(1, W), stride=(1, 1))

    net['pool2'] = tf.transpose(net['pool2'], (0, 3, 2, 1))
    net['pool2_flat'] = slim.flatten(net['pool2'])

    net['fc6'] = slim.fully_connected(
        net['pool2_flat'], NUM_DENSE_UNITS,
        activation_fn=None, biases_initializer=None)
    net['fc6'] = tf.nn.leaky_relu(net['fc6'], alpha=1/3)

    net['fc7'] = slim.fully_connected(
        net['fc6'], NUM_DENSE_UNITS,
        activation_fn=None, biases_initializer=None)
    net['fc7'] = tf.nn.leaky_relu(net['fc7'], alpha=1/3)

    net['fc8'] = slim.fully_connected(net['fc7'], 2, activation_fn=None)
    # net['fc8'] = tf.nn.leaky_relu(net['fc8'], alpha=1/3)
    net['output'] = tf.nn.softmax(net['fc8'])

    return net
