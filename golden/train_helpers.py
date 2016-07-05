import numpy as np
import os
import yaml
import cPickle as pickle
import lasagne
import theano.tensor as T
import nolearn.lasagne
from ml_helpers import minibatch_generators as mbg
import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DimshuffleLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, very_leaky_rectify as vlr
import theano
from lasagne.layers import batch_norm, ElemwiseSumLayer, ExpressionLayer, DimshuffleLayer
import lasagne.layers


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


class Log1Plus(lasagne.layers.Layer):
    def __init__(self, incoming, off=lasagne.init.Constant(1.0), mult=lasagne.init.Constant(1.0), **kwargs):
        super(Log1Plus, self).__init__(incoming, **kwargs)
        num_channels = self.input_shape[1]
        self.off = self.add_param(off, shape=(num_channels,), name='off')
        self.mult = self.add_param(mult, shape=(num_channels,), name='mult')

    def get_output_for(self, input, **kwargs):
        return T.log(self.off.dimshuffle('x', 0, 'x', 'x') + self.mult.dimshuffle('x', 0, 'x', 'x') * input)

    def get_output_shape_for(self, input_shape):
        return input_shape


class SpecSampler(object):

    def __init__(self, batch_size, hww, do_aug, learn_log, randomise=False,
            seed=None, balanced=True):
        self.do_aug = do_aug
        self.learn_log = learn_log
        self.hww = hww
        self.seed = seed
        self.randomise = randomise
        self.balanced = balanced
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        blank_spec = np.zeros((X[0].shape[0], 2*self.hww))
        self.specs = np.hstack([blank_spec] + X + [blank_spec])[None, ...]

        blank_label = np.zeros(2*self.hww) - 1
        if y is not None:
            labels = [yy > 0 for yy in y]
        else:
            labels = [np.zeros(self.specs.shape[2] - 4*self.hww)]

        self.labels = np.hstack([blank_label] + labels + [blank_label])

        which_spec = [ii * np.ones(xx.shape[1]) for ii, xx in enumerate(X)]
        self.which_spec = np.hstack([blank_label] + which_spec + [blank_label]).astype(np.int32)

        self.medians = np.zeros((len(X), X[0].shape[0], self.hww*2))
        for idx, spec in enumerate(X):
            self.medians[idx] = np.median(spec, axis=1, keepdims=True)

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
            X = np.zeros((bs, channels, height, self.hww*2), np.float32)
            y = np.zeros(bs) * np.nan
            if self.learn_log:
                X_medians = np.zeros((bs, channels, height, self.hww*2), np.float32)
            count = 0

            for loc in sampled_locs:
                which = self.which_spec[loc]

                X[count] = self.specs[:, :, (loc-self.hww):(loc+self.hww)]

                if not self.learn_log:
                    X[count, 1] = X[count, 0] - self.medians[which]
                    # X[count, 0] = (X[count, 0] - X[count, 0].mean()) / X[count, 0].std()
                    X[count, 0] = (X[count, 1] - X[count, 1].mean(0, keepdims=True)) / (X[count, 1].std(0, keepdims=True) + 0.001)

                    X[count, 2] = (X[count, 1] - X[count, 1].mean()) / X[count, 1].std()
                    X[count, 3] = X[count, 1] / X[count, 1].max()

                y[count] = self.labels[loc]
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
                yield X.astype(np.float32), y.astype(np.int32)


class HelpersBaseClass(object):
    # sub dir is a class attribute so subclasses can override it
    subdir = ""

    def __init__(self, logging_dir):
        self.savedir = force_make_dir(logging_dir + self.subdir)


class SaveHistory(HelpersBaseClass):
    def __init__(self, logging_dir):
        super(SaveHistory, self).__init__(logging_dir)
        self.savepath = self.savedir + "history.yaml"
        with open(self.savepath, 'w'):
            pass

    def __call__(self, net, history):
        '''
        Dumps nolearn history to disk, one line at a time
        '''
        # handling numpy bool values
        for key, item in history[-1].iteritems():
            if type(item) == np.bool_:
                history[-1][key] = bool(item)
            elif type(item).__module__ == 'numpy':
                history[-1][key] = float(item)

        yaml.dump([history[-1]], open(self.savepath, 'a'), default_flow_style=False)


class SaveWeights(HelpersBaseClass):
    subdir = "/weights/"

    def __init__(self, logging_dir, save_weights_every, new_file_every):
        super(SaveWeights, self).__init__(logging_dir)
        self.save_weights_every = save_weights_every
        self.new_file_every = new_file_every

    def __call__(self, net, history):
        '''
        Dumps weights to disk. Saves weights every epoch, but only starts a
        new file every X epochs.
        '''
        if (len(history) - 1) % self.save_weights_every == 0:
            filenum = (len(history) - 1) / self.new_file_every
            savepath = self.savedir + "weights_%06d.pkl" % filenum
            net.save_params_to(savepath)


class EarlyStopping(object):
    # https://github.com/dnouri/nolearn/issues/18
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = 0#np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_accuracy']
        current_epoch = train_history[-1]['epoch']
        if current_valid > self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()  # updated
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()


def create_net(SPEC_HEIGHT, HWW, LEARN_LOG, NUM_FILTERS,
    WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM):

    if not DO_BATCH_NORM:
        batch_norm = lambda x: x
    else:
        from lasagne.layers import batch_norm

    channels = 1 if LEARN_LOG else 4

    # main input layer, then logged
    net = {}
    net['input'] = InputLayer((None, channels, SPEC_HEIGHT, HWW*2), name='input')

    if LEARN_LOG:
        off = lasagne.init.Constant(0.01)
        mult = lasagne.init.Constant(1.0)

        net['input_logged'] = Log1Plus(net['input'], off, mult)

        # logging the median and multiplying by -1
        net['input_med'] = InputLayer((None, channels, SPEC_HEIGHT, HWW*2), name='input_med')
        net['med_logged'] = Log1Plus(
            net['input_med'], off=net['input_logged'].off, mult=net['input_logged'].mult)
        net['med_logged'] = ExpressionLayer(net['med_logged'], lambda X: -X)

        # summing the logged input with the negative logged median
        net['input'] = ElemwiseSumLayer((net['input_logged'], net['med_logged']))

    net['conv1_1'] = batch_norm(
        ConvLayer(net['input'], NUM_FILTERS, (SPEC_HEIGHT - WIGGLE_ROOM, CONV_FILTER_WIDTH), nonlinearity=vlr))
    net['pool1'] = PoolLayer(net['conv1_1'], pool_size=(2, 2), stride=(2, 2), mode='max')
    net['pool1'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv1_2'] = batch_norm(ConvLayer(net['pool1'], NUM_FILTERS, (1, 3), nonlinearity=vlr))
    # net['pool2'] = PoolLayer(net['conv1_2'], pool_size=(1, 2), stride=(1, 1))
    net['pool2'] = DropoutLayer(net['conv1_2'], p=0.5)

    net['fc6'] = batch_norm(DenseLayer(net['pool2'], num_units=NUM_DENSE_UNITS, nonlinearity=vlr))
    net['fc6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = batch_norm(DenseLayer(net['fc6'], num_units=NUM_DENSE_UNITS, nonlinearity=vlr))
    net['fc7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net


def noisy_loss_objective(predictions, targets):
    # epsilon = np.float32(1.0e-6)
    one = np.float32(1.0)
    beta = np.float32(0.75)
    # pred = T.clip(predictions, epsilon, one - epsilon)

    # assume targets are just indicator values...
    A = (one - targets) * (beta + (one - beta) * T.round(predictions[:, 0])) * T.log(predictions[:, 0])
    B = targets * T.log(predictions[:, 1])
    return - (A + B)
