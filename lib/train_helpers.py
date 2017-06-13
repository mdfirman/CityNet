import numpy as np
import os
import yaml
import lasagne
import theano.tensor as T
from ml_helpers import minibatch_generators as mbg

from lasagne.layers import InputLayer, DimshuffleLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
try:
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
except:
    from lasagne.layers import Conv2DLayer as ConvLayer
    from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, very_leaky_rectify as vlr
import theano
try:
    from lasagne.layers import batch_norm
except:
    from normalization import batch_norm
from lasagne.layers import ElemwiseSumLayer, ExpressionLayer, DimshuffleLayer
import lasagne.layers

# Which parameters are used in the network generation?
net_params = ['DO_BATCH_NORM', 'NUM_FILTERS', 'NUM_DENSE_UNITS',
              'CONV_FILTER_WIDTH', 'WIGGLE_ROOM', 'HWW', 'LEARN_LOG', 'SPEC_HEIGHT']


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


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
            X = np.zeros((bs, channels, height, self.hww*2), np.float32)
            y = np.zeros(bs) * np.nan
            if self.learn_log:
                X_medians = np.zeros((bs, channels, height), np.float32)
            count = 0

            for loc in sampled_locs:
                which = self.which_spec[loc]

                X[count] = self.specs[:, :, (loc-self.hww):(loc+self.hww)]

                if not self.learn_log:
                    X[count, 1] = X[count, 0] - self.medians[which][:, None]
                    # X[count, 0] = (X[count, 0] - X[count, 0].mean()) / X[count, 0].std()
                    X[count, 0] = (X[count, 1] - X[count, 1].mean(1, keepdims=True)) / (X[count, 1].std(1, keepdims=True) + 0.001)

                    X[count, 2] = (X[count, 1] - X[count, 1].mean()) / X[count, 1].std()
                    X[count, 3] = X[count, 1] / X[count, 1].max()

                y[count] = self.labels[(loc-self.hww):(loc+self.hww)].max()
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


class NormalisationLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(NormalisationLayer, self).__init__(incomings, **kwargs)
        # no parameters to define

    def get_output_for(self, input, **kwargs):

        x, medians = input
        # Subtracting logged median from each channel
        x = x - medians

        flattened = x.flatten(3)  # BxCxHxW --> BxCxHW

        # Rescaling each channel to be between 0 and 1
        A = flattened / (flattened.max(2, keepdims=True) + 0.000001)
        A = A.reshape(x.shape)

        B = x# - medians

        # Whitening each spectrogram
        C = flattened - flattened.mean(2, keepdims=True)
        C = C / (C.std(2, keepdims=True) + 0.0000001)
        C = C.reshape(x.shape)

        # Whitening each row of each spectrogram
        D = x - x.mean(3, keepdims=True)
        D = D / (D.std(3, keepdims=True) + 0.0000001)

        return T.concatenate((A, B, C, D), axis=1)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return (input_shape[0], 4*input_shape[1], input_shape[2], input_shape[3])


class LearnLogLayer(lasagne.layers.Layer):
    """Learn the parameters of a logarithm log(Ax + B), repeated multiple times...

    Assumes input is of shape None x 1 x H x W"""

    def __init__(self, incoming, num_repeats, A=lasagne.init.Normal(),
                 B=lasagne.init.Normal(), **kwargs):
        super(LearnLogLayer, self).__init__(incoming, **kwargs)
        self.num_repeats = num_repeats
        self.A = self.add_param(A, (1, num_repeats, 1, 1), name='A')
        self.B = self.add_param(B, (1, num_repeats, 1, 1), name='B')

    def get_output_for(self, input, **kwargs):
        # Using sigmoids to contrain A and B. Could use exp if want
        # unbounded upper limit. Also, this might be undercontrained
        # right now.
        _A = T.nnet.sigmoid(self.A) * 0 + 10.0
        _B = T.nnet.sigmoid(self.B) * 0 + 0.001

        # big hack to make input broadcastable...
        input = input[:, 0, :, :][:, None, :, :]

        return T.log(_A + _B * input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_repeats, input_shape[2], input_shape[3])


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

        NUM_LOG_CHANNELS = 1
        net['input_logged'] = LearnLogLayer(net['input'], NUM_LOG_CHANNELS)
        _A, _B = net['input_logged'].A, net['input_logged'].B

        # logging the median and multiplying by -1
        net['input_med'] = InputLayer((None, channels, SPEC_HEIGHT), name='input_med')
        net['input_med'] = DimshuffleLayer(net['input_med'], (0, 1, 2, 'x'))
        net['med_logged'] = LearnLogLayer(net['input_med'], NUM_LOG_CHANNELS, A=_A, B=_B)

        # net['med_logged'] = ExpressionLayer(net['med_logged'], lambda X: -X)
        #
        # # summing the logged input with the negative logged median
        # net['input'] = ElemwiseSumLayer((net['input_logged'], net['med_logged']))
        #
        # performing the multiplications
        net['input'] = NormalisationLayer((net['input_logged'], net['med_logged']))


    net['conv1_1'] = batch_norm(
        ConvLayer(net['input'], NUM_FILTERS, (SPEC_HEIGHT - WIGGLE_ROOM, CONV_FILTER_WIDTH), nonlinearity=vlr))
    # net['pool1'] = PoolLayer(net['conv1_1'], pool_size=(2, 2), stride=(2, 2), mode='max')
    net['pool1'] = DropoutLayer(net['conv1_1'], p=0.5)
    net['conv1_2'] = batch_norm(ConvLayer(net['pool1'], NUM_FILTERS, (1, 3), nonlinearity=vlr))
    W = net['conv1_2'].output_shape[3]
    net['pool2'] = PoolLayer(net['conv1_2'], pool_size=(1, W), stride=(1, 1), mode='max')
    net['pool2'] = DropoutLayer(net['pool2'], p=0.5)

    net['fc6'] = batch_norm(DenseLayer(net['pool2'], num_units=NUM_DENSE_UNITS, nonlinearity=vlr))
    net['fc6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = batch_norm(DenseLayer(net['fc6'], num_units=NUM_DENSE_UNITS, nonlinearity=vlr))
    net['fc7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7'], num_units=2, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
