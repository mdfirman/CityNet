import os
import lasagne
import theano.tensor as T
import nolearn.lasagne
from ml_helpers import minibatch_generators as mbg
import numpy as np


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


class Log1Plus(lasagne.layers.Layer):
    def __init__(self, incoming, off=lasagne.init.Constant(1.0), mult=lasagne.init.Constant(1.0), **kwargs):
        super(Log1Plus, self).__init__(incoming, **kwargs)
        num_channels = 1#self.input_shape[1]
        self.off = self.add_param(off, shape=(num_channels,), name='off')
        self.mult = self.add_param(mult, shape=(num_channels,), name='mult')

    def get_output_for(self, input, **kwargs):
        return T.log(self.off.dimshuffle('x', 0, 'x', 'x') + self.mult.dimshuffle('x', 0, 'x', 'x') * input)

    def get_output_shape_for(self, input_shape):
        return input_shape



# setting up network
class MyTrainSplit(nolearn.lasagne.TrainSplit):
    # custom data split
    def __call__(self, data, Yb, net):
        return data['train']['X'], data['val']['X'], data['train']['y'], data['val']['y']


def augment(X):
    for idx in xrange(X.shape[0]):
        mult = (1.0 + np.random.randn() * 0.1)
        add = np.random.randn() * 0.2
        X[idx] *= mult
        X[idx] += add
        if np.random.rand() > 0.95:
            shift = np.random.randint(0, 224)
            X[idx, 0] = np.roll(X[idx, 0], shift, 1)
    return X


class MyBatch(nolearn.lasagne.BatchIterator):
    def transform(self, Xb, yb):
        return augment(Xb), yb

    def __iter__(self):
        bs = self.batch_size
        for batch_idxs in mbg.minibatch_idx_iterator(
                self.y, bs, randomise=1, balanced=1):

            Xb = self.X[batch_idxs]
            yb = self.y[batch_idxs]
            yield self.transform(Xb, yb)
