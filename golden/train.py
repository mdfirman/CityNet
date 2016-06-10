# creating spectrograms from all the files, and saving split labelled versions to disk ready for machine learning
import matplotlib.pyplot as plt

import os
import sys
import cPickle as pickle
import numpy as np
import time
import random
import yaml

import nolearn
import nolearn.lasagne
import lasagne.layers

from lasagne.layers import InputLayer, DimshuffleLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, elu as vlr
import theano
from lasagne.layers import batch_norm, ElemwiseSumLayer, ExpressionLayer, DimshuffleLayer
import theano.tensor as T

from helpers import SpecSampler, Log1Plus
from data_io import load_data, load_splits

HWW = 15
SPEC_HEIGHT = 300
LEARN_LOG = True
DO_AUGMENTATION = True
DO_BATCH_NORM = True
NUM_FILTERS = 32
NUM_DENSE_UNITS = 64
CLASSNAME = 'anthrop'

# loading data
train_files, test_files = load_splits()
train_data, test_data = load_data(train_files, test_files, SPEC_HEIGHT, LEARN_LOG, CLASSNAME)
print len(test_data[0]), len(train_data[0])

# creaging samplers and batch iterators
train_sampler = SpecSampler(train_data[0], train_data[1], HWW, DO_AUGMENTATION, LEARN_LOG)
test_sampler = SpecSampler(test_data[0], test_data[1], HWW, False, LEARN_LOG)

class MyBatch(nolearn.lasagne.BatchIterator):
    def __iter__(self):
        for _ in range(32):
            yield self.X.sample(self.batch_size)


class MyBatchTest(nolearn.lasagne.BatchIterator):
    def __iter__(self):
        for idx in range(128):
            yield self.X.sample(self.batch_size, seed=idx)


class MyTrainSplit(nolearn.lasagne.TrainSplit):
    # custom data split
    def __call__(self, data, Yb, net):
        return train_sampler, test_sampler, None, None


if not DO_BATCH_NORM:
    batch_norm = lambda x: x

# main input layer, then logged
net = {}
net['input'] = InputLayer((None, 1, SPEC_HEIGHT, HWW*2), name='input')

if LEARN_LOG:
    off = lasagne.init.Constant(0.01)
    mult = lasagne.init.Constant(1.0)

    net['input_logged'] = Log1Plus(net['input'], off, mult)

    # logging the median and multiplying by -1
    net['input_med'] = InputLayer((None, 1, SPEC_HEIGHT, HWW*2), name='input_med')
    net['med_logged'] = Log1Plus(net['input_med'], off=net['input_logged'].off, mult=net['input_logged'].mult)
    net['med_logged'] = ExpressionLayer(net['med_logged'], lambda X: -X)

    # summing the logged input with the negative logged median
    net['input'] = ElemwiseSumLayer((net['input_logged'], net['med_logged']))

net['conv1_1'] = batch_norm(
    ConvLayer(net['input'], NUM_FILTERS, (SPEC_HEIGHT - 6, 6), nonlinearity=vlr))
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

net = nolearn.lasagne.NeuralNet(
    layers=net['prob'],
    max_epochs=500,
    update=lasagne.updates.adam,
    update_learning_rate=0.0005,
#     update_momentum=0.975,
    verbose=1,
    batch_iterator_train=MyBatch(128),
    batch_iterator_test=MyBatchTest(128),
    train_split=MyTrainSplit(None),
    check_input=False
)
net.fit(None, None)
