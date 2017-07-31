import sys
class_to_use = sys.argv[1]

import cPickle as pickle
import numpy as np

import lasagne
import theano.tensor as T
import theano

from tqdm import tqdm
from easydict import EasyDict as edict
import yaml

sys.path.append('../lib')
from train_helpers import SpecSampler
import train_helpers
import data_io
from ml_helpers import ui

small = True


def train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir, opts, TEST_FOLD=99,
        val_X=None, val_y=None):
    '''
    Doesn't do any data loading - assumes the train and test data are passed
    in as parameters!
    '''
    if val_X is None:
        val_X = test_X
        val_y = test_y

    # # creaging samplers and batch iterators
    train_sampler = SpecSampler(64, opts.HWW_X, opts.HWW_Y, opts.DO_AUGMENTATION, opts.LEARN_LOG, randomise=True, balanced=True)
    test_sampler = SpecSampler(64, opts.HWW_X, opts.HWW_Y, False, opts.LEARN_LOG, randomise=False, seed=10, balanced=True)

    height = train_X[0].shape[0]
    net = train_helpers.create_net(height, opts.HWW_X, opts.LEARN_LOG, opts.NUM_FILTERS,
        opts.WIGGLE_ROOM, opts.CONV_FILTER_WIDTH, opts.NUM_DENSE_UNITS, opts.DO_BATCH_NORM)

    y_in = T.ivector()
    x_in = net['input'].input_var

    # print lasagne.layers.get_output_shape_for(net['prob'])
    trn_output = lasagne.layers.get_output(net['prob'], deterministic=False)
    test_output = lasagne.layers.get_output(net['prob'], deterministic=True)
    params = lasagne.layers.get_all_params(net['prob'], trainable=True)

    _trn_loss = lasagne.objectives.categorical_crossentropy(trn_output, y_in).mean()
    _test_loss = lasagne.objectives.categorical_crossentropy(test_output, y_in).mean()
    _trn_acc = T.mean(T.eq(T.argmax(trn_output, axis=1), y_in))
    _test_acc = T.mean(T.eq(T.argmax(test_output, axis=1), y_in))

    updates = lasagne.updates.adam(_trn_loss, params, learning_rate=opts.LEARNING_RATE)

    print "Compiling...",
    train_fn = theano.function([x_in, y_in], [_trn_loss, _trn_acc], updates=updates)
    val_fn = theano.function([x_in, y_in], [_test_loss, _test_acc])
    pred_fn = theano.function([x_in], test_output)
    print "DONE"

    for epoch in range(opts.MAX_EPOCHS):

        ######################
        # TRAINING
        trn_losses = []
        trn_accs = []

        for xx, yy in tqdm(train_sampler(train_X, train_y)):
            trn_ls, trn_acc = train_fn(xx, yy)
            trn_losses.append(trn_ls)
            trn_accs.append(trn_acc)

        ######################
        # VALIDATION
        val_losses = []
        val_accs = []

        for xx, yy in test_sampler(test_X, test_y):
            val_ls, val_acc = val_fn(xx, yy)
            val_losses.append(val_ls)
            val_accs.append(val_acc)

        print " %03d :: %02f  -  %02f  -  %02f  -  %02f" % (epoch, np.mean(trn_losses),
            np.mean(trn_accs), np.mean(val_losses), np.mean(val_accs))

    #######################
    # TESTING
    results_savedir = train_helpers.force_make_dir(logging_dir + 'results/')
    predictions_savedir = train_helpers.force_make_dir(logging_dir + 'per_file_predictions/')

    test_sampler = SpecSampler(64, opts.HWW_X, opts.HWW_Y, False, opts.LEARN_LOG, randomise=False, seed=10, balanced=False)

    for fname, spec, y in zip(test_files, test_X, test_y):
        probas = []
        y_true = []
        for Xb, yb in test_sampler([spec], [y]):
            probas.append(pred_fn(Xb))
            y_true.append(yb)

        y_pred_prob = np.vstack(probas)
        y_true = np.hstack(y_true)
        y_pred = np.argmax(y_pred_prob, axis=1)

        with open(predictions_savedir + fname, 'w') as f:
            pickle.dump([y_true, y_pred_prob], f, -1)

    # save weights from network
    param_vals = lasagne.layers.get_all_param_values(net['prob'])
    with open(results_savedir + "weights_%d.pkl" % TEST_FOLD, 'w') as f:
        pickle.dump(param_vals, f, -1)


def train_large_test_golden(RUN_TYPE, opts):

    print opts.CLASSNAME, RUN_TYPE

    # loading testing data
    # (remember, here we are testing on ALL golden... so it doesn't matter what the test fold is
    # print "WARING" * 10
    golden_1, golden_2 = data_io.load_splits(test_fold=0)
    test_files = golden_1 + golden_2
    if small:
        test_files = test_files[:3]
        max_to_load = 3
    else:
        max_to_load = 10000000
    test_X, test_y = data_io.load_data(
        test_files, opts.SPEC_TYPE, opts.LEARN_LOG, opts.CLASSNAME, opts.A, opts.B)

    # loading training data...
    train_X, train_y = data_io.load_large_data(
        opts.SPEC_TYPE, opts.LEARN_LOG, opts.CLASSNAME, opts.A, opts.B, max_to_load=max_to_load)

    for idx in range(opts.ENSEMBLE_MEMBERS):
        logging_dir = data_io.base + 'predictions/%s/%d/%s/' % (RUN_TYPE, idx, opts.CLASSNAME)
        train_helpers.force_make_dir(logging_dir)
        sys.stdout = ui.Logger(logging_dir + 'log.txt')

        opts.height = train_X[0].shape[0]
        with open(logging_dir + 'network_opts.yaml', 'w') as f:
            yaml.dump(opts, f, default_flow_style=False)

        train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir, opts)


if __name__ == '__main__':
    opts = edict(dict(
        SPEC_TYPE = 'mel',
        ENSEMBLE_MEMBERS = 5,

        # data preprocessing options
        HWW_X = 10,
        HWW_Y = 10,
        LEARN_LOG = 0,
        DO_AUGMENTATION = 1,

        # network parameters
        DO_BATCH_NORM = 1,
        NUM_FILTERS = 32 * 4,
        NUM_DENSE_UNITS = 128 * 4,
        CONV_FILTER_WIDTH = 4,
        WIGGLE_ROOM = 5,
        MAX_EPOCHS = 5,
        LEARNING_RATE = 0.001,

        CLASSNAME = class_to_use
        ))
    opts.B = 10.0 if opts.CLASSNAME == 'biotic' else 2.00
    opts.A = 0.001 if opts.CLASSNAME == 'biotic' else 0.025

    TRAINING_DATA = 'large'

    print "Training data: ", TRAINING_DATA
    for key, val in opts.iteritems():
        print "   ", key.ljust(20), val

    RUN_TYPE = 'ensemble_train_tmp_' + class_to_use

    train_large_test_golden(RUN_TYPE, opts)
