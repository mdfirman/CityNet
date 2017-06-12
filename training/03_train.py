# creating spectrograms from all the files, and saving split labelled versions to disk ready for machine learning
import matplotlib.pyplot as plt

import sys
class_to_use = sys.argv[1]

import cPickle as pickle
import numpy as np

import lasagne
import theano.tensor as T
import theano

sys.path.append('../lib')
from train_helpers import SpecSampler
import train_helpers
import data_io
from ml_helpers import ui
from ml_helpers.evaluation import plot_confusion_matrix
from tqdm import tqdm

def train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir,
        CLASSNAME, HWW, DO_AUGMENTATION, LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM,
        CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS,
        LEARNING_RATE, TEST_FOLD=99, val_X=None, val_y=None):
    '''
    Doesn't do any data loading - assumes the train and test data are passed
    in as parameters!
    '''
    if val_X is None:
        val_X = test_X
        val_y = test_y

    # # creaging samplers and batch iterators
    train_sampler = SpecSampler(64, HWW, DO_AUGMENTATION, LEARN_LOG, randomise=True, balanced=True)
    test_sampler = SpecSampler(64, HWW, False, LEARN_LOG, randomise=False, seed=10, balanced=True)

    height = train_X[0].shape[0]
    net = train_helpers.create_net(height, HWW, LEARN_LOG, NUM_FILTERS,
        WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM)

    y_in = T.ivector()
    x_in = net['input'].input_var

    # print lasagne.layers.get_output_shape_for(net['prob'])
    net_output = lasagne.layers.get_output(net['prob'])
    params = lasagne.layers.get_all_params(net['prob'], trainable=True)

    loss = lasagne.objectives.categorical_crossentropy(net_output, y_in).mean()
    acc = T.mean(T.eq(T.argmax(net_output, axis=1), y_in))

    updates = lasagne.updates.adam(loss, params, learning_rate=LEARNING_RATE)

    print "Compiling...",
    train_fn = theano.function([x_in, y_in], [loss, acc], updates=updates)
    val_fn = theano.function([x_in, y_in], [loss, acc])
    pred_fn = theano.function([x_in], net_output)
    print "DONE"

    for epoch in range(MAX_EPOCHS):

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

    test_sampler = SpecSampler(64, HWW, False, LEARN_LOG, randomise=False, seed=10, balanced=False)

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


def train_large_test_golden(RUN_TYPE, SPEC_TYPE, CLASSNAME, HWW,
        DO_AUGMENTATION,
        LEARN_LOG, A, B, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
        NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE):

    print CLASSNAME, RUN_TYPE

    logging_dir = data_io.base + 'predictions/%s/%s/' % (RUN_TYPE, CLASSNAME)
    train_helpers.force_make_dir(logging_dir)
    sys.stdout = ui.Logger(logging_dir + 'log.txt')

    # train_files_large, test_files_large = data_io.load_splits(
    #     test_fold=test_fold, large_data=True)
    # all_train_files = train_files_large + test_files_large

    # loading testing data
    # (remember, here we are testing on ALL golden... so it doesn't matter what the test fold is
    print "WARING" * 10
    train_files, test_files = data_io.load_splits(test_fold=0)
    test_X, test_y = data_io.load_data(
        test_files[:1] + train_files[:1], SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B)

    # loading training data...
    print "WARING" * 10
    train_X, train_y = data_io.load_large_data(
        SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B, max_to_load=10)

    train_and_test(train_X, test_X, train_y, test_y, test_files + train_files,
            logging_dir, CLASSNAME, HWW, DO_AUGMENTATION,
            LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
            NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE)


if __name__ == '__main__':
    params = dict(
        SPEC_TYPE = 'mel',

        # data preprocessing options
        HWW = 5,
        LEARN_LOG = 0,
        DO_AUGMENTATION = 1,

        # network parameters
        DO_BATCH_NORM = 1,
        NUM_FILTERS = 32,
        NUM_DENSE_UNITS = 128,
        CONV_FILTER_WIDTH = 4,
        WIGGLE_ROOM = 5,
        MAX_EPOCHS = 1,
        LEARNING_RATE = 0.001,

        CLASSNAME = class_to_use
        )
    params['B'] = 10.0 if params['CLASSNAME'] == 'biotic' else 2.00
    params['A'] = 0.001 if params['CLASSNAME'] == 'biotic' else 0.025

    TRAINING_DATA = 'large'

    print "Training data: ", TRAINING_DATA
    for key, val in params.iteritems():
        print "   ", key.ljust(20), val

    params['NUM_FILTERS'] *= 4
    params['NUM_DENSE_UNITS'] *= 4
    train_large_test_golden(
        RUN_TYPE = 'tmp',
        **params
    )
