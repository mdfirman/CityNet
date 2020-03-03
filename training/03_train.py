import sys
class_to_use = sys.argv[1]
assert class_to_use in ['biotic', 'anthrop']

import pickle
import numpy as np
from IPython import embed
import tensorflow as tf
from tensorflow.contrib import slim

from tqdm import tqdm
from easydict import EasyDict as edict
import yaml

sys.path.append('../lib')
from train_helpers import SpecSampler, force_make_dir, create_net
import data_io
# import ui

small = False


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
    net = create_net(height, opts.HWW_X, opts.LEARN_LOG, opts.NUM_FILTERS,
        opts.WIGGLE_ROOM, opts.CONV_FILTER_WIDTH, opts.NUM_DENSE_UNITS, opts.DO_BATCH_NORM)

    y_in = tf.placeholder(tf.int32, (None))
    x_in = net['input']

    print("todo - fix this up...")
    trn_output = net['fc8']
    test_output = net['fc8']

    _trn_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=trn_output, labels=y_in))
    _test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=test_output, labels=y_in))
    print(y_in, trn_output, tf.argmax(trn_output, axis=1))

    pred = tf.cast(tf.argmax(trn_output, axis=1), tf.int32)
    _trn_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

    pred = tf.cast(tf.argmax(test_output, axis=1), tf.int32)
    _test_acc = tf.reduce_mean(tf.cast(tf.equal(y_in, pred), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=opts.LEARNING_RATE, beta1=0.5, beta2=0.9)

    train_op = slim.learning.create_train_op(_trn_loss, optimizer)

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(opts.MAX_EPOCHS):

            ######################
            # TRAINING
            trn_losses = []
            trn_accs = []

            for xx, yy in tqdm(train_sampler(train_X, train_y)):
                trn_ls, trn_acc, _ = sess.run(
                    [_trn_loss, _trn_acc, train_op], feed_dict={x_in: xx, y_in: yy})
                trn_losses.append(trn_ls)
                trn_accs.append(trn_acc)

            ######################
            # VALIDATION
            val_losses = []
            val_accs = []

            for xx, yy in test_sampler(test_X, test_y):
                val_ls, val_acc = sess.run([_test_loss, _test_acc], feed_dict={x_in: xx, y_in: yy})
                val_losses.append(val_ls)
                val_accs.append(val_acc)

            print(" %03d :: %02f  -  %02f  -  %02f  -  %02f" % (epoch, np.mean(trn_losses),
                np.mean(trn_accs), np.mean(val_losses), np.mean(val_accs)))

        #######################
        # TESTING
        if small:
            results_savedir = force_make_dir(logging_dir + 'results_SMALL_TEST/')
            predictions_savedir = force_make_dir(logging_dir + 'per_file_predictions_SMALL_TEST/')
        else:
            results_savedir = force_make_dir(logging_dir + 'results/')
            predictions_savedir = force_make_dir(logging_dir + 'per_file_predictions/')

        test_sampler = SpecSampler(64, opts.HWW_X, opts.HWW_Y, False, opts.LEARN_LOG, randomise=False, seed=10, balanced=False)

        for fname, spec, y in zip(test_files, test_X, test_y):
            probas = []
            y_true = []
            for Xb, yb in test_sampler([spec], [y]):
                preds = sess.run(test_output, feed_dict={x_in: Xb})
                probas.append(preds)
                y_true.append(yb)

            y_pred_prob = np.vstack(probas)
            y_true = np.hstack(y_true)
            y_pred = np.argmax(y_pred_prob, axis=1)

            print("Saving to {}".format(predictions_savedir))
            with open(predictions_savedir + fname, 'wb') as f:
                pickle.dump([y_true, y_pred_prob], f, -1)

        # save weights from network
        saver.save(sess, results_savedir + "weights_%d.pkl" % TEST_FOLD, global_step=1)


def train_large_test_golden(RUN_TYPE, opts):

    print(opts.CLASSNAME, RUN_TYPE)

    # loading testing data
    # (remember, here we are testing on ALL golden... so it doesn't matter what the test fold is
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
        force_make_dir(logging_dir)
        # sys.stdout = ui.Logger(logging_dir + 'log.txt')

        opts.height = train_X[0].shape[0]
        with open(logging_dir + 'network_opts.yaml', 'w') as f:
            yaml.dump(opts, f, default_flow_style=False)

        train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir, opts)


def train_golden(RUN_TYPE, opts):

    print(opts.CLASSNAME, RUN_TYPE)

    # loading testing data
    # (remember, here we are testing on ALL golden... so it doesn't matter what the test fold is
    golden_1, golden_2 = data_io.load_splits(test_fold=0)
    train_files = golden_1 + golden_2
    test_files = golden_2

    if small:
        test_files = test_files[:3]
        train_files = train_files[:3]

    train_X, train_y = data_io.load_data(
        train_files, opts.SPEC_TYPE, opts.LEARN_LOG, opts.CLASSNAME, opts.A, opts.B)

    test_X, test_y = data_io.load_data(
        test_files, opts.SPEC_TYPE, opts.LEARN_LOG, opts.CLASSNAME, opts.A, opts.B)

    for idx in range(opts.ENSEMBLE_MEMBERS):
        logging_dir = data_io.base + 'predictions/%s/%d/%s/' % (RUN_TYPE, idx, opts.CLASSNAME)
        force_make_dir(logging_dir)
        # sys.stdout = ui.Logger(logging_dir + 'log.txt')

        opts.height = train_X[0].shape[0]
        with open(logging_dir + 'network_opts.yaml', 'w') as f:
            yaml.dump(opts, f, default_flow_style=False)

        train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir, opts)


if __name__ == '__main__':
    opts = edict(dict(
        SPEC_TYPE = 'mel',
        ENSEMBLE_MEMBERS = 1,

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

    print("Training data: ", TRAINING_DATA)
    for key, val in opts.items():
        print("   ", key.ljust(20), val)

    RUN_TYPE = 'train_overlap_split_' + class_to_use

    train_golden(RUN_TYPE, opts)
