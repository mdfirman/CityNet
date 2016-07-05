# creating spectrograms from all the files, and saving split labelled versions to disk ready for machine learning
import matplotlib.pyplot as plt

import sys
import cPickle as pickle
import numpy as np

import nolearn
import nolearn.lasagne
import lasagne

from train_helpers import SpecSampler
import train_helpers
import data_io
from ml_helpers import ui
from ml_helpers.evaluation import plot_confusion_matrix


def train_and_test(train_X, test_X, train_y, test_y, test_files, logging_dir,
        CLASSNAME, HWW, DO_AUGMENTATION, LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM,
        CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS,
        LEARNING_RATE, NOISY_LOSS, TEST_FOLD=99):
    '''
    Doesn't do any data loading - assumes the train and test data are passed
    in as parameters!
    '''
    # # creaging samplers and batch iterators
    train_sampler = SpecSampler(64, HWW, DO_AUGMENTATION, LEARN_LOG, randomise=True)
    test_sampler = SpecSampler(64, HWW, False, LEARN_LOG, randomise=True, seed=10)

    class MyTrainSplit(nolearn.lasagne.TrainSplit):
        # custom data split
        def __call__(self, Xb, Yb, net):
            return train_X, test_X, train_y, test_y

    height = train_X[0].shape[0]
    net = train_helpers.create_net(height, HWW, LEARN_LOG, NUM_FILTERS,
        WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM)

    save_history = train_helpers.SaveHistory(logging_dir)

    def print_ab(net, history):
        print "A, B = ",
        print net.layers_['log1plus1'].off.get_value(),
        print net.layers_['log1plus1'].mult.get_value()

    if NOISY_LOSS:
        objective_loss_function = train_helpers.noisy_loss_objective
    else:
        objective_loss_function = lasagne.objectives.categorical_crossentropy

    net = nolearn.lasagne.NeuralNet(
        layers=net['prob'],
        max_epochs=MAX_EPOCHS,
        update=lasagne.updates.adam,
        update_learning_rate=LEARNING_RATE,
        verbose=1,
        batch_iterator_train=train_sampler,
        batch_iterator_test=test_sampler,
        train_split=MyTrainSplit(None),
        custom_epoch_scores=[('N/A', lambda x, y: 0.0)],
        on_epoch_finished=[save_history],
        objective_loss_function=objective_loss_function,
        check_input=False
    )
    net.initialize()
    print net.layers_.keys()
    net.fit(None, None)

    results_savedir = train_helpers.force_make_dir(logging_dir + 'results/')
    predictions_savedir = train_helpers.force_make_dir(
        logging_dir + 'per_file_predictions/')

    # now test the algorithm and save:
    probas = []
    y_true = []
    for Xb, yb in test_sampler(test_X, test_y):
        probas.append(net.apply_batch_func(net.predict_iter_, Xb))
        y_true.append(yb)
    y_pred_prob = np.vstack(probas)
    y_true = np.hstack(y_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # confusion matrix
    plt.figure(figsize=(5, 5))
    plot_confusion_matrix(y_true, y_pred, normalise=True, cls_labels=['None', CLASSNAME])
    plt.savefig(results_savedir + 'conf_mat_%d.png' % TEST_FOLD)
    plt.close()

    # final predictions
    # todo - actually save one per file? (no, let's do this balanced...)
    with open(results_savedir + "predictions_%d.pkl" % TEST_FOLD, 'w') as f:
        pickle.dump([y_true, y_pred_prob], f, -1)

    # now final predictions per file...
    test_sampler = SpecSampler(64, HWW, False, LEARN_LOG, randomise=False,
        seed=10, balanced=False)

    for fname, spec, y in zip(test_files, test_X, test_y):
        probas = []
        y_true = []
        for Xb, yb in test_sampler([spec], [y]):
            probas.append(net.apply_batch_func(net.predict_iter_, Xb))
            y_true.append(yb)
        y_pred_prob = np.vstack(probas)
        y_true = np.hstack(y_true)
        y_pred = np.argmax(y_pred_prob, axis=1)

        with open(predictions_savedir + fname, 'w') as f:
            pickle.dump([y_true, y_pred_prob], f, -1)

    # save weights from network
    net.save_params_to(results_savedir + "weights_%d.pkl" % TEST_FOLD)


def train_golden_all_folds(RUN_TYPE, SPEC_TYPE, CLASSNAME, HWW,
        DO_AUGMENTATION,
        LEARN_LOG, A, B, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
        NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE):

    print CLASSNAME, RUN_TYPE

    for test_fold in [0, 1, 2]:

        logging_dir = data_io.base + 'predictions/%s/%s/' % (RUN_TYPE, CLASSNAME)
        train_helpers.force_make_dir(logging_dir)
        sys.stdout = ui.Logger(logging_dir + 'log.txt')

        # loading data
        train_files, test_files = data_io.load_splits(test_fold)

        train_X, train_y = data_io.load_data(train_files, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B)
        test_X, test_y = data_io.load_data(test_files, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B)

        train_and_test(train_X, test_X, train_y, test_y, test_files,
                logging_dir, CLASSNAME, HWW, DO_AUGMENTATION,
                LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
                NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE,
                test_fold)


def train_large_test_golden(RUN_TYPE, SPEC_TYPE, CLASSNAME, HWW,
        DO_AUGMENTATION,
        LEARN_LOG, A, B, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
        NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE):

    print CLASSNAME, RUN_TYPE

    logging_dir = data_io.base + 'predictions/%s/%s/' % (RUN_TYPE, CLASSNAME)
    train_helpers.force_make_dir(logging_dir)
    sys.stdout = ui.Logger(logging_dir + 'log.txt')

    # loading testing data
    # (remember, here we are testing on ALL golden... so it doesn't matter what the test fold is
    train_files, test_files = data_io.load_splits(test_fold=0)
    test_X, test_y = data_io.load_data(
        test_files + train_files, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B)

    # loading training data...
    train_X, train_y = data_io.load_large_data(
        SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B, max_to_load=10000000)

    train_and_test(train_X, test_X, train_y, test_y, test_files + train_files,
            logging_dir, CLASSNAME, HWW, DO_AUGMENTATION,
            LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM, CONV_FILTER_WIDTH,
            NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS, LEARNING_RATE, NOISY_LOSS=False)


if __name__ == '__main__':
    params = dict(
        SPEC_TYPE = 'mel',

        # data preprocessing options
        A = 0.018,
        B = 10.04,
        HWW = 5,
        LEARN_LOG = 0,
        DO_AUGMENTATION = 1,

        # network parameters
        DO_BATCH_NORM = 1,
        NUM_FILTERS = 32,
        NUM_DENSE_UNITS = 128,
        CONV_FILTER_WIDTH = 4,
        WIGGLE_ROOM = 5,
        MAX_EPOCHS = 20,
        LEARNING_RATE = 0.0005,
        )

    TRAINING_DATA = 'golden'

    print "Training data: ", TRAINING_DATA
    for key, val in params.iteritems():
        print "   ", key.ljust(20), val

    if TRAINING_DATA == 'golden':
        train_golden_all_folds(
            RUN_TYPE = 'mel32_train_golden_new',
            CLASSNAME = 'biotic',
            **params
            )
    else:
        params['NUM_FILTERS'] *= 4
        params['NUM_DENSE_UNITS'] *= 4
        train_large_test_golden(
            RUN_TYPE = 'mel32_train_large_new',
            CLASSNAME = 'anthrop',
            **params
            )
