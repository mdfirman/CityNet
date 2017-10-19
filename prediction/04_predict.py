# creating spectrograms from all the files, and saving split labelled versions to disk ready for machine learning
import sys
import cPickle as pickle
import numpy as np
import collections
import yaml

import lasagne
import theano.tensor as T
import theano

sys.path.append('../lib')
from train_helpers import SpecSampler
import train_helpers
import data_io

train_name = 'ensemble_train_anthrop'
classname = 'anthrop'

base = yaml.load(open('../CONFIG.yaml'))['base_dir'] + '/predictions/'

load_path = base + '/%s/%d/%s/results/weights_99.pkl'
predictions_savedir = train_helpers.force_make_dir(
    base + '/%s/%s/per_file_predictions/' % (train_name, classname))


def predict(A, B, ENSEMBLE_MEMBERS, SPEC_TYPE,
        CLASSNAME, HWW_X, HWW_Y, DO_AUGMENTATION, LEARN_LOG, NUM_FILTERS, WIGGLE_ROOM,
        CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM, MAX_EPOCHS,
        LEARNING_RATE):

    # Loading data
    golden_1, golden_2 = data_io.load_splits(test_fold=0)
    test_files = golden_1 + golden_2
    test_X, test_y = data_io.load_data(test_files, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B)

    # # creaging samplers and batch iterators
    test_sampler = SpecSampler(64, HWW_X, HWW_Y, False, LEARN_LOG, randomise=False, seed=10, balanced=True)

    height = test_X[0].shape[0]
    net = train_helpers.create_net(height, HWW_X, LEARN_LOG, NUM_FILTERS,
        WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM)

    x_in = net['input'].input_var

    test_output = lasagne.layers.get_output(net['prob'], deterministic=True)
    pred_fn = theano.function([x_in], test_output)

    test_sampler = SpecSampler(64, HWW_X, HWW_Y, False, LEARN_LOG, randomise=False, seed=10, balanced=False)

    y_preds_proba = collections.defaultdict(list)
    y_gts = {}

    def bal_acc(y_true, y_pred_class):
        total = {}
        total['tm'] = y_true.shape[0]
        total['tp'] = np.logical_and(y_true == y_pred_class, y_true == 1).sum()
        total['tn'] = np.logical_and(y_true == y_pred_class, y_true == 0).sum()
        total['fp'] = np.logical_and(y_true != y_pred_class, y_true == 0).sum()
        total['fn'] = np.logical_and(y_true != y_pred_class, y_true == 1).sum()

        A = float(total['tp']) / sum(total[key] for key in ['tp', 'fn'])
        B = float(total['tn']) / sum(total[key] for key in ['fp', 'tn'])
        return (A + B) / 2.0

    # Load network weights
    for idx in range(ENSEMBLE_MEMBERS):

        print "Predicting with ensemble member: %d" % idx

        weights = pickle.load(open(load_path % (train_name, idx, classname)))
        lasagne.layers.set_all_param_values(net['prob'], weights)

        #######################
        # TESTING
        for fname, spec, y in zip(test_files, test_X, test_y):

            probas = []
            y_true = []

            for Xb, yb in test_sampler([spec], [y]):
                probas.append(pred_fn(Xb))
                y_true.append(yb)

            y_preds_proba[fname].append(np.vstack(probas))
            y_gts[fname] = np.hstack(y_true)

        all_pred_prob = np.vstack(y_preds_proba[fname][-1] for fname in test_files)
        all_pred = np.argmax(all_pred_prob, axis=1)
        all_gt = np.hstack(y_gts[fname] for fname in test_files)
        print bal_acc(all_gt, all_pred)

    # aggregating ensemble members
    all_probs = []
    all_gt = []

    for fname in test_files:

        combined_preds_prob = np.stack(y_preds_proba[fname]).mean(0)
        combined_preds = np.argmax(combined_preds_prob, axis=1)
        y_gt = y_gts[fname]

        with open(predictions_savedir + fname, 'w') as f:
            pickle.dump([y_gt, combined_preds_prob], f, -1)

        all_probs.append(combined_preds)
        all_gt.append(y_gt)

    y_pred_class = np.hstack(all_probs)
    y_true = np.hstack(all_gt)
    print "Combined:", bal_acc(y_true, y_pred_class)


if __name__ == '__main__':
    params = dict(
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
        MAX_EPOCHS = 30,
        LEARNING_RATE = 0.001,

        CLASSNAME = classname
        )
    params['B'] = 10.0 if params['CLASSNAME'] == 'biotic' else 2.00
    params['A'] = 0.001 if params['CLASSNAME'] == 'biotic' else 0.025

    predict(**params)
