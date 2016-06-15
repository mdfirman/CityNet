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

RUN_TYPE = 'standard_spec'

# loading data options
TRAINING_DATA = 'golden'
TEST_FOLD = 1
CLASSNAME = 'biotic'
SPEC_TYPE = 'mel'
SPEC_HEIGHT = 32

# data preprocessing options
A = 0.001
B = 10.0
HWW = 5
LEARN_LOG = 0
DO_AUGMENTATION = 1

# network parameters
DO_BATCH_NORM = 1
NUM_FILTERS = 32
NUM_DENSE_UNITS = 128
CONV_FILTER_WIDTH = 4
WIGGLE_ROOM = 5
MAX_EPOCHS = 50
LEARNING_RATE = 0.0005

logging_dir = data_io.base + 'predictions/%s/%s/' % (RUN_TYPE, CLASSNAME)
train_helpers.force_make_dir(logging_dir)
sys.stdout = ui.Logger(logging_dir + 'log.txt')

# loading data
if TRAINING_DATA == 'golden':
    train_files, test_files = data_io.load_splits(TEST_FOLD)
else:
    raise Exception("Not implemented!")

train_X, train_y = data_io.load_data(train_files, SPEC_TYPE, SPEC_HEIGHT, LEARN_LOG, CLASSNAME, A, B)
test_X, test_y = data_io.load_data(test_files, SPEC_TYPE, SPEC_HEIGHT, LEARN_LOG, CLASSNAME, A, B)

# # creaging samplers and batch iterators
train_sampler = SpecSampler(64, HWW, DO_AUGMENTATION, LEARN_LOG, randomise=True)
test_sampler = SpecSampler(64, HWW, False, LEARN_LOG, randomise=True, seed=10)


class MyTrainSplit(nolearn.lasagne.TrainSplit):
    # custom data split
    def __call__(self, Xb, Yb, net):
        return train_X, test_X, train_y, test_y


net = train_helpers.create_net(SPEC_HEIGHT, HWW, LEARN_LOG, NUM_FILTERS,
    WIGGLE_ROOM, CONV_FILTER_WIDTH, NUM_DENSE_UNITS, DO_BATCH_NORM)

save_history = train_helpers.SaveHistory(logging_dir)

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
    on_epoch_finished=[save_weights, save_history],
    check_input=False
)
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

# we could now do one per file... each spectrogram at a time... We could do the full plotting etc
# probably not worth it.
# let's do this another time
