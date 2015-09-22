import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import sys
import numpy as np
import collections
import scipy.io
import time
import cPickle as pickle
from copy import deepcopy
import skimage.transform
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# CNN bits
import theano

# for evaluation
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation, audio_utils, cnn_utils


import urban8k_helpers as helpers

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
theano.config.profile = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting parameters!!
base_path = '/media/michael/Seagate/urban8k/'
split = 3
slice_width = 128
slices = True
num_epochs = 500
small_dataset = False

# NOTE that the *actual* minibatch size will be something like num_classes*minibatch_size
minibatch_size = 100 # optimise
augment_data = True

augment_options = {
    'roll': True,
    'rotate': False,
    'flip': False,
    'volume_ramp': False,
    'normalise': False
    }

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

learning_rate = 0.00025

# loading the data
loadpath = base_path + 'splits_128/split' + str(split) + '.pkl'
data, num_classes = helpers.load_data(
    loadpath, small_dataset=small_dataset, do_median_normalise=True)


def extract_slice(spec):
    # takes a single random slice from a single spectrogram
    assert slice_width % 2 == 0
    hww = slice_width / 2
    if spec.shape[1] > 2*hww:
        idx = np.random.randint(hww, spec.shape[1]-hww)
        # print "Not tiling", spec[:, idx-hww:idx+hww].shape
        return spec[:, idx-hww:idx+hww]
    else:
        # tile wrap the spec and return the whole thing!
        num_tiles = np.ceil(float(slice_width) / spec.shape[1])
        tiled = np.tile(spec, (1, num_tiles))
        # print "Tiling", tiled[:, :slice_width].shape
        return tiled[:, :slice_width]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs), batchsize):

        end_idx = min(start_idx + batchsize, len(inputs))

        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = np.arange(start_idx, end_idx)

        # take a single random slice from each of the training examples
        these_spectrograms = [inputs[xx] for xx in excerpt]
        Xs = [helpers.tile_pad(x[0], slice_width) for x in these_spectrograms]
        Xs_to_return = cnn_utils.form_correct_shape_array(Xs)
        yield Xs_to_return, targets[excerpt]

# def generate_balanced_minibatches_multiclass2(
#         inputs, targets, items_per_minibatch, shuffle=True):

#     assert len(inputs) == len(targets)

#     all_targets = np.unique(targets)

#     def class_generator(class_idx):
#         '''
#         endlessly generates training pairs for a specfic class
#         '''
#         class_exemplars = np.where(targets==class_idx)[0]

#         todo - loop
#         for idx in class_exemplars:
#             yield inputs[idx], targets[idx]

#     largest_class_size = np.max(np.bincount(targets))

#     per_class_per_minibatch = items_per_minibatch / len(all_targets)
#     num_minibatches =


def generate_balanced_minibatches_multiclass(
        inputs, targets, items_per_minibatch, shuffle=True):

    assert len(inputs) == len(targets)

    all_targets = np.unique(targets)

    idxs = {this_targ: np.where(targets==this_targ)[0]
            for this_targ in all_targets}

    if shuffle:
        for key in idxs.keys():
            np.random.shuffle(idxs[key])

    num_per_class_per_minibatch = items_per_minibatch / len(all_targets)

    # find the largest class - this will define the epoch size
    examples_in_epoch = max([len(x) for _, x in idxs.iteritems()])

    # in each batch, new data from largest class is provided
    # data from other class is reused once it runs out
    for start_idx in range(0,
                           examples_in_epoch,
                           num_per_class_per_minibatch):

        end_idx = min(start_idx + num_per_class_per_minibatch, len(inputs))

        # get indices for each of the excerpts, wrapping back to the beginning...
        excerpts = []
        for target, this_target_idxs in idxs.iteritems():
            these_idxs = np.take(this_target_idxs,
                np.arange(start_idx, end_idx), mode='wrap')
            excerpts.append(these_idxs)

        # for each of the training indices for this minibatch, extract and
        # pre-process a training instance
        training_images = []
        full_idxs = np.hstack(excerpts)

        for idx in full_idxs:
            this_image = inputs[idx]
            this_image = helpers.tile_pad(this_image, slice_width)
            if augment_data:
                this_image = helpers.augment_slice(this_image, **augment_options)
            training_images.append(this_image)

        yield (cnn_utils.form_correct_shape_array(training_images), targets[full_idxs])


def form_slices_validation_set(data):

    val_X = [helpers.tile_pad(xx, slice_width) for xx in data['val_X']]
    val_X = [audio_utils.median_normalise(xx) for xx in val_X]

    val_y = np.hstack(data['val_y'])
    val_X = cnn_utils.form_correct_shape_array(val_X)

    print "validation set is of size ", val_X.shape, val_y.shape

    return val_X, val_y


# form a proper validation set here...
val_X, val_y = form_slices_validation_set(data)

network, train_fn, predict_fn, val_fn = \
    helpers.prepare_network(learning_rate, network_input_size, num_classes)

print "Starting training..."
print "There will be %d minibatches per epoch" % \
    (data['train_y'].shape[0] / (minibatch_size*num_classes))

headerline = """     epoch   train loss   train/val   valid auc   valid acc     dur
   -------  -----------  -----------  -----------  -----------  -------"""
print headerline,
sys.stdout.flush()

best_validation_accuracy = 0.0
best_model = None
run_results = []

this_run_dir = 'results/test_dir/'
if not os.path.exists(this_run_dir):
    os.mkdir(this_run_dir)

if not os.path.exists(this_run_dir + '/conf_mat/'):
    os.mkdir(this_run_dir + '/conf_mat/')

out_fid = open(this_run_dir + 'results_slices.txt', 'w')
out_fid.write(headerline)

for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_loss = train_batches = 0
    start_time = time.time()

    # we now create the generator for this epoch
    epoch_gen = generate_balanced_minibatches_multiclass(
            data['train_X'], data['train_y'], int(minibatch_size), shuffle=True)
    threaded_epoch_gen = helpers.threaded_gen(epoch_gen)

    results = {}

    ########################################################################
    # TRAINING PHASE

    for count, batch in enumerate(threaded_epoch_gen):
        inputs, targets = batch
        train_loss += train_fn(inputs, targets)
        train_batches += 1

        # print a progress bar
        if count % 10 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

    # computing the training loss
    results['train_loss'] = train_loss / train_batches
    print " total = ", train_batches,

    results['train_time'] = time.time() - start_time

    ########################################################################
    # VALIDATION PHASE

    start_time = time.time()
    batch_val_loss = val_acc = val_batches = 0
    y_preds = []
    y_gts = []

    # doing the normal validation
    val_sum = 0
    for batch in iterate_minibatches(val_X, val_y, 8):

        inputs, targets = batch

        err, acc = val_fn(inputs, targets)
        batch_val_loss += err
        val_acc += acc
        val_batches += 1

        val_sum += targets.shape[0]

        y_preds.append(predict_fn(inputs))
        y_gts.append(targets)

    probability_predictions = np.vstack(y_preds)
    class_predictions = np.argmax(probability_predictions, axis=1)

    results['auc'] = cnn_utils.multiclass_auc(
        np.hstack(y_gts), probability_predictions)
    results['mean_val_accuracy'] = np.array(val_acc).sum() / val_sum
    results['val_loss'] = batch_val_loss / val_batches
    results['val_time'] = time.time() - start_time

    ########################################################################
    # LOGGING AND PLOTTING

    # Then we print the results for this epoch:
    results_row = "\n" + \
        "     " + str(epoch).ljust(6) + \
        ("%0.06f" % (train_loss)).ljust(10) + \
        ("%0.06f" % (val_loss)).ljust(10) + \
        ("%0.06f" % (train_loss / val_loss)).ljust(10) + \
        ("%0.06f" % (mean_val_accuracy)).ljust(10) + \
        ("%0.06f" % (auc)).ljust(10) + \
        ("%0.04f" % (time.time() - start_time)).ljust(10)
    print results_row,
    # sys.stdout.flush()

    run_results.append(
        {'train_loss': train_loss,
         'val_loss': val_loss,
         'mean_val_accuracy': mean_val_accuracy,
         'auc': auc,
         'time': 0.0})

    # let's try saving a confusion matrix on each run... (maybe just add this to a run vector)
    savepath = this_run_dir + '/conf_mat/%05d.mat' % epoch
    cm = metrics.confusion_matrix(np.hstack(y_gts), class_predictions)
    scipy.io.savemat(savepath, {'cm':cm})

    out_fid.write(results_row + "\n")

    # saving the model, if we are the best so far
    if mean_val_accuracy > best_validation_accuracy:
        best_model = (network, predict_fn)
        best_validation_accuracy = mean_val_accuracy

        with open(this_run_dir + 'best_model_slices.pkl', 'w') as f:
            pickle.dump(best_model, f)

    # updating a training/val loss graph...
    all_train_losses = [result['train_loss'] for result in run_results]
    all_val_losses = [result['val_loss'] for result in run_results]

    graph_savepath = this_run_dir + 'training_graph.png'
    plt.plot(all_train_losses, label='train_loss')
    plt.plot(all_val_losses, label='val_loss')
    plt.savefig(graph_savepath, bbox_inches='tight')



out_fid.close()
