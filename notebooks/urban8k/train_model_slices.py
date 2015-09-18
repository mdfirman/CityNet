import matplotlib.pyplot as plt
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

# CNN bits
import theano
import theano.tensor as T
import lasagne
# import lasagne.layers.cuda_convnet
from lasagne import layers

# for evaluation
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
theano.config.profile = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting parameters!!
base_path = '/media/michael/Seagate/urban8k/'
split = 2
slice_width = 128
slices = True

num_epochs = 100

small_dataset = False

# NOTE that the *actual* minibatch size will be something like num_classes*minibatch_size
minibatch_size = 128 # optimise

augment_data = True

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)


def load_data():
    # load in the data
    loadpath = base_path + 'splits_128/split' + str(split) + '.pkl'
    data = pickle.load(open(loadpath))

    num_classes = np.unique(data['train_y']).shape[0]
    print "There are %d classes " % num_classes
    print np.unique(data['train_y'])
    data['train_y'] = data['train_y'].ravel().astype(np.int32)
    data['test_y'] = data['test_y'].ravel().astype(np.int32)
    data['val_y'] = data['val_y'].ravel().astype(np.int32)

    for key, val in data.iteritems():
        if not key.startswith('__'):
            print key, len(val)

    # doing small sample...
    if small_dataset:
        for data_type in ['train_', 'test_', 'val_']:
            num = len(data[data_type + 'X'])
            to_use = np.random.choice(num, 100, replace=False)
            data[data_type + 'X'] = [data[data_type + 'X'][temp_idx] for temp_idx in to_use]
            data[data_type + 'y'] = data[data_type + 'y'][to_use]

    return data, num_classes


data, num_classes = load_data()


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    input_dropout = 0.18
    filter_sizes = 5
    num_filters = 50
    num_filter_layers = 2
    pool_size_x = 4
    pool_size_y = 4
    dense_dropout = 0.5
    num_dense_layers = 2
    num_dense_units = 800

    nonlin_choice = lasagne.nonlinearities.very_leaky_rectify

    # Input layer, followed by dropout
    network_shape = (None, 1, network_input_size[0], network_input_size[1])
    print "Making network of input size ", network_shape
    network = layers.InputLayer(shape=network_shape, input_var=input_var)
    network = layers.dropout(network, p=input_dropout)

    for _ in range(num_filter_layers):

        # see also: layers.cuda_convnet.Conv2DCCLayer
        network = layers.Conv2DLayer(
            network,
            num_filters=num_filters,
            filter_size=(filter_sizes, filter_sizes),
            # pad = (2, 2),
            # stride=(2, 2),
            nonlinearity=nonlin_choice,
            W=lasagne.init.GlorotUniform())

        network = layers.MaxPool2DLayer(
            network,
            pool_size=(pool_size_x, pool_size_y))

    for _ in range(num_dense_layers):

        network = layers.DenseLayer(
            layers.dropout(network, p=dense_dropout),
            num_units=num_dense_units,
            nonlinearity=nonlin_choice)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = layers.DenseLayer(
            layers.dropout(network, p=dense_dropout),
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# Prepare Theano variables for inputs and targets
def prepare_network():

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    reg_l2 = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
    # loss = loss + 0.0001 * reg_l2

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
            # loss, params, learning_rate=0.00249, momentum=0.5)
    updates = lasagne.updates.rmsprop(
             loss, params, learning_rate=0.00025) #0.000249

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
#     predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    predict_fn = theano.function([input_var], test_prediction)

    return network, train_fn, predict_fn, val_fn


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


def form_correct_shape_array(X):
    temp =  np.dstack(X).transpose((2, 0, 1))
    S = temp.shape
    temp = temp.astype(np.float32).reshape(S[0], 1, S[1], S[2])
    return temp


def augment_slice(slice_in):
    '''
    does some stuff to a single slice example to augment the dataset
    '''

    this_slice = slice_in.copy().astype(np.float64)

    # if the spectrogram is too small, consider either tiling it, or padding with zeros
    # todo

    # rolling the spectrogram
    n = slice_in.shape[1]
    roll_amount = np.random.randint(n)
    this_slice = np.roll(this_slice, roll_amount, axis=1)

    # # rotating
    # angle = np.random.rand() * 2
    # this_slice = skimage.transform.rotate(
    #     this_slice, angle=angle, mode='nearest')

    # # cropping in height - specify crop maximums as fractions
    # max_crop_top = 0.1
    # max_crop_bottom = 0.1
    # top_amount = max_crop_top * float(this_slice.shape[0])
    # bottom_amount = max_crop_bottom * float(this_slice.shape[0])
    # this_slice = this_slice[top_amount:-bottom_amount, :]

    # # cropping in length
    # max_crop_left = 0.1
    # max_crop_right = 0.1
    # left_amount = max_crop_left * float(this_slice.shape[1])
    # right_amount = max_crop_right * float(this_slice.shape[1])
    # this_slice = this_slice[:, left_amount:-right_amount]

    # # scaling
    # this_slice = skimage.transform.resize(this_slice, network_input_size)

    # volume ramping
    min_vol_ramp = 0.8
    max_vol_ramp = 1.2
    start_vol = np.random.rand() * (max_vol_ramp - min_vol_ramp) + min_vol_ramp
    end_vol = np.random.rand() * (max_vol_ramp - min_vol_ramp) + min_vol_ramp
    vol_ramp = np.linspace(start_vol, end_vol, this_slice.shape[1])
    this_slice = this_slice * vol_ramp[None, :]

    return this_slice.astype(np.float32)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = np.arange(start_idx, start_idx + batchsize)

        if slices:
            # take a single random slice from each of the training examples
            these_spectrograms = [inputs[xx][0, :, :] for xx in excerpt]
            Xs = [extract_slice(x) for x in these_spectrograms]
            Xs_to_return = form_correct_shape_array(Xs)
            yield Xs_to_return, targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt]


def iterate_balanced_minibatches_multiclass(inputs, targets, full_batchsize, shuffle=True):

    assert len(inputs) == len(targets)

    all_targets = np.unique(targets)

    idxs = {this_targ: np.where(targets==this_targ)[0]
            for this_targ in all_targets}

    if shuffle:
        for key in idxs.keys():
            np.random.shuffle(idxs[key])

    per_class_batchsize = full_batchsize / len(all_targets)

    # find the largest class - this will define the epoch size
    examples_in_epoch = max([len(x) for _, x in idxs.iteritems()])

    # in each batch, new data from largest class is provided
    # data from other class is reused once it runs out
    for start_idx in range(0, examples_in_epoch - per_class_batchsize + 1, per_class_batchsize):

        # get indices for each of the excerpts, wrapping back to the beginning...
        excerpts = []
        for target, this_target_idxs in idxs.iteritems():
            excerpts.append(np.take(
                this_target_idxs, np.arange(start_idx, start_idx + per_class_batchsize), mode='wrap'))

        # reform the full balanced inputs and output
        full_idxs = np.hstack(excerpts)

        if slices:
            # take a single random slice from each of the training examples
            these_spectrograms = [inputs[xx] for xx in full_idxs]
            Xs = [extract_slice(x) for x in these_spectrograms]
            if augment_data:
                Xs = map(augment_slice, Xs)
            yield form_correct_shape_array(Xs), targets[full_idxs]
        else:
            if augment_data:
                Xs = map(augment_slice, inputs[full_idxs])
            else:
                Xs = inputs[full_idxs]
            yield Xs, targets[full_idxs]


def form_slices_validation_set(data):

    max_validation_slices_per_spec = 1

    val_X = []
    val_y = []

    for this_x, this_y in zip(data['val_X'], data['val_y']):

        # choose how many slices to extract
        how_many = 1 #this_x.shape[1] / 8
        val_X += [extract_slice(this_x) for _ in range(how_many)]
        val_y += [this_y] * how_many

    val_y = np.hstack(val_y)
    val_X = form_correct_shape_array(val_X)

    print "validation set is of size ", val_X.shape, val_y.shape

    return val_X, val_y



# form a proper validation set here...
val_X, val_y = form_slices_validation_set(data)

network, train_fn, predict_fn, val_fn = prepare_network()

print "Starting training..."
print "There will be %d minibatches per epoch" % (data['train_y'].shape[0] / (minibatch_size*num_classes))

headerline = """     epoch   train loss   train/val   valid auc   valid acc     dur
   -------  -----------  -----------  -----------  -----------  -------"""
print headerline,
sys.stdout.flush()

best_validation_accuracy = 0.0
best_model = None

out_fid = open('results_slices.txt', 'w')
out_fid.write(headerline)

for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_loss = train_batches = 0
    start_time = time.time()

    print "Minibatch: ",
    for count, batch in enumerate(iterate_balanced_minibatches_multiclass(
            data['train_X'], data['train_y'], int(minibatch_size), shuffle=True)):
        if count % 100 == 0:
            print '.',
        inputs, targets = batch
        # print inputs.shape, targets.shape, inputs.max(), targets.max(), inputs.min(), targets.min(), inputs.dtype, targets.dtype
        train_loss += train_fn(inputs, targets)
        train_batches += 1
        sys.stdout.flush()

    train_loss = train_loss / train_batches
    print " total = ", train_batches,

    # And a full pass over the validation data:
    val_err = val_acc = val_batches = 0
    y_preds = []
    y_gts = []

    # doing the normal validation
    for batch in iterate_minibatches(val_X, val_y, int(minibatch_size/6)):

        inputs, targets = batch
        # print inputs.shape, targets.shape

        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

        y_preds.append(predict_fn(inputs))
        y_gts.append(targets)

    class_predictions = np.argmax(np.vstack(y_preds), axis=1)
    mean_val_accuracy = metrics.accuracy_score(np.hstack(y_gts), class_predictions)

    auc = metrics.roc_auc_score(np.hstack(y_gts), np.vstack(y_pred))

    val_loss = val_err / val_batches

    # # now doing the per-slice validation...
    # hww = slice_width/2
    # offset = 8
    # for this_val_x, this_val_y in zip(data['val_X'], data['val_y']):
    #     slice_preds = []
    #     for location in range(hww, this_val_x.shape[1], offset):
    #         temp_X = form_correct_shape_array([this_val_x[:, location-hww:location+hww]])
    #         print temp_X.shape
    #         slice_preds.append(predict_fn(temp_X))
    #     all_slice_preds = np.vstack(slice_preds).mean(0)

    #     y_preds.append(np.argmax(all_slice_preds))
    # slice_val_accuracy = metrics.accuracy_score(data['val_y'], np.hstack(y_preds))



#     mean_val_accuracy = val_acc / val_batches

    # Then we print the results for this epoch:
    results_row = "\n" + \
        "     " + str(epoch).ljust(8) + \
        ("%0.06f" % (train_loss)).ljust(12) + \
        ("%0.06f" % (train_loss / val_loss)).ljust(12) + \
        ("%0.06f" % (auc)).ljust(12) + \
        ("%0.06f" % (mean_val_accuracy)).ljust(10) + \
        ("%0.04f" % (time.time() - start_time)).ljust(10)
    print results_row,
    # sys.stdout.flush()

    # let's try saving a confusion matrix on each run...
    savepath = './conf_mat/%05d.mat' % epoch
    cm = metrics.confusion_matrix(np.hstack(y_gts), class_predictions)
    scipy.io.savemat(savepath, {'cm':cm})

    out_fid.write(results_row + "\n")

    if mean_val_accuracy > best_validation_accuracy:
        best_model = (network, predict_fn)
        best_validation_accuracy = mean_val_accuracy

        with open('best_model_slices.pkl', 'w') as f:
            pickle.dump(best_model, f)

out_fid.close()