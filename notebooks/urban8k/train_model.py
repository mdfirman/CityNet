import matplotlib.pyplot as plt
import csv
import os
import sys
import numpy as np
import collections
import scipy.io
import time
import cPickle as pickle

# CNN bits
import theano
import theano.tensor as T
import lasagne
import lasagne.layers.cuda_convnet
from lasagne import layers

# for evaluation
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
theano.config.profile = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# load in the data
base_path = '/home/michael/projects/engaged_hackathon_data/urban_8k/'
split = 1
loadpath = base_path + 'splits/split' + str(split) + '.mat'
data = scipy.io.loadmat(loadpath)

num_classes = np.unique(data['train_y']).shape[0]
print "There are %d classes " % num_classes
print np.unique(data['train_y'])
data['train_y'] = data['train_y'].ravel().astype(np.int32)
data['test_y'] = data['test_y'].ravel().astype(np.int32)
data['val_y'] = data['val_y'].ravel().astype(np.int32)

if False:
    for data_type in ['train_', 'test_', 'val_']:
        num = data[data_type + 'X'].shape[0]
        to_use = np.random.choice(num, 100, replace=False)
        data[data_type + 'X'] = data[data_type + 'X'][to_use, :]
        data[data_type + 'y'] = data[data_type + 'y'][to_use]

for key, val in data.iteritems():
    if not key.startswith('__'):
        print key, val.max(), val.min(), val.shape

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    _, _, im_width, im_height = data['train_X'].shape
    print im_width, im_height
    network = layers.InputLayer(shape=(None, 1, im_width, im_height),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    network = layers.dropout(network, p=0.18)

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    # network = layers.cuda_convnet.Conv2DCCLayer(
    #         network, num_filters=32, filter_size=(3, 3),
    #         # stride=(2, 2),
    #         nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
    #         W=lasagne.init.GlorotUniform())

    network = layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            # stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            # stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
#     network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = layers.MaxPool2DLayer(network, pool_size=(4, 8))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3,3),
            # stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

    network = layers.MaxPool2DLayer(network, pool_size=(2, 4))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = layers.DenseLayer(
            layers.dropout(network, p=.5),
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = layers.DenseLayer(
            layers.dropout(network, p=.5),
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = layers.DenseLayer(
            layers.dropout(network, p=.5),
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
    loss = loss + 0.0001 * reg_l2

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
            # loss, params, learning_rate=0.00249, momentum=0.5)
    updates = lasagne.updates.rmsprop(
             loss, params, learning_rate=0.000249)


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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
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

        # for now, sanity check...
        yield inputs[full_idxs], targets[full_idxs]


from copy import deepcopy
num_epochs = 100

# NOTE that the *actual* minibatch size will be num_classes*minibatch_size
minibatch_size = 40 # optimise
print "Warning - turn down to 6 if using..."

network, train_fn, predict_fn, val_fn = prepare_network()

print "Starting training..."
print "There will be %d minibatches per epoch" % (data['train_y'].shape[0] / (minibatch_size*num_classes))

headerline = """     epoch   train loss   valid loss   train/val    valid acc     dur
   -------  -----------  -----------  -----------  -----------  -------"""
print headerline,
sys.stdout.flush()

best_validation_accuracy = 0.0
best_model = None

out_fid = open('results.txt', 'w')
out_fid.write(headerline)

for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:
    train_loss = train_batches = 0
    start_time = time.time()

    print "Minibatch: ",
    for count, batch in enumerate(
            iterate_balanced_minibatches_multiclass(
            data['train_X'], data['train_y'], int(minibatch_size), shuffle=True)):
        if count % 100 == 0:
            print '.',
        inputs, targets = batch
        train_loss += train_fn(inputs, targets)
        train_batches += 1
        sys.stdout.flush()

    train_loss = train_loss / train_batches
    print " total = ", train_batches,

    # And a full pass over the validation data:
    val_err = val_acc = val_batches = 0
    y_preds = []
    y_gts = []

    for batch in iterate_minibatches(
                data['val_X'], data['val_y'], int(minibatch_size)):

        inputs, targets = batch

        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

        y_preds.append(predict_fn(inputs))
        y_gts.append(targets)


    class_predictions = np.argmax(np.vstack(y_preds), axis=1)
    mean_val_accuracy = metrics.accuracy_score(np.hstack(y_gts), class_predictions)

    val_loss = val_err / val_batches
#     mean_val_accuracy = val_acc / val_batches

    # Then we print the results for this epoch:
    results_row = "\n" + \
        "     " + str(epoch).ljust(8) + \
        ("%0.06f" % (train_loss)).ljust(12) + \
        ("%0.06f" % (0.0)).ljust(12) + \
        ("%0.06f" % (train_loss / val_loss)).ljust(12) + \
        ("%0.06f" % (mean_val_accuracy)).ljust(10) + \
        ("%0.04f" % (time.time() - start_time)).ljust(10) + "\n"
    print results_row,
    sys.stdout.flush()

    out_fid.write(results_row)

    if mean_val_accuracy > best_validation_accuracy:
        best_model = (network, predict_fn)
        best_validation_accuracy = mean_val_accuracy

        with open('best_model.pkl', 'w') as f:
            pickle.dump(best_model, f)

out_fid.close()