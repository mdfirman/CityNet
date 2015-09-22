import cPickle as pickle
import numpy as np
import theano.tensor as T
import theano
import lasagne
# import lasagne.layers.cuda_convnet
from lasagne import layers
import skimage.transform

import sys
sys.path.append('/home/michael/projects/engaged_hackathon/engaged/features/')
import audio_utils


def load_data(loadpath, do_median_normalise=True, small_dataset=False):
    '''
    Loading in all the data at once, and get into the correct format.
    '''
    print "Loading data"
    data = pickle.load(open(loadpath))

    num_classes = np.unique(data['train_y']).shape[0]

    print "There are %d classes " % num_classes
    print np.unique(data['train_y'])

    for key, val in data.iteritems():

        if not key.startswith('__'):
            print key, len(val),

        # normalisation
        if key.endswith('_X'):
            if do_median_normalise:
                for idx in range(len(data[key])):
                    data[key][idx] = audio_utils.median_normalise(data[key][idx])

        # ensuring correct types
        elif key.endswith('_y'):
            data[key] = data[key].ravel().astype(np.int32)

    # doing small sample...
    if small_dataset:
        for data_type in ['train_', 'test_', 'val_']:
            num = len(data[data_type + 'X'])
            to_use = np.random.choice(num, 100, replace=False)
            data[data_type + 'X'] = [data[data_type + 'X'][temp_idx] for temp_idx in to_use]
            data[data_type + 'y'] = data[data_type + 'y'][to_use]

    return data, num_classes


def build_cnn(input_var, network_input_size, num_classes):
    '''
    Function to programmatically build a CNN in lasagne.
    TODO - give architechture as input parameters.
    '''
    input_dropout = 0.1
    filter_sizes = 5
    num_filters = 40
    num_filter_layers = 2
    pool_size_x = 4
    pool_size_y = 2
    dense_dropout = 0.5
    num_dense_layers = 3
    num_dense_units = 800
    inital_filter_layer = False

    nonlin_choice = lasagne.nonlinearities.very_leaky_rectify

    # Input layer, followed by dropout
    network_shape = (None, 1, network_input_size[0], network_input_size[1])
    print "Making network of input size ", network_shape
    network = layers.InputLayer(shape=network_shape, input_var=input_var)
    network = layers.dropout(network, p=input_dropout)

    if inital_filter_layer:
        network = layers.Conv2DLayer(
            network,
            num_filters=num_filters,
            filter_size=(filter_sizes, filter_sizes),
            nonlinearity=nonlin_choice,
            W=lasagne.init.GlorotUniform())

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
def prepare_network(learning_rate, network_input_size, num_classes):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    print network_input_size, num_classes
    network = build_cnn(input_var, network_input_size, num_classes)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # reg_l2 = lasagne.regularization.regularize_network_params(
        # network, lasagne.regularization.l2)
    # loss = loss + 0.0001 * reg_l2

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.000249, momentum=0.5)
    updates = lasagne.updates.rmsprop(
             loss, params, learning_rate=learning_rate) #0.000249

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var),
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


def threaded_gen(generator, num_cached=50):
    '''
    Threaded generator to multithread the data loading pipeline
    code from daniel, he got it from a chatroom or something...
    '''
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def augment_slice(slice_in, roll=True, rotate=False, flip=True,
        volume_ramp=True, normalise=True):
    '''
    does some stuff to a single slice example to augment the dataset
    '''

    this_slice = slice_in.copy().astype(np.float64)

    # if the spectrogram is too small, consider either tiling it, or padding with zeros
    # todo

    if roll:
        # rolling the spectrogram
        n = slice_in.shape[1]
        roll_amount = np.random.randint(n)
        this_slice = np.roll(this_slice, roll_amount, axis=1)

    if rotate:
        angle = np.random.rand() * 2
        this_slice = skimage.transform.rotate(
            this_slice, angle=angle, mode='nearest')

    # # cropping in height - specify crop maximums as fractions
    # max_crop_top = 0.05
    # max_crop_bottom = 0.05
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

    if flip:
        if np.random.rand() > 0.5:
            this_slice = this_slice[:, ::-1]

    if volume_ramp:
        min_vol_ramp = 0.8
        max_vol_ramp = 1.2
        start_vol = np.random.rand() * (max_vol_ramp - min_vol_ramp) + min_vol_ramp
        end_vol = np.random.rand() * (max_vol_ramp - min_vol_ramp) + min_vol_ramp
        vol_ramp = np.linspace(start_vol, end_vol, this_slice.shape[1])
        this_slice = this_slice * vol_ramp[None, :]

    if normalise:
        slice_in = audio_utils.median_normalise(slice_in)

    return this_slice.astype(np.float32)


class Logger(object):
    def __init__(self, fname):
        self.out_fid = open(fname, 'w')
        headerline = """     epoch   train loss   train/val   valid auc   valid acc     dur
                    -------  -----------  -----------  -----------  -----------  -------"""
        self.out_fid.write(headerline)

    def log(self, tolog):
        self.out_fid.write(tolog)
        print tolog,

    def __del__(self):
        self.out_fid.close()


# def force_slice_length(spec, location, slice_width):
#     '''
#     extract a slice from a specific location, but if there isn't enough spectrogram to
#     go around then maybe do something else... e.g. wrapping
#     '''
#     hww = slice_width / 2
#     to_return = spec[:, location-hww:location+hww]
#     return tile_pad(to_return, slice_width)


def tile_pad(this_slice, desired_width):
    if this_slice.shape[1] == desired_width:
        return this_slice
    else:
        num_tiles = np.ceil(float(desired_width) / this_slice.shape[1])
        tiled = np.tile(this_slice, (1, num_tiles))
        return tiled[:, :desired_width]
