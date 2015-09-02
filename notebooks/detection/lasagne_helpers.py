import numpy as np
from sklearn.cross_validation import train_test_split
import scipy
from sklearn import metrics
import lasagne

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


def iterate_balanced_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)

    indices_pos = np.where(targets==1)[0]
    indices_neg = np.where(targets==0)[0]

    np.random.shuffle(indices_pos)
    np.random.shuffle(indices_neg)

    for start_idx in range(0, len(indices_neg) - batchsize + 1, batchsize):
        # in each batch, new negative data is provided, positive data is reused

        # get indices for each of the excerpts, wrapping back to the beginning...
        excerpt_pos = np.take(
            indices_pos, np.arange(start_idx, start_idx + batchsize), mode='wrap')
        excerpt_neg = np.take(
            indices_neg, np.arange(start_idx, start_idx + batchsize), mode='wrap')

        # reform the full balanced inputs and output
        full_idxs = np.hstack((excerpt_pos, excerpt_neg))
        yield inputs[full_idxs], targets[full_idxs]


def load_dataset(loadpath):
    data_big = scipy.io.loadmat(loadpath)

    # dealing with the problem with not being able to save big variables
    if not 'X_train' in data_big:
        data_big['X_train'] = np.vstack(data_big['X_train_split'])

    X_train_val = data_big['X_train']
    y_train_val = data_big['y_train'].ravel()
    X_test = data_big['X_test']
    y_test = data_big['y_test'].ravel()

    train_idxs, val_idxs = train_test_split(
        np.arange(X_train_val.shape[0]), test_size=0.3)

    X_train = X_train_val[train_idxs]
    y_train = y_train_val[train_idxs]
    X_val = X_train_val[val_idxs]
    y_val = y_train_val[val_idxs]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(
        input_var=None,
        im_width=-1,
        im_height=-1,
        drop_input=.0,
        conv_depth = 1,
        num_filters1=32,
        num_filters2=32,
        pool_size = 2,
        dense_depth=2,
        dense_width=800,
        drop_dense_hidden=.5):

    print drop_input, conv_depth, pool_size

    # using same pool size for each max pool
    print im_width, im_height
    very_leaky_rectify = lasagne.nonlinearities.very_leaky_rectify

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):

    network = lasagne.layers.InputLayer(shape=(None, 1, im_width, im_height),
                                        input_var=input_var)

    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # Conv layers (fixing to two sets of layers, each with 2*conv
    # and a max pool)
    for _ in range(int(conv_depth)):
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=int(num_filters1),
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

        network = lasagne.layers.Conv2DLayer(
            network, num_filters=int(num_filters2),
            filter_size=(3, 3),
            stride=(2, 2),
            nonlinearity=very_leaky_rectify,
            W=lasagne.init.GlorotUniform())

        network = lasagne.layers.MaxPool2DLayer(
            network, pool_size=(int(pool_size), int(pool_size)))

# apparently this is tricky to get right as should be applied somehow, ask daniel
#         if drop_conv:
#             network = lasagne.layers.dropout(network, p=drop_conv)

    # Dense layers and dropout
    for _ in range(int(dense_depth)):
        network = lasagne.layers.DenseLayer(
                network, int(dense_width), nonlinearity=very_leaky_rectify)
        if drop_dense_hidden:
            network = lasagne.layers.dropout(network, p=drop_dense_hidden)

    # Output layer:
    network = lasagne.layers.DenseLayer(network, 2,
        nonlinearity=lasagne.nonlinearities.softmax)
    return network


