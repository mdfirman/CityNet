import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import nolearn
import nolearn.lasagne
import lasagne.layers
from ml_helpers import minibatch_generators as mbg
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, DimshuffleLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, elu
import socket
from helpers import Log1Plus, MyTrainSplit, MyBatch, force_make_dir
from ml_helpers import evaluation
import librosa

if socket.gethostname() == 'biryani':
    base = '/media/michael/Seagate/engage/urban8k/'
else:
    base = '/home/mfirman/Data/audio/urban8k/'

base_logging_dir = base + 'dopey_runs/their_specs/'

# loading data
all_data = pickle.load(open(base + 'paper_specs.pkl'))

# get class labels
mapper = dict(zip(folds[1]['classID'], folds[1]['class']))
cls_labels = [mapper[xx] for xx in range(10)]


def mean_normalise(X):
    med = np.mean(X.reshape(X.shape[0], -1), 1)
    med[med==0] = 0.0001
    return X / med[:, None, None, None]


for val_fold, test_fold in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)):

    train_idxs = list(set(range(1, 11)) - set((val_fold, test_fold)))
    splits = {'train': train_idxs, 'val': [val_fold], 'test': [test_fold]}

    data = {'train': {'X': [], 'y': []},
            'val': {'X': [], 'y': []},
            'test': {'X': [], 'y': []}}

    for key, val in splits.iteritems():
        for fold_id in val:
            idxs = np.array(all_data['fold'] == fold_id)
            data[key]['X'].append(all_data['X'][idxs][:, None, :, :].astype(np.float32))
            data[key]['y'].append(all_data['y'][idxs])

        data[key]['X'] = np.vstack(data[key]['X']).astype(np.float32)
        data[key]['y'] = np.hstack(data[key]['y']).astype(np.int32)

    X_mean = np.mean(data['train']['X'])
    X_std = np.std(data['train']['X'])
    data['train']['X'] = (data['train']['X'] - X_mean) / X_std
    data['val']['X'] = (data['val']['X'] - X_mean) / X_std

    def generate_deltas(X):
        new_dim = np.zeros(np.shape(X))
        X = np.concatenate((X, new_dim), axis=1)
        del new_dim

        for i in range(len(X)):
            X[i, 1, :, :] = librosa.feature.delta(X[i, 0, :, :])

        return X

    for key in data:
        data[key]['X'] = generate_deltas(data[key]['X']).astype(np.float32)

    net = {}
    net['input'] = InputLayer((None, 2, 60, 41))
    net['conv1_1'] = ConvLayer(net['input'], 80, (57, 6), nonlinearity=vlr)
    net['pool1'] = PoolLayer(net['conv1_1'], pool_size=(4, 3), stride=(1, 3))
    net['pool1'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv1_2'] = ConvLayer(net['pool1'], 80, (1, 3), nonlinearity=vlr)
    net['pool2'] = PoolLayer(net['conv1_2'], pool_size=(1, 3), stride=(1, 1))
    net['fc6'] = DenseLayer(net['pool2'], num_units=512, nonlinearity=vlr)
    net['fc6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6'], num_units=512, nonlinearity=vlr)
    net['fc7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7'], num_units=10, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    logging_dir = base_logging_dir + 'val_fold_%02d/' % val_fold
    force_make_dir(logging_dir)

    network = nolearn.lasagne.NeuralNet(
        layers=net['prob'],
        max_epochs=300,
        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=0.002,
        verbose=1,
        train_split=MyTrainSplit(None),
        batch_iterator_train=MyBatch(batch_size=64)
    )

    network.initialize()
    network.fit(data, None)

    # save weights, conf mat etc
    network.save_params_to(logging_dir + 'params.pkl')

    # do test and validation conf mat and predictions
    for foldname in ['val', 'test']:
        y_pred = network.predict(data[foldname]['X'])
        y_gt = data[foldname]['y']
        plt.clf()
        evaluation.plot_confusion_matrix(y_gt, y_pred,
            title = foldname, cls_labels=cls_labels)
        plt.savefig(logging_dir + '%s_conf_mat.png' % foldname)

        with open('%s_predictions.pkl' % foldname, 'w') as f:
            pickle.dump({'y_pred': y_pred, 'y_gt': y_gt}, f, -1)

    # save the history
    with open(logging_dir + 'history.pkl', 'w') as f:
        pickle.dump(network.train_history_, f, -1)
