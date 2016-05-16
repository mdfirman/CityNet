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

if socket.gethostname() == 'biryani':
    base = '/media/michael/Seagate/engage/urban8k/'
else:
    base = '/home/mfirman/Data/audio/urban8k/'

base_logging_dir = base + 'dopey_runs/first_on_dopey/'

# loading data
folds = {}
for fold in range(1, 11):
    loadpath = base + 'specs_no_log/fold%d.pkl' % fold
    folds[fold] = pickle.load(open(loadpath))

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
            data[key]['X'] += folds[fold_id]['X']
            data[key]['y'] += list(folds[fold_id]['classID'])

        data[key]['X'] = mbg.form_correct_shape_array(data[key]['X']).astype(np.float32)
        data[key]['X'] = mean_normalise(data[key]['X'])
        data[key]['y'] = np.array(data[key]['y']).astype(np.int32)
        print data[key]['X'].min(), data[key]['X'].max(), data[key]['X'].mean()

    net = {}
    net['input'] = InputLayer((None, data['train']['X'].shape[1], 224, 224))
    net['input2'] = PoolLayer(net['input'], 2, mode='average_inc_pad')
    net['input_logged'] = Log1Plus(net['input2'])
    net['conv1_1'] = ConvLayer(net['input_logged'], 32, 3, nonlinearity=elu)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 32, 3, nonlinearity=elu)
    net['pool1'] = PoolLayer(net['conv1_2'], 4)
    net['conv2_1'] = ConvLayer(net['pool1'], 32, 3, nonlinearity=elu)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 32, 3, nonlinearity=elu)
    net['pool2'] = PoolLayer(net['conv2_2'], 4)
    net['fc6'] = DenseLayer(net['pool2'], num_units=256, nonlinearity=elu)
    net['fc6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6'], num_units=256, nonlinearity=elu)
    net['fc7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc7'], num_units=10, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    logging_dir = base_logging_dir + 'val_fold_%02d/' % val_fold
    force_make_dir(logging_dir)

    network = nolearn.lasagne.NeuralNet(
        layers=net['prob'],
        max_epochs=2,
        update=lasagne.updates.adam,
        update_learning_rate=0.001,
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
