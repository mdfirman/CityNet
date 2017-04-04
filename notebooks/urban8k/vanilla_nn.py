import numpy as np
import os
import sys

# IO
import cPickle as pickle
import yaml

# CNN
import theano.tensor as T
import theano
import lasagne
# import lasagne.layers.cuda_convnet
from lasagne import layers

# sys.path.append('../')
import urban8k_helpers as helpers


nonlin_choice = lasagne.nonlinearities.leaky_rectify

params = yaml.load(open('default_params.yaml'))
params['learning_rate'] = 0.0001
params['num_filter_layers'] = 0
params['normalisation'] = ['sum_to_one']
params['augment_flip'] = False
params['augment_roll'] = False
params['dense_dropout'] = 0.1
params['num_dense_units'] = 1000
params['num_dense_layers'] = 4

input_var = T.tensor4('inputs')

# cnn = helpers.build_cnn(input_var, (1, 128), 1, 10, params)

split_num = 2
base_path = '/media/michael/Seagate/urban8k/'
loadpath = base_path + 'splits_128/split' + str(split_num) + '.pkl'
data, num_classes = helpers.load_data(loadpath)

for key in ['train_X', 'test_X', 'val_X']:
    for types in ['_median', '_mean', '_var', '_max']:
        for idx in range(len(data[key + types])):
            data[key + types][idx] = data[key + types][idx][None, :, None]

    full_set = (data[key + '_mean'], data[key + '_median'], data[key + '_var'], data[key + '_max'])
    data[key] = np.concatenate(full_set, axis=2)
    size = data[key].shape[2]
    print "Size is ", size

# print len(data['train_X'])
# print data['train_X'][0].shape
# print data['val_X'][0].shape

# import cPickle as pickle
# pickle.dump(data, open('./tmp.pkl', 'w'), -1)
# sds

# print len(data['train_X_median'])
# print data['train_y'].shape

for A, B in helpers.generate_balanced_minibatches_multiclass(
    data['train_X_median'], data['train_y'],
    int(params['minibatch_size']), 128, augment_options={},
    shuffle=True):
    print A.shape, B.shape
    break


import perform_run
perform_run.run('./results/vanilla_nn/', (1, size), 10, data, params)