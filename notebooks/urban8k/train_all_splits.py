# General imports
import numpy as np
import os
import sys
import time

# IO
import scipy.io
import cPickle as pickle
import yaml
import csv

# CNN bits
import theano
import urban8k_helpers as helpers
import perform_run
import lasagne

# Evaluation
from sklearn import metrics
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation, audio_utils, cnn_utils

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting parameters!!
base_path = '/media/michael/Seagate/urban8k/'
slice_width = 128

#########################
# KEY PARAMS
small_dataset = False

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

##################################
# Model params (as loaded from one row of a hyperopt file)

params = yaml.load(open('default_params.yaml'))

###############################################################
# Overall setup
global_dir = helpers.create_numbered_folder('results/full_split_run_%04d/')

for option in ['train_and_val_data', 'train_data']:

    for split_num in range(1, 11):

        print 'Performing split_%d_%s' % (split_num, option)

        # loading the data
        loadpath = base_path + 'splits_128/split' + str(split_num) + '.pkl'
        data, num_classes = helpers.load_data(
            loadpath,
            normalisation='stowell_half',
            small_dataset=small_dataset)

        if option == 'train_data':
            pass # just leave as is
        elif option == 'train_and_val_data':
            # Combine the two together.
            # NB This means that choosing the 'best' model doesn't really work
            # So for this option should always pick the last model!
            # *BUT* verify that we are not overfitting using the 'train_data'
            # equivalent option
            data['train_X'] = data['train_X'] + data['val_X']
            data['train_y'] = np.hstack((data['train_y'], data['val_y']))
        else:
            raise Exception('Unknown option')

        run_directory_name = 'split_%d_%s' % (split_num, option)
        perform_run.run(
            global_dir,
            network_input_size=network_input_size,
            num_classes=num_classes,
            data=data,
            params=params,
            run_directory_name=run_directory_name)

        # Don't need to save anything here, as perform_run.run should do all of this for us