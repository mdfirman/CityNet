# testing normalisation strategies...

# General imports
import numpy as np
import os
import sys

# CNN bits
# import theano
import urban8k_helpers as helpers
import perform_run
# import lasagne

# Evaluation
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation, audio_utils, cnn_utils

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting parameters!!
base_path = '/media/michael/Seagate/urban8k/'
slice_width = 128
slices = True

#########################
# KEY PARAMS
num_epochs = 256
small_dataset = False

# should make minibatch size a multiple of 10 really
minibatch_size = 80 # optimise
augment_data = True

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

##################################
# Model params (as loaded from one row of a hyperopt file)

params = [0.0002791786739, 0.15, 40, 0.04, 0.02271767800627,
            5, 78, 2, 6, 3, 0.58768842676161, 3, 718, False, False, True, True]

###############################################################
# Overall setup

"""
Normalisation strategies
========================

1. per-row median subtraction
2. overall median subtraction
3. make whole spec sum to one - careful! normalise per length
4. make whole spec of length one
5. make exp of whole spec sum to one
6. make whole spec sum to one then do median subtraction (of some type)
7. local nomlisation kagle wahles pipeline job
    gaussian filter, box filter...? gaussianfilkter can cause rining
"""

# using the split which gave the median performance across all splits
split_num = 2

# global_dir = helpers.create_numbered_folder('results/normalisation_run_%04d/')
global_dir = 'results/normalisation_run_0000/'


#
strategies = ['full_whiten']
            # None, 'stowell_half', 'stowell_half_rescale', 'stowell_full',
            #   'overall_median', 'overall_median_rescale', 'sum_to_one',
            #   'equal_power', 'local_normalisation']

for count, normalisation_strategy in enumerate(strategies):

    print "\n\n***** RUN %d *****" % count

    print 'Performing split_%d_%s' % (split_num, normalisation_strategy)

    # loading the data
    loadpath = base_path + 'splits_128/split' + str(split_num) + '.pkl'
    data, num_classes = helpers.load_data(
        loadpath,
        normalisation=normalisation_strategy,
        small_dataset=small_dataset)

    run_directory_name = 'split_%d_%s' % (split_num, normalisation_strategy)
    perform_run.run(
        global_dir,
        network_input_size=network_input_size,
        num_classes=num_classes,
        data=data,
        num_epochs=num_epochs,
        do_median_normalise=False,
        minibatch_size=minibatch_size,
        params=params,
        augment_data=augment_data,
        run_directory_name=run_directory_name)

    # Don't need to save anything here, as perform_run.run should do all of this for us