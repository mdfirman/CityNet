# testing normalisation strategies...

# General imports
import numpy as np
import os
import sys
import yaml
import itertools

# CNN bits
import urban8k_helpers as helpers
import perform_run

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
small_dataset = True

# should make minibatch size a multiple of 10 really
minibatch_size = 80 # optimise
augment_data = True

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

##################################
# Loading model params
params = yaml.load(open('default_params.yaml'))

# using the split which gave the median performance across all splits
split_num = 3

global_dir = helpers.create_numbered_folder('results/augment_run_%04d/')


augmentation_strategies = itertools.product((True, False), repeat=3)

for count, augment in enumerate(augmentation_strategies):

    print "\n\n***** RUN %d *****" % count

    print 'Performing split_%d_%s' % (split_num, augment)

    params['augment_flip'] = augment[0]
    params['augment_roll'] = augment[1]
    params['augment_vol_ramp'] = augment[2]

    # loading the data
    loadpath = base_path + 'splits_128/split' + str(split_num) + '.pkl'
    data, num_classes = helpers.load_data(
        loadpath,
        normalisation='stowell_half',
        normalisation_params=(
            params['norm_mean_std'], params['norm_std_std']),
        small_dataset=small_dataset)

    run_directory_name = 'split_%d_%d/' % (split_num, count)

    perform_run.run(
        global_dir,
        network_input_size=network_input_size,
        num_classes=num_classes,
        data=data,
        params=params)

    np.savetxt(global_dir + run_directory_name + 'params.txt', params)
    # Don't need to save anything here, as perform_run.run should do all of this for us

