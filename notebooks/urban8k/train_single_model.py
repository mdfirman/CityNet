# testing normalisation strategies...

# General imports
import numpy as np
import os
import sys
import yaml

# CNN bits
import urban8k_helpers as helpers
import perform_run

# Evaluation
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation, audio_utils, cnn_utils

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

base_path = '/media/michael/Seagate/urban8k/'

#########################
# KEY PARAMS
small_dataset = False

# what size will the CNN get ultimately? - optimise this!
slice_width = 128
network_input_size = (128, slice_width)

##################################
# Model params
params = yaml.load(open('default_params.yaml'))

###############################################################
# Overall setup

# using the split which gave the median performance across all splits
split_num = 2

# global_dir = helpers.create_numbered_folder('results/normalisation_run_%04d/')
global_dir = 'results/single_run_0001/'
normalisation_strategy = 'stowell_half'

print 'Performing split_%d' % (split_num)

# loading the data
loadpath = base_path + 'splits_128/split' + str(split_num) + '.pkl'
data, num_classes = helpers.load_data(
    loadpath,
    normalisation=params['normalisation'],
    small_dataset=small_dataset)

run_directory_name = 'split_%d_%s' % (split_num, normalisation_strategy)

perform_run.run(
    global_dir,
    network_input_size=network_input_size,
    num_classes=num_classes,
    data=data,
    params=params,
    run_directory_name=run_directory_name)

# Don't need to save anything here, as perform_run.run should do all of this for us
