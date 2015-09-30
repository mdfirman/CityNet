# General imports
import numpy as np
import os
import sys
import time
import collections

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials

# Evaluation
from sklearn import metrics
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import evaluation, audio_utils, cnn_utils

# https://groups.google.com/forum/#!topic/lasagne-users/t_rMTLAtpZo
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# setting parameters!!
base_path = '/media/michael/Seagate/urban8k/'
split = 3
slice_width = 128
slices = True

#########################
# KEY PARAMS
small_dataset = True
max_evals = 128  # hyperopt

# should make minibatch size a multiple of 10 really

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

###############################################################
# Overall setup
global_dir = helpers.create_numbered_folder('results/hyperopt_run_%04d/')

# defining default parameters, which will be overwritten if hyperopting
default_params = yaml.load(open('default_params.yaml'))

# set up the hyperopt search space as a dictionary first...
# (Do this before run_wrapper so it has access to the variable names)
hopt_dict = collections.OrderedDict((
     ('inital_filter_layer', (hp.choice, [False, False])),
     ('norm_mean_std', (hp.uniform, 0, 60)),
     ('norm_std_std', (hp.uniform, 0, 60))
))


def run_wrapper(hopt_params):
    """
    This is the wrappper which is minimised
    hopt_params
        vector of parameters from the hyperopt
    """
    global run_counter
    global o_f

    run_counter += 1
    print "\n****** RUN %d ******" % run_counter
    sys.stdout.flush()

    start_time = time.time()

    # use the default params, except where values have been defined in hopt_params:
    # (basically this is the magic to get around hyoperopt
    # a) not allowing defaults and
    # b) giving guesses as a vector not a dict )
    for key, val in zip(hopt_dict, hopt_params):
        default_params[key] =val

    # try:
    if True:
        # loading the data
        loadpath = base_path + 'splits_128/split' + str(split) + '.pkl'
        data, num_classes = helpers.load_data(
            loadpath, normalisation=default_params['normalisation'],
            small_dataset=small_dataset,
            normalisation_params=(
                default_params['norm_mean_std'], default_params['norm_std_std']))

        # form a proper validation set here...
        val_X, val_y = helpers.form_slices_validation_set(
            data, slice_width, default_params['do_median_normalise'])

        final_summary = perform_run.run(
            global_dir,
            network_input_size=network_input_size,
            num_classes=num_classes,
            data=data,
            params=default_params)

        print "\n\n",
        for key, val in final_summary.iteritems():
            print "%s : %0.4f" % (key, val),
        print "took %0.3fs" % (time.time() - start_time)

        writer.writerow([
            final_summary['best_val_loss'],
            final_summary['best_val_acc'],
            final_summary['final_val_acc']
            ] + list( default_params.itervalues() ))
        o_f.flush()

        return {
            'loss': final_summary['best_val_loss'],
            'status': STATUS_OK
            }

    # except Exception as e:
    else:

        print "Failed run: ", str(e)
        print "Took %0.3fs" % (time.time() - start_time)

        writer.writerow(['FAILURE', 'FAILURE', 'FAILURE'] + list(default_params))
        o_f.flush()

        return {
            'loss': np.inf,
            'status': STATUS_FAIL,
            'exception': str(e)
            }

# convert the dictionary to a search space:
# equivalent to:
#space = [ hp.uniform( 'learning_rate',  0.0001,  0.0005), <...etc> ]
space = []
for key, val in hopt_dict.iteritems():
    space.append(val[0](key, *val[1:]))

# Hyperopt params and setup
output_file = global_dir + 'hyperopt_log.csv'

global run_counter
run_counter = 0

o_f = open( output_file, 'wb' )
writer = csv.writer( o_f )
writer.writerow( list( hopt_dict.keys() ) )
o_f.flush()

if False:
    trials = Trials()
else:
    print "\n\n***\n\nLoading trials from disk...\n\n***\n\n"
    trials = pickle.load(open("/home/michael/projects/engaged_hackathon/"
        "notebooks/urban8k/results/hyperopt_run_0005/trials.pkl"))


class SaveTrials:
    """
    Setting this up as a context manager class so that if the program is
    interrupted (e.g. via keyboard interrupt), the trials object is saved to
    disk
    """
    def __enter__(self):
        print "Entering SaveTrails"
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        print "Overall time is ", time.time() - self.start_time

        # Save the trials object in case we want to start from here again...
        with open(global_dir + 'trials.pkl', 'w') as f:
            pickle.dump(trials, f, -1)


with SaveTrials():

    best = fmin(run_wrapper,
        space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)




# space = (
#     # hp.uniform( 'initial_learning_rate',  0.0001,  0.0005),
#     # hp.uniform( 'final_learning_rate_fraction', 0.08, 0.15),
#     # hp.quniform( 'epochs_of_initial', 25, 125, 5),
#     # hp.uniform( 'falloff', 0.01, 0.2),
#     # hp.uniform( 'input_dropout', 0.0, 0.1),
#     # hp.choice( 'filter_sizes', [3, 4, 5]),
#     # hp.quniform( 'num_filters', 32, 80, 1),
#     # hp.choice( 'num_filter_layers', [1, 2, 3]),
#     # hp.choice( 'pool_size_x', [2, 3, 4, 5, 6]),
#     # hp.choice( 'pool_size_y', [2, 3, 4, 5, 6]),
#     # hp.uniform( 'dense_dropout', 0.3, 0.7),
#     # hp.choice( 'num_dense_layers', [2, 3, 4]),
#     # hp.quniform( 'num_dense_units', 500, 1000, 1),
#     # hp.choice( 'augment_flip', [False, True]),
#     # hp.choice( 'augment_roll', [False, True]),
#     # hp.choice( 'augment_vol_ramp', [False, True]),
# )
