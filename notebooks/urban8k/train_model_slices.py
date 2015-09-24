# General imports
import numpy as np
import os
import sys
import time

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
num_epochs = 256
small_dataset = False
max_evals = 10000  # hyperopt


# should make minibatch size a multiple of 10 really
minibatch_size = 80 # optimise
augment_data = True

do_median_normalise = True

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

# loading the data
loadpath = base_path + 'splits_128/split' + str(split) + '.pkl'
data, num_classes = helpers.load_data(
    loadpath, do_median_normalise=do_median_normalise, small_dataset=small_dataset)


# form a proper validation set here...
val_X, val_y = helpers.form_slices_validation_set(
    data, slice_width, do_median_normalise)

###############################################################
# Overall setup
global_dir = helpers.create_numbered_folder('results/hyperopt_run_%04d/')


def perform_run(params=None):
    """
    Perform a single training run, given a list of parameters as might
    be provided by e.g. hyperopt
    """

    if params is not None:
        learning_rate = params[0]
        final_learning_rate_fraction = params[1]
        epochs_of_initial = params[2]
        falloff = params[3]
        augment_options = {
            'roll': params[-2],
            'rotate': False,
            'flip': params[-3],
            'volume_ramp': params[-1],
            'normalise': False
            }
        print "Learning rate is ", learning_rate
    else:
        learning_rate = 0.00025
        final_learning_rate_fraction = 0.1
        epochs_of_initial = 50
        falloff = 0.2
        augment_options = {
            'roll': True,
            'rotate': False,
            'flip': False,
            'volume_ramp': True,
            'normalise': False
            }

    # setting up the network
    network, train_fn, predict_fn, val_fn, input_var, target_var, loss = \
        helpers.prepare_network(learning_rate, network_input_size, num_classes, params)

    best_validation_loss = 10000
    best_model = None
    best_epoch = 0
    run_results = []

    this_run_dir = helpers.create_numbered_folder(global_dir + 'run_%06d/')

    lgr = helpers.Logger(this_run_dir + 'results_log.txt')
    lr = helpers.LearningRate(learning_rate,
        learning_rate * final_learning_rate_fraction, epochs_of_initial, falloff)

    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_loss = train_batches = 0
        start_time = time.time()

        # we now create the generator for this epoch
        epoch_gen = helpers.generate_balanced_minibatches_multiclass(
                data['train_X'], data['train_y'], int(minibatch_size),
                slice_width, augment_data=augment_data, augment_options=augment_options,
                shuffle=True)
        threaded_epoch_gen = helpers.threaded_gen(epoch_gen)

        global run_counter
        results = {'epoch': epoch, 'run_counter': run_counter}

        ########################################################################
        # TRAINING PHASE

        for count, batch in enumerate(threaded_epoch_gen):
            inputs, targets = batch
            train_loss += train_fn(inputs, targets, np.float32(lr.get_lr(epoch)))
            train_batches += 1

            # print a progress bar
            if count % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        # computing the training loss
        results['train_loss'] = train_loss / train_batches
        print " total = ", train_batches,

        results['train_time'] = time.time() - start_time

        ########################################################################
        # VALIDATION PHASE

        start_time = time.time()
        batch_val_loss = val_acc = val_batches = 0
        y_preds = []
        y_gts = []

        # explicitly keep track of number of examples in case the minibatch
        # iterator does something funny, e.g. class balancing
        num_validation_items = 0

        for batch in helpers.iterate_minibatches(val_X, val_y, 8, slice_width):

            inputs, targets = batch

            err, acc = val_fn(inputs, targets)
            batch_val_loss += err
            val_acc += acc
            val_batches += 1

            num_validation_items += targets.shape[0]

            y_preds.append(predict_fn(inputs))
            y_gts.append(targets)

        probability_predictions = np.vstack(y_preds)

        results['auc'] = cnn_utils.multiclass_auc(
            np.hstack(y_gts), probability_predictions)
        results['mean_val_accuracy'] = np.sum(val_acc) / num_validation_items
        results['val_loss'] = batch_val_loss / val_batches
        results['val_time'] = time.time() - start_time

        class_predictions = np.argmax(probability_predictions, axis=1)
        results['cm'] = metrics.confusion_matrix(np.hstack(y_gts), class_predictions)

        results['learning_rate'] = lr.get_lr(epoch)

        ########################################################################
        # LOGGING AND PLOTTING AND SAVING

        start_time = time.time()

        # Rember results for this epoch and log
        run_results.append(results)
        lgr.log(results)

        # saving the model, if we are the best so far (todo - thread this!)
        if results['val_loss'] < best_validation_loss:
            best_model = (network, predict_fn, results)
            best_validation_loss = results['val_loss']
            best_epoch = epoch

            with open(this_run_dir + 'best_model.pkl', 'w') as f:
                pickle.dump(best_model, f, -1)

        # updating a training/val loss graph...
        # (only bother with this every so many epochs)
        if (epoch % 10) == 0:
            all_train_losses = [result['train_loss'] for result in run_results]
            all_val_losses = [result['val_loss'] for result in run_results]

            graph_savepath = this_run_dir + 'training_graph.png'
            plt.clf()
            plt.plot(all_train_losses, label='train_loss')
            plt.plot(all_val_losses, label='val_loss')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(graph_savepath, bbox_inches='tight')

        ########################################################################
        # EARLY STOPPING (MY EARLY STOPPING)
        # I'm doing this more for speed than overfitting.
        # I think I'll say that if the best result wasn't in the most recent
        # 50% of epochs, then quit.
        if epoch > 50:
            all_val_losses = [result['val_loss'] for result in run_results]
            the_best_epoch = np.argmin(np.array(all_val_losses))
            if the_best_epoch < epoch / 2:
                print "EARLY EXIT"
                break

    # Save training history to a mat file
    results_savepath = this_run_dir + '/training_history.mat'
    scipy.io.savemat(results_savepath, dict(run_results=run_results, params=list(params)))

    # saving the final model
    final_model = (network, predict_fn, results)
    with open(this_run_dir + 'final_model.pkl', 'w') as f:
        pickle.dump(final_model, f, -1)

    # for posterity, let's save separately the best and final validation loss and accuracy
    final_summary = {
        'final_val_acc': float(run_results[-1]['mean_val_accuracy']),
        'final_val_loss': float(run_results[-1]['val_loss']),
        'best_val_acc': float(run_results[best_epoch]['mean_val_accuracy']),
        'best_val_loss': float(run_results[best_epoch]['val_loss'])
        }

    with open(this_run_dir + 'best_and_final.yaml', 'w') as fid:
        fid.write( yaml.dump(final_summary) )

    return final_summary


def run_wrapper(params):
    """
    This is the wrappper which is minimised
    """
    global run_counter
    global o_f

    run_counter += 1
    print "\n****** RUN %d ******" % run_counter
    sys.stdout.flush()

    start_time = time.time()

    try:
        final_summary = perform_run(params)

        print "\n\n",
        for key, val in final_summary.iteritems():
            print "%s : %0.4f" % (key, val),
        print "took %0.3fs" % (time.time() - start_time)

        writer.writerow([
            final_summary['best_val_loss'],
            final_summary['best_val_acc'],
            final_summary['final_val_acc']
            ] + list( params ))
        o_f.flush()

        return {
            'loss': final_summary['best_val_loss'],
            'status': STATUS_OK
            }

    except Exception as e:

        print "Failed run: ", str(e)
        print "Took %0.3fs" % (time.time() - start_time)

        writer.writerow(['FAILURE', 'FAILURE', 'FAILURE'] + list(params))
        o_f.flush()

        return {
            'loss': np.inf,
            'status': STATUS_FAIL,
            'exception': str(e)
            }

# set up the hyperopt search space
space = (
    hp.uniform( 'initial_learning_rate',  0.0001,  0.0005),
    hp.uniform( 'final_learning_rate_fraction', 0.01, 0.1),
    hp.quniform( 'epochs_of_initial', 25, 125, 5),
    hp.uniform( 'falloff', 0.01, 0.2),
    hp.uniform( 'input_dropout', 0.0, 0.5),
    hp.choice( 'filter_sizes', [3, 4, 5]),
    hp.quniform( 'num_filters', 32, 80, 1),
    hp.choice( 'num_filter_layers', [1, 2, 3]),
    hp.choice( 'pool_size_x', [2, 3, 4, 5, 6]),
    hp.choice( 'pool_size_y', [2, 3, 4, 5, 6]),
    hp.uniform( 'dense_dropout', 0.3, 0.7),
    hp.choice( 'num_dense_layers', [2, 3, 4]),
    hp.quniform( 'num_dense_units', 500, 1000, 1),
    hp.choice( 'inital_filter_layer', [False, True]),
    hp.choice( 'augment_flip', [False, True]),
    hp.choice( 'augment_roll', [False, True]),
    hp.choice( 'augment_vol_ramp', [False, True])
)

# Hyperopt params and setup
output_file = global_dir + 'hyperopt_log.csv'

global run_counter
run_counter = 0

headers = [ 'best_val_loss', 'best_val_acc', 'final_val_acc', 'initial_learning_rate',
            'final_learning_rate_fraction',
            'epochs_of_initial', 'falloff', 'input_dropout', 'filter_sizes',
            'num_filters', 'num_filter_layers', 'pool_size_x', 'pool_size_y',
            'dense_dropout', 'num_dense_layers', 'num_dense_units',
            'inital_filter_layer', 'augment_flip', 'augment_roll', 'augment_vol_ramp']

o_f = open( output_file, 'wb' )
writer = csv.writer( o_f )
writer.writerow( headers )
o_f.flush()

trials = Trials()


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
