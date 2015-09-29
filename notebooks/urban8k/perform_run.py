# General
import numpy as np
import time
import sys
import os

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

# IO
import scipy.io
import yaml
import cPickle as pickle

# my helpers
import urban8k_helpers as helpers
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import cnn_utils


def run(global_dir, network_input_size, num_classes, data, num_epochs,
        do_median_normalise, minibatch_size, augment_data=True, params=None, run_directory_name=None):
    """
    Perform a single training run, given a list of parameters as might
    be provided by e.g. hyperopt
    """

    if params is not None:
        temp_params = [0.00030558346816, 0.13865800304103, 35, 0.05863104967699, 0.03909097055997,
            3, 60, 2, 3, 4, 0.6880736253895, 3, 800, None, True, False, True,
            None, None]
        temp_params[-2] = params[1] # norm_mean_std
        temp_params[-1] = params[2] # norm_std_std
        temp_params[-6] = params[0] # initial filter layer
        params = temp_params

        learning_rate = params[0]
        final_learning_rate_fraction = params[1]
        epochs_of_initial = params[2]
        falloff = params[3]
        augment_options = {
            'roll': params[-3],
            'rotate': False,
            'flip': params[-5],
            'volume_ramp': params[-4],
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

    # forming validation data
    slice_width = network_input_size[1]
    val_X, val_y = helpers.form_slices_validation_set(
                data, slice_width, do_median_normalise, 'val_')

    # setting up the network
    network, train_fn, predict_fn, val_fn, input_var, target_var, loss = \
        helpers.prepare_network(learning_rate, network_input_size, num_classes, params)

    best_validation_loss = 10000
    best_model = None
    best_epoch = 0
    run_results = []

    if run_directory_name is None:
        this_run_dir = helpers.create_numbered_folder(global_dir + 'run_%06d/')
    else:
        this_run_dir = global_dir + '/' + run_directory_name + '/'
        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)

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

        results = {'epoch': epoch}

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

        try:
            results['auc'] = cnn_utils.multiclass_auc(
                np.hstack(y_gts), probability_predictions)
        except:
            print "Failed to compute the AUC"
            results['auc'] = np.nan

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
