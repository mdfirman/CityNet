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

# CNN bits
import theano
import urban8k_helpers as helpers
import lasagne

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
num_epochs = 2
small_dataset = True

# NOTE that the *actual* minibatch size will be something like num_classes*minibatch_size
minibatch_size = 100 # optimise
augment_data = True

do_median_normalise = True

augment_options = {
    'roll': True,
    'rotate': False,
    'flip': False,
    'volume_ramp': True,
    'normalise': False
    }

# what size will the CNN get ultimately? - optimise this!
network_input_size = (128, slice_width)

learning_rate = 0.00025

# loading the data
loadpath = base_path + 'splits_128/split' + str(split) + '.pkl'
data, num_classes = helpers.load_data(
    loadpath, do_median_normalise=do_median_normalise, small_dataset=small_dataset)

###############################################################
# Overall setup
global_dir = helpers.create_numbered_folder('results/hyperopt_run_%04d/')

# form a proper validation set here...
val_X, val_y = helpers.form_slices_validation_set(
    data, slice_width, do_median_normalise)

# setting up the network
network, train_fn, predict_fn, val_fn, input_var, target_var, loss = \
    helpers.prepare_network(learning_rate, network_input_size, num_classes)

###############################################################
# Setup for *this* run

print "Starting training..."
print "There will be %d minibatches per epoch" % \
    (data['train_y'].shape[0] / (minibatch_size*num_classes))

best_validation_accuracy = 0.0
best_model = None
best_epoch = 0
run_results = []

this_run_dir = helpers.create_numbered_folder(global_dir + 'run_%06d/')

lgr = helpers.Logger(this_run_dir + 'results_log.txt')
lr = helpers.LearningRate(learning_rate, learning_rate / 10.0, 50, 0.02)

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
    if results['mean_val_accuracy'] > best_validation_accuracy:
        best_model = (network, predict_fn, results)
        best_validation_accuracy = results['mean_val_accuracy']
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

# Save training history to a mat file
results_savepath = this_run_dir + '/training_history.mat'
scipy.io.savemat(results_savepath, dict(run_results=run_results))

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
