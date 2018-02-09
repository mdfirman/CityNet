'''
Script to analyse a results folder and save all the plots, wav files etc
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import sys
import cPickle as pickle
import yaml
import numpy as np
import seaborn as sns
import sklearn
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


run_type = sys.argv[1]
classname = sys.argv[2]
SPEC_TYPE = 'mel'  # only for visualisation
VOLUME_BOOST = 5  # for saving wav files
print run_type, classname

base_dir = yaml.load(open('../CONFIG.yaml'))['base_dir']

results_dir = base_dir + 'predictions/%s/%s/per_file_predictions/' % (run_type, classname)
spec_pkl_dir = base_dir + 'extracted/specs/'
annotation_pkl_dir = base_dir + 'extracted/annotations/'

# where to save
savedir = force_make_dir(results_dir + '../analysis/')
per_file_plots_dir = force_make_dir(results_dir + '../per_file_plots/')
wav_results_dir = force_make_dir(results_dir + '../wav_results/')
print "Saving to:", savedir

##############################################################################
# ASSESS FP ETC
##############################################################################
print "\n\nResults across all files:"
scores = {}
total = dict.fromkeys(['tm', 'tp', 'tn', 'fp', 'fn'], 0)
all_y_true = []
all_y_pred = []
all_y_soft = []
sums_pred_hard = []
sums_pred_soft = []
sums_gt = []
fnames = []

for fname in os.listdir(results_dir):
    _y_true, y_pred_proba = pickle.load(open(results_dir + fname))
    y_pred_class = y_pred_proba[:, 1] > 0.5
    #
    # # only trust ndsi truth
    # if classname == 'biotic':
    #     tmp_results_dir = base_dir + 'predictions/%s/%s/per_file_predictions/' % ('warblr_challenge_baseline', classname)
    # _y_true, _ = pickle.load(open(tmp_results_dir + fname))

    assert _y_true.shape[0] == y_pred_class.shape[0]

    # for the one-second pooling, we need to apply a filter over the ground truth...
    y_true = _y_true.copy()
    if 'ensemble' not in run_type:
        hww = 10
        for idx, _ in enumerate(_y_true):
            start = max(idx - hww, 0)
            end = min(idx + hww, _y_true.shape[0])
            y_true[idx] = _y_true[start:end].max()

    y_true = y_true > 0.5

    total['tm'] += y_true.shape[0]
    total['tp'] += np.logical_and(y_true == y_pred_class, y_true == 1).sum()
    total['tn'] += np.logical_and(y_true == y_pred_class, y_true == 0).sum()
    total['fp'] += np.logical_and(y_true != y_pred_class, y_true == 0).sum()
    total['fn'] += np.logical_and(y_true != y_pred_class, y_true == 1).sum()

    all_y_true.append(y_true)
    all_y_pred.append(y_pred_class)
    all_y_soft.append(y_pred_proba[:, 1])

    slice_size = 60.0 / y_true.shape[0]
    sums_gt.append(y_true.sum() * slice_size)
    sums_pred_hard.append(y_pred_class.sum() * slice_size)
    sums_pred_soft.append(y_pred_proba[:, 1].sum() * slice_size)
    fnames.append(fname.replace('.pkl', ''))


scores['unbalanced_accuracy'] = \
    float(total['tp'] + total['tn']) / sum(total[key] for key in ['tp', 'tn', 'fp', 'fn'])

A = float(total['tp']) / sum(total[key] for key in ['tp', 'fn'])
B = float(total['tn']) / sum(total[key] for key in ['fp', 'tn'])
scores['balanced_accuracy'] = (A + B) / 2.0

all_y_true_stacked = np.hstack(all_y_true)
all_y_soft_stacked = np.hstack(all_y_soft)

if all_y_soft_stacked.max() > 1.0:  # rescale...
    all_y_soft_stacked -= all_y_soft_stacked.min()
    all_y_soft_stacked /= all_y_soft_stacked.max()

scores['f1'] = f1_score(all_y_true_stacked, all_y_soft_stacked > 0.5)


##############################################################################
# COMPUTING PR CURVE
##############################################################################
prec, recall, thresholds = precision_recall_curve(all_y_true_stacked, all_y_soft_stacked)

# finding the P/R at threshold=0.5
_true, _pred = all_y_true_stacked, all_y_soft_stacked > 0.5
totals = {}
totals['tp'] = np.logical_and(_true == _pred, _true == 1).sum()
totals['tn'] = np.logical_and(_true == _pred, _true == 0).sum()
totals['fp'] = np.logical_and(_true != _pred, _true == 0).sum()
totals['fn'] = np.logical_and(_true != _pred, _true == 1).sum()
scores['precision'] = totals['tp'] / float(totals['tp'] + totals['fp'])
scores['recall'] = totals['tp'] / float(totals['tp'] + totals['fn'])
scores['avg_precision'] = average_precision_score(all_y_true_stacked, all_y_soft_stacked)


with open(savedir + 'pr_results.pkl', 'w') as f:
    pickle.dump((prec, recall, thresholds, scores['precision'], scores['recall']), f, -1)

# Plotting PR curve
plt.plot(recall, prec)
plt.plot(scores['recall'], scores['precision'], 'ob', ms=6)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
plt.savefig(savedir + 'pr_curve.pdf')

##############################################################################
# COMPUTING EXTRA STATS FOR TABLE
##############################################################################

from sklearn.metrics import precision_score, recall_score

# Recall at 0.95 precision
min_idx = np.argmin(np.abs(np.hstack(prec) - 0.95))
scores['recall_at_095_prec'] = recall[min_idx]

min_idx = np.argmin(np.abs(np.hstack(prec) - 0.5))
scores['recall_at_050_prec'] = recall[min_idx]

for key, val in scores.iteritems():
    print "%s - %0.3f" % (key.ljust(20), val)

with open(savedir + 'scores.yaml', 'w') as f:
    yaml.dump(scores, f)

##############################################################################
# SAVING CSV SUMMARY
##############################################################################
import pandas as pd
res = pd.DataFrame(zip(fnames, sums_gt, sums_pred_hard, sums_pred_soft),
                   columns=['Filename', 'Ground truth', 'Prediction (hard)', 'Prediction (soft)'])
res.to_csv(savedir + 'per_file_summary_%s.csv' % classname, index=False)

##############################################################################
# PLOTTING CONFUSION MATRIX
##############################################################################
from sklearn.metrics import confusion_matrix

print "\nPlotting conf matrix:"
all_y_true = np.hstack(all_y_true)
print "Print mean", all_y_true.mean()
all_y_pred = all_y_soft_stacked > thresholds[min_idx]
cm = (confusion_matrix(all_y_true, all_y_pred) * slice_size).astype(float)[::-1, ::-1]#.T[::-1, ::-1].T
#cm = cm / cm.sum() * 100
labels = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
fmt = lambda x: "%s\n\n%2.1fs" % (labels[x], cm.ravel()[x])
annots = np.array([fmt(xx) for xx in range(4)]).reshape(2, 2)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes((0.18,0.15,0.8,0.8))
sns.heatmap(cm, annot=annots, fmt='s', ax=ax, cbar=1, vmin=0, vmax=cm.max(), annot_kws={'fontsize': 14}) #
#plt.savefig(savedir + 'confusion_matrix1.pdf')
ax.grid('off')
ax.set_aspect(1.0)
plt.xticks([0.5, 1.5], [classname.capitalize(), 'None'])
plt.yticks([0.5, 1.5], ['None', classname.capitalize()])
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel('Actual', fontsize=22)
plt.xlabel('Predicted', fontsize=22)
mapper = {'ensemble_train_anthrop': 'C)', 'ensemble_train': 'A)', 'warblr_challenge_baseline': 'B)'}
if run_type not in mapper:
    sys.exit()
plt.text(-0.35, 2.1, mapper[run_type], fontsize=22)
plt.savefig(savedir + 'confusion_matrix.pdf')
plt.savefig(savedir + 'confusion_matrix.png', dpi=800)
plt.close()
sys.exit()

##############################################################################
# PLOT PER-FILE SCATTER PLOT
##############################################################################
plt.figure(figsize=(5, 5))
plt.plot(sums_gt, sums_pred_hard, 'ob', markersize=3, label='Hard')
plt.plot(sums_gt, sums_pred_soft, '^r', markersize=4, label='Soft')
plt.plot([0, 60], [0, 60],':')
plt.gca().set_aspect('equal')
plt.xlabel('Ground truth (s)', fontsize=15)
plt.ylabel('Predicted (s)', fontsize=15)
leg = plt.legend(loc='upper left', frameon=True)
leg.get_frame().set_edgecolor([0.7, 0.7, 0.7])
leg.get_frame().set_linewidth(1.0)
plt.title('Per-file predictions for %s sound' % classname)
plt.savefig(savedir + 'overall_success.pdf')
plt.savefig(savedir + 'overall_success.png', dpi=800)
plt.close()

plt.figure(figsize=(5, 5))
plt.plot(sums_gt, sums_pred_hard, 'ob', markersize=3)
plt.plot([0, 60], [0, 60],':')
plt.gca().set_aspect('equal')
plt.xlabel('Ground truth biotic sound (s)', fontsize=15)
plt.ylabel('Predicted biotic sound (s)', fontsize=15)
plt.title('Per-file predictions for %s sound' % classname)
plt.savefig(savedir + 'overall_success_hard_only.pdf')
plt.savefig(savedir + 'overall_success_hard_only.png', dpi=1500)
plt.close()

from sklearn.metrics import *
print "\nResults Hard:"
for cc in [mean_absolute_error, mean_squared_error, median_absolute_error, r2_score]:
    print cc(sums_gt, sums_pred_hard)

print "\nResults Soft"
for cc in [mean_absolute_error, mean_squared_error, median_absolute_error, r2_score]:
    print cc(sums_gt, sums_pred_soft)

##############################################################################
# PLOTTING RESULTS SPECTROGRAMS
##############################################################################
print "\nPlotting results spectrograms"

for fname in os.listdir(results_dir):

    y_true, y_pred_proba = pickle.load(open(results_dir + fname))
    spec = pickle.load(open(spec_pkl_dir + SPEC_TYPE + '/' + fname))

    scale = 30
    spec_bin_width_in_s = 60 / float(spec.shape[1])

    fig = plt.figure(figsize=(50, 5))
    ax = fig.add_axes((0.05,0.16,0.95,0.9))

    ax.imshow(np.log(0.01+ 10 * spec), cmap='gray_r', aspect='auto')

    ax.plot(y_pred_proba[:, 1] * scale, 'g', label='Predicted')
    ax.plot(y_true * scale, 'r', label='True')
    ax.plot(y_true * 0 + scale/2, '--', color=[0.5, 0.5, 0.5], label='True')

    # sort out axes
    ax.invert_yaxis()
    plt.xlim(0, spec.shape[1] * 1.01)
    plt.ylim(0, spec.shape[0] * 1.1)
    plt.xticks(np.arange(0, 65, 5) / spec_bin_width_in_s, np.arange(0, 65, 5))
    plt.yticks([0, scale/2, scale], [0, 0.5, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xlabel('Time (s)', fontsize=25)
    plt.ylabel('Label', fontsize=25)

    ax.grid('off')
    savepath = per_file_plots_dir + fname.replace('.pkl', '.pdf')
    plt.savefig(savepath, pad_inches=0.0, frame_on=0)
    plt.close()

    print ".",

##############################################################################
# SAVING WAV FILES
##############################################################################
print "\nSaving wav files..."
from scipy.io import wavfile
from scipy.ndimage.interpolation import zoom
import librosa

samples = {xx: [] for xx in ['true_positive', 'true_negative', 'false_positive', 'false_negative']}

for fname in os.listdir(results_dir):

    # load wav and predictions
    # sr, wav = wavfile.read(base_dir + 'wavs/' + fname.replace('.pkl', '.wav'))
    wav, sr = librosa.load(base_dir + 'wavs/' + fname.replace('.pkl', '.wav'))
    y_true, y_pred_proba = pickle.load(open(results_dir + fname))
    y_pred_class = y_pred_proba[:, 1] > 0.5
    factor = wav.shape[0] / float(y_true.shape[0])

    # get locations of tp etc
    locs = {}
    locs['true_positive'] = np.logical_and(y_true == y_pred_class, y_true == 1)
    locs['true_negative'] = np.logical_and(y_true == y_pred_class, y_true == 0)
    locs['false_positive'] = np.logical_and(y_true != y_pred_class, y_true == 0)
    locs['false_negative'] = np.logical_and(y_true != y_pred_class, y_true == 1)

    # append correct bits
    for key, loc in locs.iteritems():
        to_extract = zoom(loc, factor)
        samples[key].append(wav[to_extract > 0.5])

# reform wav files
for key in samples:
    samples[key] = np.hstack(samples[key])
    savepath = wav_results_dir + key + '.wav'
    wavfile.write(savepath, sr, VOLUME_BOOST * samples[key])
