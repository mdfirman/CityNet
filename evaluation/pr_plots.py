import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import seaborn as sns
import itertools
import cPickle as pickle
import yaml

base_dir = yaml.load(open('../CONFIG.yaml'))['base_dir']
save_dir = '../plots/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# fontsize
fs = 28

def plot_pr(classname, runs):
    palette = itertools.cycle(sns.color_palette())

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=18)

    m = {'biotic': 'biotic', 'anthrop': 'anthropogenic'}
    mapper = {'warblr_challenge_baseline': 'bulbul',
         'oneSec_BI_baseline': 'BI',
              'oneSec_NDSI_baseline': 'NDSI (%s)' % m[classname],
         'oneSec_ACI_baseline': 'ACI',
              'oneSec_ADI_baseline': 'ADI',
             'ensemble_train': 'CityBioNet',
             'ensemble_train_anthrop': 'CityAnthroNet'}

    for run_type in runs:
        loaddir = base_dir + 'predictions/%s/%s/analysis/' % (run_type, classname)
        with open(loaddir + 'pr_results.pkl') as f:
            prec, recall, thresholds, prec_at_05, recall_at_05 = pickle.load(f)

        col = next(palette)
        if run_type in mapper:
            ax.plot(recall, prec, color=col, label=mapper[run_type])
        else:
            ax.plot(recall, prec, color=col, label=run_type)
        ax.plot(recall_at_05, prec_at_05, 'o', ms=12, color=col)


    box = ax.get_position()
    ax.set_position([box.x0 - 0.0, box.y0 + 0.01, box.width * 1.05, box.height * 0.95])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel('Precision', fontsize=fs)
    plt.xlabel('Recall', fontsize=fs)
    plt.tick_params(axis='both', which='major', labelsize=0.8*fs)
    plt.tick_params(axis='both', which='minor', labelsize=0.8*fs)

    if classname == 'anthrop':
        legend = ax.legend(loc='center', fontsize=fs, frameon=True)
        lab = 'B)'
    else:
        lab='A)'
        legend = ax.legend(loc='lower center', fontsize=0.7*fs, frameon=True)

    plt.text(-0.05, 1.1, lab, fontsize=fs*1.3)

    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set

    sns.set_style("whitegrid")

    plt.savefig(save_dir + '/pr_%s.png' % classname, dpi=200)
    plt.savefig(save_dir + '/pr_%s.pdf' % classname)


biotic_runs = [
    'ensemble_train',
    'warblr_challenge_baseline',
    'oneSec_ACI_baseline',
    'oneSec_ADI_baseline',
    'oneSec_BI_baseline',
    'oneSec_NDSI_baseline']

plot_pr('biotic', biotic_runs)


anthrop_runs = [
    'ensemble_train_anthrop',
    'oneSec_NDSI_baseline']

plot_pr('anthrop', anthrop_runs)
