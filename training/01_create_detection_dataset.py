# A one-off script to create a labelling for each moment of the one minute dataset, saying if there is human noise, animal noise or both

import matplotlib.pyplot as plt

import os
import sys
import cPickle as pickle

from lib.data_helpers import load_annotations

base = yaml.load(open('../CONFIG.yaml'))['base_dir']
where_to_save = base + '/extracted/annotations/'
base_path = base + '/wavs/'


# load in the annotations
for fname in os.listdir(base_path):
    savename = where_to_save + fname

    # load the annottion
    annots, wav, sample_rate = load_annotations(fname)

    # save to disk
    with open(savename, 'w') as f:
        pickle.dump((annots, wav, sample_rate), f, -1)
