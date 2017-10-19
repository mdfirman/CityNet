# A one-off script to create a labelling for each moment of the one minute dataset, saying if there is human noise, animal noise or both

import os
import sys
import cPickle as pickle
import pandas as pd
from data_helpers import load_annotations

base = yaml.load(open('../CONFIG.yaml'))['large_dataset']
where_to_save = base + '/annots/'
base_path = base + '/raw_annots/'

# saving all annotations as pkl files
wav_path = base + '/wavs/'

for fname in os.listdir(wav_path):
    try:
        savename = where_to_save + fname.replace('.wav', '.pkl')

        # load the annottion
        print fname
        annots, wav, sample_rate = load_annotations(
            fname, labels_dir=base_path, wav_dir=base_path.replace('raw_annots', 'wavs'))

        # save to disk
        with open(savename, 'w') as f:
            pickle.dump((annots, wav, sample_rate), f, -1)

    except Exception as e:
        print e
