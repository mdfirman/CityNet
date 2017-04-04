import csv
import os
import numpy as np
import collections
import pandas as pd
import scipy.io.wavfile
import collections
import wavio

import sys, os
sys.path.append(os.path.expanduser('~/projects/engaged_hackathon/'))
from engaged.features import frequency


base_path = '/home/michael/projects/engaged_hackathon_data/urban_8k/'
meta_path = base_path + 'UrbanSound8K/metadata/UrbanSound8K.csv'
wav_path = base_path + 'UrbanSound8K/audio/'

spectrogram_parameters = {
    'nfft': 512,
    'window_width': 0.03,
    'overlap': 0.02
    }


####################################
# load in the meta data
data = pd.read_csv(meta_path)


####################################
# load in all the data
all_spec = []
all_fold_idxs = []

for idx in data.index:
    # load the audio and convert to spec
    folder = 'fold' + str(data['fold'][idx]) + '/'
    loadpath = wav_path + folder + data['slice_file_name'][idx]
    sample_rate, wav = scipy.io.wavfile.read(loadpath)

    # just take one channel of stereo files
    if len(wav.shape) == 2:  wav = wav[:, 0]

    spec, spec_sr = frequency.spectrogram(
        wav.ravel(), sample_rate, **spectrogram_parameters)
    spec -= np.median(spec, axis=1)[:, None]
#     spec[spec<0] = 0
    spec = spec.astype(np.float32)
    spec[np.isnan(spec)] = 0
    spec[np.isneginf(spec)] = -10
    spec[np.isposinf(spec)] = 10

    # convert to mel...
    spec = frequency.spec_to_mel(spec, spec_sr, 40)

    # crop height
    ss = spec.shape[0]
    new_height = (2.0/3.0) * float(ss)
    spec = spec[:new_height, :]

    # resize
#     zoom(spec, )

    # add to the list
    all_spec.append(spec.astype(np.float32))
    all_fold_idxs.append(data['fold'][idx])

    if idx % 100 == 0:
        print idx,


####################################
# pad spectrograms which are too short.
# Modifying list in-place, which is not pretty but it works...
for idx in range(len(all_spec)):

    spec = all_spec[idx]

    # padding the spectrogram to be the same length...
    desired_length = int(4.0 / spectrogram_parameters['overlap'] - 1)

    if spec.shape[1] < desired_length:
        # wrap the spectrogram
        num_tiles = np.ceil(float(desired_length) / spec.shape[1])
        tiled = np.tile(spec, (1, num_tiles))
        new_spec = tiled[:, :desired_length]
        all_spec[idx] = new_spec

    elif spec.shape[1] > desired_length:
        all_spec[idx] = spec[:, :desired_length]


####################################
