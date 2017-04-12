import os
import sys
import yaml
import numpy as np
import cPickle as pickle

import librosa
from librosa.feature import melspectrogram

# Neural network imports
import nolearn
import lasagne

# Custom functions and classes
sys.path.append('..')
from lib import train_helpers


# specify where the pretrained model is that we want to load
models_dir = '/home/michael/Dropbox/engage/FairbrassFirmanetal_/data/models/biotic_trained_large/'

# Specify the names of the files we want to load in
# (Keep as they are to load in the model in the dropbox folder)
weights_path = os.path.join(models_dir, 'weights_99.pkl')
options_path = os.path.join(models_dir, 'network_params.yaml')

# Loading the options for network architecture, spectrogram type etc
options = yaml.load(open(options_path))


class Classifier(object):

    def __init__(self):
        # Create the layers of the neural network, with the same options we used in training
        net_options = {xx: options[xx] for xx in train_helpers.net_params}
        network = train_helpers.create_net(**net_options)

        # Create an object which will iterate over the test spectrograms appropriately
        test_sampler = train_helpers.SpecSampler(
            4, options['HWW'], False, options['LEARN_LOG'], randomise=0, seed=10)

        # Create a nolearn object to contain the network and push data through
        self.net = nolearn.lasagne.NeuralNet(
            layers=network['prob'], update=lasagne.updates.adam, batch_iterator_test=test_sampler)

        # Initialise the network and load in the pretrained parameters
        self.net.initialize()
        self.net.load_params_from(weights_path)

    def load_and_classify(self, wavpath):
        '''Load in a wav file, and apply classifier'''
        wav, sample_rate = librosa.load(wavpath, 22050)

        # Compute the spectrogram
        spec = melspectrogram(wav, sr=sample_rate, n_fft=options['N_FFT'],
                              hop_length=options['HOP_LENGTH'], n_mels=options['N_MELS'])

        # Do log conversion:
        spec = np.log(options['A'] + options['B'] * spec)
        spec -= np.median(spec, axis=1, keepdims=True)
        spec = spec.astype(np.float32)

        # Apply the classifier
        pred = self.net.predict_proba([spec])
        return pred
