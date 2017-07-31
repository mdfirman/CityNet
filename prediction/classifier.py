import os
import sys
import yaml
import numpy as np
from time import time
import cPickle as pickle
from easydict import EasyDict as edict

import librosa
from librosa.feature import melspectrogram
from scipy.io import wavfile

# Neural network imports
import lasagne
import theano

# Custom functions and classes
sys.path.append('..')
from lib import train_helpers

N_FFT = 2048
HOP_LENGTH = 1024 # 512
N_MELS = 32 # 128

# specify where the pretrained model is that we want to load
models_dir = '/media/michael/Engage/data/audio/alison_data/golden_set/predictions/ensemble_train_anthrop/0/anthrop/'

# Specify the names of the files we want to load in
# (Keep as they are to load in the model in the dropbox folder)
weights_path = os.path.join(models_dir, 'results/weights_99.pkl')
options_path = os.path.join(models_dir, 'network_opts.yaml')

# Loading the options for network architecture, spectrogram type etc
default_opts = edict(yaml.load(open(options_path)))


class Classifier(object):

    def __init__(self, opts=default_opts, weights_path=weights_path):
        # Create the layers of the neural network, with the same options we used in training
        self.opts = opts

        net_options = {xx: opts[xx] for xx in train_helpers.net_params}
        print "TODO - Remove the height thing..."
        net = train_helpers.create_net(SPEC_HEIGHT=32, **net_options)

        # Create theano functions
        test_output = lasagne.layers.get_output(net['prob'], deterministic=True)
        x_in = net['input'].input_var
        self.pred_fn = theano.function([x_in], test_output)

        # Load params
        weights = pickle.load(open(weights_path))
        lasagne.layers.set_all_param_values(net['prob'], weights)

        # Create an object which will iterate over the test spectrograms appropriately
        self.test_sampler = train_helpers.SpecSampler(
            256, opts.HWW_X, opts.HWW_Y, False, opts.LEARN_LOG, randomise=0, seed=10)

    def load_wav(self, wavpath, loadmethod='librosa'):
        tic = time()

        if loadmethod == 'librosa':
            self.wav, self.sample_rate = librosa.load(wavpath, 22050)
        elif loadmethod == 'wavfile':
            self.sample_rate, self.wav = wavfile.read(open(wavpath))
            self.wav = self.wav[::2] / 32768.0
            self.sample_rate /= 2
        else:
            raise Exception()

        # print "Took %fs to load wav" % (time() - tic)
        #self.fullwav = self.wav.copy()
        # self.wav = self.wav[:10000]

    def compute_spec(self):
        tic = time()
        spec = melspectrogram(self.wav, sr=self.sample_rate, n_fft=N_FFT,
                              hop_length=HOP_LENGTH, n_mels=N_MELS)
        # Do log conversion:
        # spec = np.log(self.opts.A + self.opts.B * spec)
        # spec -= np.median(spec, axis=1, keepdims=True)
        spec = spec.astype(np.float32)
        self.spec = spec

        # print "Took %fs to create spec" % (time() - tic)

    def classify(self):
        """Apply the classifier"""
        sshape = self.spec.shape[1]
        s_sr = 21.5332749639  # sshape / float(len_in_s)
        labels = np.hstack((np.diff(np.round(np.arange(sshape) / s_sr)), [0]))

        tic = time()
        probas = []
        for Xb, _ in self.test_sampler([self.spec], [labels]):
            probas.append(self.pred_fn(Xb))
        print "Took %fs to classify" % (time() - tic)

        return np.vstack(probas)

    def load_and_classify(self, wavpath):
        '''Load in a wav file, and apply classifier'''
        self.loadwav(wavpath)
        self.compute_spec()
        return self.classify()
