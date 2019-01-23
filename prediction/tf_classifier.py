import os
import sys
import yaml
import numpy as np
from time import time
import pickle
from collections import namedtuple

import tensorflow as tf
import librosa
from librosa.feature import melspectrogram
from scipy.io import wavfile


# Custom functions and classes
sys.path.append('../lib')
import train_helpers

N_FFT = 2048
HOP_LENGTH = 1024 # 512
N_MELS = 32 # 128


class TFClassifier(object):

    def __init__(self, opts, weights_path):
        """Create the layers of the neural network, with the same options we used in training"""
        self.opts = namedtuple("opts", opts.keys())(*opts.values())
        tf.reset_default_graph()

        net_options = {xx: opts[xx] for xx in train_helpers.net_params}
        self.net = train_helpers.create_net(SPEC_HEIGHT=N_MELS, **net_options)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        train_saver = tf.train.Saver()
        print("Loading from {}".format(weights_path))
        train_saver.restore(self.sess, weights_path)

        # Create an object which will iterate over the test spectrograms appropriately
        self.test_sampler = train_helpers.SpecSampler(
            256, self.opts.HWW_X, self.opts.HWW_Y, False, self.opts.LEARN_LOG, randomise=False, seed=10, balanced=False)

    def __exit__(self):
        # tear down the tensorflow session
        self.sess.close()

    def load_wav(self, wavpath, loadmethod='librosa'):
        tic = time()

        if loadmethod == 'librosa':
            # a more correct and robust way -
            # this resamples any audio file to 22050Hz
            self.wav, self.sample_rate = librosa.load(wavpath, 22050)
        elif loadmethod == 'wavfile':
            # a hack for speed - resampling is done assuming raw audio is
            # sampled at 44100Hz. Not recommended for general use.
            self.sample_rate, self.wav = wavfile.read(open(wavpath))
            self.wav = self.wav[::2] / 32768.0
            self.sample_rate /= 2
        else:
            raise Exception("Unknown load method")

    def compute_spec(self):
        tic = time()
        spec = melspectrogram(self.wav, sr=self.sample_rate, n_fft=N_FFT,
                              hop_length=HOP_LENGTH, n_mels=N_MELS)

        spec = np.log(self.opts.A + self.opts.B * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        self.spec = spec.astype(np.float32)

    def classify(self, wavpath=None):
        """Apply the classifier"""

        if wavpath is not None:
            self.load_wav(wavpath, loadmethod='librosa')
            self.compute_spec()

        labels = np.zeros(self.spec.shape[1])

        tic = time()
        probas = []
        for Xb, _ in self.test_sampler([self.spec], [labels]):
            pred = self.sess.run(
                self.net['output'],
                feed_dict={self.net['input']: Xb})
            probas.append(pred)
        print("Took %0.3fs to classify" % (time() - tic))

        return np.vstack(probas)[:, 1]
