# creating spectrograms from all the files, and saving split labelled versions to disk ready for machine learning
import os
import sys
import cPickle as pickle
import numpy as np
import librosa

base = yaml.load(open('../CONFIG.yaml'))['base_dir']
annotation_pkl_dir = base + '/extracted/annotations/'
savedir = base + '/extracted/specs/'


def gen_spec(x, sr):
    specNStepMod = 0.01    # horizontal resolution of spectogram 0.01
    specNWinMod = 0.03     # vertical resolution of spectogram 0.03

    ## Parameters
    nstep = int(sr * specNStepMod)
    nwin  = int(sr * specNWinMod)
    nfft = nwin

    # Get all windows of x with length n as a single array, using strides to avoid data duplication
    #shape = (nfft, len(range(nfft, len(x), nstep)))
    shape = (nfft, ((x.shape[0] - nfft - 1)/nstep)+1)
    strides = (x.itemsize, nstep*x.itemsize)
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Apply hamming window
    x_wins_ham = np.hamming(x_wins.shape[0])[..., np.newaxis] * x_wins

    # compute fft
    fft_mat = np.fft.fft(x_wins_ham, n=nfft, axis=0)[:(nfft/2), :]

    # log magnitude
    fft_mat_lm = np.log(np.abs(fft_mat))
    fft_mat = np.abs(fft_mat)

    return fft_mat


# create (multi-channel?) spectrogram
files = os.listdir(annotation_pkl_dir)

spec_type = '330'
this_save_dir = savedir + spec_type + '/'
for fname in files:
    if fname.endswith('.pkl'):

        with open(annotation_pkl_dir + fname) as f:
            annots, wav, sample_rate = pickle.load(f)

        spec = gen_spec(wav, sample_rate)

        with open(this_save_dir + fname, 'w') as f:
            pickle.dump(spec, f, -1)

        print fname


# create (multi-channel?) spectrogram
files = os.listdir(annotation_pkl_dir)

spec_type = 'mel'

this_save_dir = savedir + spec_type + '/'
for fname in files:
    if fname.endswith('.pkl'):

        with open(annotation_pkl_dir + fname) as f:
            annots, wav, sample_rate = pickle.load(f)

        spec = librosa.feature.melspectrogram(
            wav, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        spec = spec.astype(np.float32)
        print spec.shape
        sds

        with open(this_save_dir + fname, 'w') as f:
            pickle.dump(spec, f, -1)

        print fname


import librosa

N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 64

# create spectrogram
files = os.listdir(annotation_pkl_dir)

spec_type = 'mel64'

this_save_dir = savedir + spec_type + '/'
for fname in files:
    if fname.endswith('.pkl'):

        with open(annotation_pkl_dir + fname) as f:
            annots, wav, sample_rate = pickle.load(f)

        spec = librosa.feature.melspectrogram(
            wav, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        spec = spec.astype(np.float32)

        with open(this_save_dir + fname, 'w') as f:
            pickle.dump(spec, f, -1)

        print fname


for fname in files[20:]:
    if fname.endswith('.pkl'):

        with open(annotation_pkl_dir + fname) as f:
            annots, wav, sample_rate = pickle.load(f)

        spec = librosa.feature.melspectrogram(
            wav, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        spec = spec.astype(np.float32)
        sds
