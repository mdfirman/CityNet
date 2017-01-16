'''
I wrote the below to do caching of spectrograms, but I removed it for now...
'''


# Generate spectrograms for the provided files, or else load from cache

specs = {}  # spectrograms are stored in a dictionary, keys are the filenames

if load_from_cache or save_to_cache:

    # Create the cache dir if it doesn't exist already
    cache_dir = os.path.join(wav_dir, 'spectogram_files')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


# Loop over each file we want to predict for
for filename in filenames:

    # Create the path to where the spectrogram is/will be saved
    pickle_filename = filename.replace('.wav', '_' + options.SPEC_TYPE + '.pkl')
    cache_filename = os.path.join(cache_dir, pickle_filename)

    # If possible, let's load the spectogram from the cache file
    if load_from_cache and os.path.exists(cache_filename):
        specs[filename] = pickle.load(open(cache_filename))

    else:  # have not been able to load - must compute spectogram here

        # Read in the wav file and compute the spectrogram
        wav, sample_rate = librosa.load(os.path.join(wav_dir, filename), 22050)

        spec = melspectrogram(wav, sr=sample_rate, n_fft=options.N_FFT,
                                      hop_length=options.HOP_LENGTH, n_mels=options.N_MELS)

        # do log conversion if needed:
        if not options.LEARN_LOG:
            spec = np.log(options.A + options.B * spec)
            spec -= np.median(spec, axis=1, keepdims=True)

        specs[filename] = spec.astype(np.float32)

        # If required, we'll save the spectrogram to the cache file
        if save_to_cache:
            pickle.dump(specs[filename], open(cache_filename, 'w'), -1)



# temp - missing files
import pandas as pd

import data_io
richness = pd.read_csv(
    data_io.large_base + '../diversity_labelled_data/resultAndSiteFiles/6SitesWith90Minutes.csv')
richness_files = set(xx.split('.')[0] for xx in richness.File)

import os
div_wav_files = os.listdir('/media/michael/Engage/data/audio/alison_data/diversity_labelled_data/')
div_wav_files = [xx.split('.')[0] for xx in div_wav_files]
missing = set(richness_files) - set(div_wav_files)
missing = [xx + '.wav' for xx in missing]


filenames = missing
wav_dir = '/media/michael/Engage/data/audio/alison_data/large_dataset/wavs/'
