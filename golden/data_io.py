import os
import cPickle as pickle
import yaml
import numpy as np
from scipy.ndimage.interpolation import zoom

base = '/media/michael/Seagate/engage/alison_data/golden_set/'
annotation_pkl_dir = base + 'extracted/annotations/'
spec_pkl_dir = base + 'extracted/specs/'
log_dir = base + 'ml_runs/'


def load_splits(test_fold):
    splits = yaml.load(open(base + 'splits/folds.yaml'))

    if test_fold == 0:
        train_files = splits[1] + splits[2]
        test_files = splits[0]
    elif test_fold == 1:
        train_files = splits[0] + splits[2]
        test_files = splits[1]
    elif test_fold == 2:
        train_files = splits[0] + splits[1]
        test_files = splits[2]

    return train_files, test_files


def load_data(train_files, test_files, SPEC_TYPE, SPEC_HEIGHT, LEARN_LOG, CLASSNAME):
    # load data and make list of specsamplers
    train_data = [[], []]
    test_data = [[], []]

    for fname in os.listdir(spec_pkl_dir + SPEC_TYPE + '/'):
        # load spectrogram and annotations
        # spec = pickle.load(open(spec_pkl_dir + fname))[:SPEC_HEIGHT, :]
        spec = pickle.load(open(spec_pkl_dir + SPEC_TYPE + '/' + fname))[-SPEC_HEIGHT:, :]
        annots, wav, sample_rate = pickle.load(open(annotation_pkl_dir + fname))

        # reshape annotations
        for classname in annots:
            factor = float(spec.shape[1]) / annots[classname].shape[0]
            annots[classname] = zoom(annots[classname], factor)

        # create sampler
        if not LEARN_LOG:
            spec = np.log(0.001 + spec)
            spec = spec - np.median(spec, axis=1, keepdims=True)

        if fname in train_files:
            train_data[0].append(spec)
            train_data[1].append(annots[CLASSNAME])
        elif fname in test_files:
            test_data[0].append(spec)
            test_data[1].append(annots[CLASSNAME])
        else:
            print "WARNING - file %s not in train or test" % fname

    return train_data, test_data
