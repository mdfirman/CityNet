import os
import cPickle as pickle
import yaml
import numpy as np
from scipy.ndimage.interpolation import zoom

# golden data
base = '/media/michael/Engage/data/audio/alison_data/golden_set/'
annotation_pkl_dir = base + 'extracted/annotations/'
spec_pkl_dir = base + 'extracted/specs/'
log_dir = base + 'ml_runs/'

# large data
large_base = '/media/michael/Engage/data/audio/alison_data/large_dataset/'
large_spec_pkl_dir = large_base + 'specs/'
large_annotation_pkl_dir = large_base + 'annots/'


def load_splits(test_fold):
    splits = yaml.load(open(base + 'splits/folds.yaml'))

    train_files = [xx for idx, split in enumerate(splits) for xx in split if idx != test_fold]
    test_files = splits[test_fold]

    return train_files, test_files


def load_data_helper(fname, SPEC_TYPE, LEARN_LOG, A, B, is_golden=True):

    if is_golden:
        _specs_dir = spec_pkl_dir
        _annotations_dir = annotation_pkl_dir
    else:
        _specs_dir = large_spec_pkl_dir
        _annotations_dir = large_annotation_pkl_dir

    # load spectrogram and annotations
    spec = pickle.load(open(_specs_dir + SPEC_TYPE + '/' + fname))
    annots, wav, sample_rate = pickle.load(open(_annotations_dir + fname))

    # reshape annotations
    for classname in annots:
        factor = float(spec.shape[1]) / annots[classname].shape[0]
        annots[classname] = zoom(annots[classname], factor)

    # create sampler
    if not LEARN_LOG:
        spec = np.log(A + B * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)

    return spec, annots


def load_data(fnames, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B):
    # load data and make list of specsamplers
    X = []
    y = []

    for fname in fnames:
        spec, annots = load_data_helper(fname, SPEC_TYPE, LEARN_LOG, A, B)
        X.append(spec)
        y.append(annots[CLASSNAME])

    height = min(xx.shape[0] for xx in X)
    X = [xx[-height:, :] for xx in X]

    return X, y


def load_large_data(SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B, max_to_load=np.iinfo(int).max):

    # loading the test data filenames - should avoid using these
    splits = yaml.load(open(base + 'splits/folds.yaml'))
    test_files = set([f for split in splits for f in split])

    # load data and make list of specsamplers
    X = []
    y = []

    num_dropped = 0
    fnames = os.listdir(large_annotation_pkl_dir)
    for fname in fnames[:max_to_load]:

        if fname in test_files:
            num_dropped += 1
            continue

        spec, annots = load_data_helper(fname, SPEC_TYPE, LEARN_LOG, A, B, is_golden=False)
        X.append(spec)
        y.append(annots[CLASSNAME])

    print "Dropped %d files" % num_dropped

    height = min(xx.shape[0] for xx in X)
    X = [xx[-height:, :] for xx in X]

    return X, y
