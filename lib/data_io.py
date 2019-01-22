import os
import pickle
import yaml
import numpy as np
from scipy.ndimage.interpolation import zoom

# golden data
base = yaml.load(open('../CONFIG.yaml'))['base_dir']
annotation_pkl_dir = base + 'extracted/annotations/'
spec_pkl_dir = base + 'extracted/specs/'
log_dir = base + 'ml_runs/'

# large data
# large_base = yaml.load(open('../CONFIG.yaml'))['large_data']
# large_spec_pkl_dir = large_base + 'specs/'
# large_annotation_pkl_dir = large_base + 'annots/'


def load_splits(test_fold, large_data=False):
    _base_path = large_base if large_data else base
    splits = yaml.load(open(_base_path + 'splits/folds.yaml'))

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
    spec = pickle.load(open(_specs_dir + SPEC_TYPE + '/' + fname, 'rb'), encoding='latin1')
    print(_annotations_dir + fname)
    annots, wav, sample_rate = pickle.load(open(_annotations_dir + fname, 'rb'), encoding='latin1')

    # reshape annotations
    for classname in annots:
        factor = float(spec.shape[1]) / annots[classname].shape[0]
        annots[classname] = zoom(annots[classname], factor)

    # create sampler
    if not LEARN_LOG:
        spec = np.log(A + B * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)

    return spec, annots


def load_data(fnames, SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B, is_golden=True):
    # load data and make list of specsamplers
    X = []
    y = []

    for fname in fnames:
        spec, annots = load_data_helper(fname, SPEC_TYPE, LEARN_LOG, A, B, is_golden)
        X.append(spec)
        y.append(annots[CLASSNAME])

    height = min(xx.shape[0] for xx in X)
    X = [xx[-height:, :] for xx in X]

    return X, y


def load_large_data(SPEC_TYPE, LEARN_LOG, CLASSNAME, A, B, max_to_load=np.iinfo(int).max):

    # loading the test data filenames - should avoid using these
    splits = yaml.load(open(base + 'splits/folds.yaml'))
    test_files = set([f for split in splits for f in split])
    test_postcodes = set([xx.split('-')[0].split('_')[0] for xx in test_files])

    # load data and make list of specsamplers
    X = []
    y = []

    num_dropped = num_used = 0
    sites_used = set()
    fnames = os.listdir(large_annotation_pkl_dir)
    for fname in fnames[:max_to_load]:

        if fname.split('-')[0].split('_')[0] in test_postcodes:
            num_dropped += 1
            continue

        num_used += 1
        sites_used.add(fname.split('-')[0].split('_')[0])

        spec, annots = load_data_helper(fname, SPEC_TYPE, LEARN_LOG, A, B, is_golden=False)
        X.append(spec)
        y.append(annots[CLASSNAME])

    print("Dropped %d files" % num_dropped)
    print("Used %d files" % num_used)
    print("Sites used: %d" % len(sites_used))

    height = min(xx.shape[0] for xx in X)
    X = [xx[-height:, :] for xx in X]

    return X, y
