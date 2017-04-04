import csv
import collections
import numpy as np
import scipy.io
from sklearn.cross_validation import train_test_split

def savemat_large(savepath, dic, modify_in_place=True, **kwargs):
    """
    Save a python dictionaries containing arbitrarily large numpy arrays.
    Essentially a wrapper for scipy.io.savemat, but numpy arrays which are
    too large are split up before saving.

    This is a hacky fix for the problem that individual objects bigger than
    2**32 bytes cannot be saved, at least in Python 2.7.

    See
    https://github.com/pv/scipy-work/blob/master/scipy/io/matlab/mio5.py

    """

    max_array_size_in_bytes = 2**(32-1)

    # maintain a list of all the keys in the dict which have been split
    dic['split_arrays'] = []

    if modify_in_place:
        # change the dictionary in place, i.e. don't create a copy.
        # best for memory but changes the input data
        for key, val in dic.iteritems():

            # if we have a numpy array we need to split...
            if isinstance(val, np.ndarray) and val.nbytes > max_array_size_in_bytes:

                num_sections = \
                    np.ceil(float(val.nbytes) / float(max_array_size_in_bytes))

                # split along the longest axis
                # (this is memory intensive, not sure how to fix this problem
                # in a simple way...)
                longest_axis = np.argmax(np.array(val.shape))
                dic[key] = np.array_split(val, num_sections, axis=longest_axis)

                # remember that we split this array and which axis we split along
                dic['split_arrays'].append((key, longest_axis))

    else:
        raise Exception("Not implemented")

    assert 'split_arrays' in dic
    scipy.io.savemat(savepath, dic, **kwargs)


def loadmat_large(loadpath):
    """
    Load a dictionary which has been saved using savemat_large
    """

    dic = scipy.io.loadmat(loadpath)
    print dic.keys()

    # reforming split arrays
    for key, axis in dic['split_arrays']:
        dic[key] = np.concatenate(dic[key], int(axis))

    return dic




def try_number(s):
    """Converts s to float if possible, else leaves as is"""
    try:
        return float(s)
    except ValueError:
        return s


def load_annotations():
    """
    Loads all the annotations for the one miinute dataset
    returns them individiaully and grouped by filename
    """
    # load in the annotations
    base_path = '/home/michael/projects/engaged_hackathon_data/raw_data/one_minute_files/'
    dataset_csv = csv.reader(open(base_path + 'urban_sounds_labels.csv'))

    annotations = []  # list of all class info

    # I'm basically reinventing pandas here - very silly
    for count, line in enumerate(dataset_csv):
        if count == 0:
            header = line
            continue

        annotation = {label:try_number(item) for label, item in zip(header, line)}
        annotation['length'] = \
            annotation['LabelEndTime_Seconds'] - annotation['LabelStartTime_Seconds']
        annotation['Label'] = annotation['Label'].strip().lower()
        annotations.append(annotation)

    # group annotations by filename
    file_annotations = collections.defaultdict(list)
    for annotation in annotations:
        file_annotations[annotation['Filename']].append(annotation)

    return annotations, file_annotations


def extract_1d_patches(array, locations, hww):
    """
    Extract vertical patches from the array, at the locations given.
    Each slice has a half window width hww

    Returns an array of shape:
    (len(locations), array.shape[0], hww*2+1)
    """
    # pad the array to account for overspill
    offset_idxs_np = np.array(locations).ravel() + hww
    extra1 = np.tile(array[:, 0], (hww, 1)).T
    extra2 = np.tile(array[:, -1], (hww, 1)).T
    a_temp = np.hstack((extra1, array, extra2))

    # set up the array of index locations to extract from
    idxs = [offset_idxs_np]
    for offset in range(1, hww+1):
        idxs.insert(0, offset_idxs_np-offset)
        idxs.append(offset_idxs_np+offset)
    new_idx = np.vstack(idxs).T.ravel()

    # extract the patches and do the appropriate reshapgin

    new_shape = (array.shape[0], offset_idxs_np.shape[0], hww*2 + 1)
    to_return = a_temp[:, new_idx].reshape(new_shape).transpose((1, 0, 2))
    return to_return


def subsample_pair(X, y, num):
    if num > X.shape[0]:
        return X, y
    else:
        idxs = np.random.choice(X.shape[0], num, replace=False)
        return X[idxs, :], y[idxs]


def load_multilabel_dataset(train_path, test_path, max_samples=None):

    train_data = loadmat_large(train_path)
    test_data = loadmat_large(test_path)

    X_train_val = train_data['slices']
    y_train_val = train_data['labels']
    X_test = test_data['slices']
    y_test = test_data['labels']

    train_idxs, val_idxs = train_test_split(
        np.arange(X_train_val.shape[0]), test_size=0.3)

    X_train = X_train_val[train_idxs]
    y_train = y_train_val[train_idxs]
    X_val = X_train_val[val_idxs]
    y_val = y_train_val[val_idxs]

    for xx, yy in zip(test_data['class_names'], train_data['class_names']):
        assert xx == yy

    if max_samples is not None:
        X_train, y_train = subsample_pair(X_train, y_train, max_samples)
        X_test, y_test = subsample_pair(X_test, y_test, max_samples)
        X_val, y_val = subsample_pair(X_val, y_val, max_samples)

    return X_train, y_train, X_val, y_val, X_test, y_test, test_data['class_names']

