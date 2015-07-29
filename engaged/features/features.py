import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import skimage.transform
from skimage.util.shape import view_as_blocks

# extract a snippet of a fixed length
# length = 4  # second
# num_spectogram_bins = length / meta['gen_spectrogram']['overlap']

def force_spectrogram_length(spec, desired_length):
    """
    pads or crops the spectrogram as needed to make it the desired length

    desired_length is in bins (not ms)
    """
    if spec.shape[1] < desired_length:
        to_add = desired_length - spec.shape[1]
        spec = np.hstack((spec, np.zeros((512, to_add))))
    elif spec.shape[1] > desired_length:
        spec = spec[:, :desired_length]

    return spec


def small_spectrogram(spec):
    spec = force_spectrogram_length(spec, 396)

    im_norm = (spec - spec.mean()) / spec.var()
    return skimage.transform.resize(im_norm, (10, 10)).flatten()

def small_spectrogram_max_pooling(spec):
    """
    Using code adapated from:
    http://scikit-image.org/docs/dev/auto_examples/plot_view_as_blocks.html
    """

    spec = force_spectrogram_length(spec, 384)
    im_norm = (spec - spec.mean()) / spec.var()

    view = view_as_blocks(im_norm, (32, 32))
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

    return np.max(flatten_view, axis=2).flatten()


def frequency_max_pooling(spec, normalise=True):
    """
    Using code adapated from:
    http://scikit-image.org/docs/dev/auto_examples/plot_view_as_blocks.html
    """

    if normalise:
        im_norm = (spec - spec.mean()) / spec.var()
    else:
        im_norm = spec

    view = view_as_blocks(im_norm, (8, spec.shape[1]))
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)

    A = np.max(flatten_view, axis=2).flatten()
    B = np.var(flatten_view, axis=2).flatten()
    C = np.mean(flatten_view, axis=2).flatten()

    return np.hstack((A, B, C))



def max_bins(spec):
    """
    Computes features from a spectrogram
    Assumes fixed length
    """
    assert spec.shape[0] == 512
    num_spectogram_bins = 396

    # pad the snippit if it is too short, crop if too long
    spec = force_spectrogram_length(spec, num_spectogram_bins)

    # low pass filter
    filtered_blob = spec[100:, :]

    # feature 1: finding the maximum frequency at each time step
    modes = np.argmax(filtered_blob, axis=0)
    mode_peaks = np.max(filtered_blob, axis=0)

    modes_small = zoom(modes, 0.10)
    mode_peaks_small = zoom(mode_peaks, 0.10)

    return np.hstack((modes_small, mode_peaks_small))
