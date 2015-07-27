import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import skimage.transform

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
    modes_small = zoom(modes, 0.05)

    return modes_small
  