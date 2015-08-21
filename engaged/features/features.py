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
    nbins = spec.shape[0]
    if spec.shape[1] < desired_length:
        to_add = desired_length - spec.shape[1]
        spec = np.hstack((spec, np.zeros((nbins, to_add))))
    elif spec.shape[1] > desired_length:
        spec = spec[:, :desired_length]

    return spec


def small_spectrogram(spec):
    # spec = force_spectrogram_length(spec, 396)
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


def frequency_max_pooling(spec, normalise=True, blocks=True):
    """
    Using code adapated from:
    http://scikit-image.org/docs/dev/auto_examples/plot_view_as_blocks.html
    """

    if normalise:
        im_norm = (spec - spec.mean()) / spec.var()
    else:
        im_norm = spec

    if blocks:
        view = view_as_blocks(im_norm, (8, spec.shape[1]))
        flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
    else:
        flatten_view = spec[None, :, :]

    A = np.max(flatten_view, axis=2).flatten()
    B = np.var(flatten_view, axis=2).flatten()
    C = np.mean(flatten_view, axis=2).flatten()

    return np.hstack((A, B, C))


def bow_hist(spec, bow_hist):
    """
    uses the bow_hist to compute a histogram
    """
    return


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

def mel_max_bins(spec):
    """
    Computes features from a spectrogram
    Assumes fixed length
    """

    # pad the snippit if it is too short, crop if too long
    num_spectogram_bins = 396
    spec = force_spectrogram_length(spec, num_spectogram_bins)

    # feature 1: finding the maximum frequency at each time step
    modes = np.argmax(spec, axis=0)
    mode_peaks = np.max(spec, axis=0)

    modes_small = zoom(modes, 0.10)
    mode_peaks_small = zoom(mode_peaks, 0.10)

    return np.hstack((modes_small, mode_peaks_small))


from scipy.ndimage.filters import gaussian_filter

def gauss_filters_generator(spec, deviation):
    """
    Returns a generator, which gives images of the spectrogram filtered under
    different gaussians
    """
    orders = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 0]]

    for order in orders:
        yield gaussian_filter(spec, deviation, order=order)


def time_max_pooling(array, blockheight):
    """
    Does max pooling along the time dimension.
    Can specify the height of the blocks
    """
    assert array.shape[0] % blockheight == 0

    reshaped_array = array.reshape((-1, blockheight * array.shape[1]))
    return reshaped_array.max(axis=1)


def freq_max_pooling(array, blockwidth):
    """
    Does max pooling along the frequency dimension.
    Can specify the width of the blocks
    """
    assert array.shape[1] % blockheight == 0

    reshaped_array = array.T.reshape((-1, blockwidth * array.shape[0]))
    return reshaped_array.max(axis=1)


def gauss_filter_max_pooling(spec, deviation, blockheight):
    """
    Returns a feature vector, where the spectrogram has been filtered with many
    different gaussian filters, and max pooling has taken place along the time
    dimension for different block sizes
    """

    feature_vectors = []

    for filt in gauss_filters_generator(spec, deviation):
        feature_vectors.append(time_max_pooling(filt, blockheight))

    return np.hstack(feature_vectors)
    # return feature_vectors[0]


def peak_map(spec, patch_hww=6, rescale_val=0.5):
    """
    Returns a patch extracted from the spectrogram around the position of the
    peak frequency response
    """

    # resize and pad the spec
    spec_resized = skimage.transform.rescale(spec, rescale_val)
    temp_spec = np.pad(spec_resized, patch_hww+1, mode='edge')

    # find the first maximum value. Don't want to find a maximum value on the edges
    # so we create a new spectrogram
    max_search_spec = np.pad(spec_resized, patch_hww+1, mode='constant', constant_values=-10000)
    row, col = np.where(max_search_spec == np.max(max_search_spec))
    row = row[0]
    col = col[0]

    # extract the patch
    patch = temp_spec[(row-patch_hww):(row+patch_hww+1), (col-patch_hww):(col+patch_hww+1)].copy()
    patch -= patch.mean()
    patch /= patch.std()

    if patch.shape[0] != 13 or  patch.shape[1] != 13:

        print "Failure!"
        patch = np.zeros((13, 13))

    return np.hstack((np.ravel(patch), np.max(spec)))

