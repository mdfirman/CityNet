import numpy as np
from scipy import ndimage

def median_normalise(slice_in):
    return slice_in - np.median(slice_in, axis=1)[:, None]


def local_normalise(im, sigma_mean, sigma_std):
    """
    From: https://github.com/benanne/kaggle-whales/blob/master/pipeline_job.py
    based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
    """
    means = ndimage.gaussian_filter(im, sigma_mean)
    im_centered = im - means
    stds = np.sqrt(ndimage.gaussian_filter(im_centered**2, sigma_std))
    return im_centered / stds


EPS = 0.000000001

def normalise(slice_in, strategy, params=None):
    '''
    perform strategy on a single spectrogram

    '''

    if strategy == 'stowell_half':
        return median_normalise(slice_in)

    if strategy == 'stowell_half_rescale':
        normed_slice = median_normalise(slice_in)
        return normed_slice / (normed_slice.var() + EPS)

    elif strategy == 'stowell_full':
        slice_out = median_normalise(slice_in)
        slice_out[slice_out < 0] = 0
        return slice_out

    elif strategy == 'overall_median':
        return slice_in - np.median(slice_in)

    elif strategy == 'overall_median_rescale':
        normed_slice = slice_in - np.median(slice_in)
        return normed_slice / (normed_slice.var() + EPS)

    elif strategy == 'sum_to_one':
        # take care to do this per second of audio...
        desired_sum = slice_in.shape[1]
        slice_out = slice_in / slice_in.sum()
        return slice_out * desired_sum

    elif strategy == 'equal_power':
        # take care to do this per second of audio...
        power_spec = np.exp(slice_in)
        desired_sum = power_spec.shape[1]
        normalised_power_spec = power_spec / power_spec.sum()
        return np.log(normalised_power_spec * desired_sum)

    elif strategy == 'local_normalisation':
        # this is the hardest one...
        return local_normalise(slice_in, params[0], params[1])

    else:
        raise Exception("Unknown normalisation %s" % strategy)
