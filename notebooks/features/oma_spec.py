import numpy as np
from skimage import filters

def gen_mag_spectrogram_mod(x, fs, ms, overlap_perc=0.95, flip=False, blur=False, log=True):
    """
    Computes magnitude spectrogram
    """

    nfft = int(ms*fs)
    noverlap = int(overlap_perc*nfft)

    # window data
    step = nfft - noverlap
    shape = (nfft, (x.shape[-1]-noverlap)//step)
    strides = (x.strides[0], step*x.strides[0])
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # apply window
    x_wins_han = np.hanning(x_wins.shape[0])[..., np.newaxis] * x_wins

    # do fft
    complex_spec = np.fft.rfft(x_wins_han, axis=0)
    #print complex_spec.shape

    # calculate magnitude
    mag_spec = np.conjugate(complex_spec) * complex_spec
    mag_spec = mag_spec.real
    # same as:
    #mag_spec = np.square(np.absolute(complex_spec))

    # orient correctly and remove dc component
    spec = mag_spec[1:, :]

    if flip:
        spec = np.flipud(spec)

    # scale
    log_scaling = 2.0 * (1.0 / fs) * (1.0/(np.abs(np.hanning(nfft))**2).sum())
    spec *= log_scaling
    
    if log:
        spec = np.log(1+spec) 

    if blur:
        spec = filters.gaussian_filter(spec, 2.0)

    return spec#spec  /  spec.sum()

