import numpy as np

def spectrogram(wave, sampling_rate, nfft, overlap, HPF=None,
    LPF=None, window_type='hamming', convert_to_mel=False, mel_bins=50,
    stowell_normalise=False):
    """
    Compute spectrogram and compute log.

    overlap is the stepsize in ms
    nfft is the number of frequency bins, also the width of each strip to be extracted...
    HPF is the high pass filter parameter
    LPF is the low pass filter parameter

    Roughly adapted from here:
    https://mail.python.org/pipermail/chicago/2010-December/007314.html

    TODO - high pass and low pass filters
    """

    # convert step sizes in ms into step sizes in 'pixels'
    noverlap = int(sampling_rate * overlap)
    # nwin  = int(sampling_rate * window_width)
    nwin = np.nan
    step = nfft - noverlap

    if noverlap >= nfft:
        raise Exception('Overlap is too high relative to the nfft!')

    # Get all windows of x with length n as a single array, using strides to avoid data duplication

    # shape = (nfft, (wave.shape[-1]-noverlap)//step)
    shape = (nfft, (wave.shape[-1]-noverlap)//step)
    strides = (wave.strides[0], step*wave.strides[0])

    x_wins = np.lib.stride_tricks.as_strided(wave, shape=shape, strides=strides)

    # Apply hamming window
    if window_type == 'hamming':
        x_wins_ham = np.hamming(x_wins.shape[0])[..., np.newaxis] * x_wins
    else:
        raise Exception('Window type %s not implemented' % window_type)

    # compute fft
    fft_mat = np.fft.fft(x_wins_ham, n=int(nfft), axis=0)[:(nfft/2+1), :]

    # log magnitude
    fft_mat_lm = np.log(np.abs(fft_mat))

    if HPF is not None or LPF is not None:
        # TODO - high pass and low pass filters here
        raise Exception('Not implemented yet')

    if convert_to_mel:
        out_spec = spec_to_mel(fft_mat_lm, sampling_rate, mel_bins)
    else:
        out_spec = fft_mat_lm

    if stowell_normalise:
        out_spec -= np.median(out_spec, axis=1)[:, None]
        out_spec[out_spec<0] = 0

    return out_spec, float(sampling_rate) / float(float(nwin) - 2*float(step))

def spec_to_mel(spec, audio_sample_rate, num_mel_bands):
    """
    Converts a standard spectrogram to a mel scaled spectrogram
    """
    M = mel_binning_matrix(
        specgram_window_size=(spec.shape[0] - 1) * 2,
        sample_frequency=audio_sample_rate,
        num_mel_bands=num_mel_bands)

    return np.dot(M.T, spec)



# This next bit is from this:
# https://gist.github.com/benanne/3274371
def freq2mel(freq):
    return 1127.01048 * np.log(1 + freq / 700.0)

def mel2freq(mel):
    return (np.exp(mel / 1127.01048) - 1) * 700

def mel_binning_matrix(specgram_window_size, sample_frequency, num_mel_bands):
    """
    function that returns a matrix that converts a regular DFT to a mel-spaced DFT,
    by binning coefficients.

    specgram_window_size: the window length used to compute the spectrograms
    sample_frequency: the sample frequency of the input audio
    num_mel_bands: the number of desired mel bands.

    The output is a matrix with dimensions (specgram_window_size/2 + 1, num_bands)
    """
    min_freq, max_freq = 0, sample_frequency / 2
    min_mel = freq2mel(min_freq)
    max_mel = freq2mel(max_freq)
    num_specgram_components = specgram_window_size / 2 + 1
    m = np.zeros((num_specgram_components, num_mel_bands))

    r = np.arange(num_mel_bands + 2) # there are (num_mel_bands + 2) filter boundaries / centers

    # evenly spaced filter boundaries in the mel domain:
    mel_filter_boundaries = r * (max_mel - min_mel) / (num_mel_bands + 1) + min_mel

    def coeff(idx, mel): # gets the unnormalised filter coefficient of filter 'idx' for a given mel value.
        lo, cen, hi = mel_filter_boundaries[idx:idx+3]
        if mel <= lo or mel >= hi:
            return 0
        # linearly interpolate
        if lo <= mel <= cen:
            return (mel - lo) / (cen - lo)
        elif cen <= mel <= hi:
            return 1 - (mel - cen) / (hi - cen)


    for k in xrange(num_specgram_components):
        # compute mel representation of the given specgram component idx
        freq = k / float(num_specgram_components) * (sample_frequency / 2)
        mel = freq2mel(freq)
        for i in xrange(num_mel_bands):
            m[k, i] = coeff(i, mel)

    # normalise so that each filter has unit contribution
    return m / m.sum(0)
