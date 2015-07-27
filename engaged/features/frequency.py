import numpy as np

def spectrogram(wave, sampling_rate, nfft, window_width, overlap, HPF=None, LPF=None, window_type='hamming'):
    """
    Compute spectrogram and compute log.
    
    window_width in ms
    overlap is the stepsize in ms
    nfft is the number of frequency bins
    HPF is the high pass filter parameter
    LPF is the low pass filter parameter
    
    Roughly adapted from here:
    https://mail.python.org/pipermail/chicago/2010-December/007314.html
    
    TODO - high pass and low pass filters
    """

    # convert step sizes in ms into step sizes in 'pixels'
    nstep = int(sampling_rate * overlap)
    nwin  = int(sampling_rate * window_width)

    # Get all windows of x with length n as a single array, using strides to avoid data duplication
    #shape = (nfft, len(range(nfft, len(x), nstep)))
    shape = (nfft, ((wave.shape[0] - nfft - 1)/nstep)+1)
    strides = (wave.itemsize, nstep*wave.itemsize)
    x_wins = np.lib.stride_tricks.as_strided(wave, shape=shape, strides=strides)

    # Apply hamming window
    if window_type == 'hamming':
        x_wins_ham = np.hamming(x_wins.shape[0])[..., np.newaxis] * x_wins
    else:
        raise Exception('Window type %s not implemented' % window_type)

    # compute fft
    fft_mat = np.fft.fft(x_wins_ham, n=int(nfft), axis=0)[:(nfft/2), :]

    # log magnitude
    fft_mat_lm = np.log(np.abs(fft_mat))

    if HPF is not None or LPF is not None:
        # TODO - high pass and low pass filters here
        raise Exception('Not implemented yet')

    return fft_mat_lm
