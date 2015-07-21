def spectrogram(df_in, meta):
    """
    Compute spectrogram and compute log.
    
    Done in a way to integrate with the AzureML framework; 
    accepts a pandas dataframe of data and another pandas dataframe
    of parameters and metadata
    
    Roughly adapted from here:
    https://mail.python.org/pipermail/chicago/2010-December/007314.html
    
    TODO - high pass and low pass filters
    """
    
    import numpy as np
    import pandas as pd
    
    def get_parameter(parameter_name, default=None):
        """
        helper function for extracting parameters from the meta data frame
        if parameter_name is in meta then return that, else use default
        """
        if parameter_name in meta['gen_spectrogram']:
            return meta['gen_spectrogram'][parameter_name]
        else:
            print "Using default for ", parameter_name
            return default

    # get the sampling rate from the metadata
    sr = meta['data']['sampling_rate']

    # get the spectogram parameters from the metadata
    nfft = get_parameter('nfft')  # number of frequency bins
    window_width = get_parameter('window_width')  # in ms
    overlap = get_parameter('overlap')  # i.e. the stepsize, in ms
    HPF = get_parameter('HPF')  # high pass filter
    LPF = get_parameter('LPF')  # low pass filter
    window_type = get_parameter('window_type', 'hamming')

    # convert step sizes in ms into step sizes in 'pixels'
    nstep = int(sr * overlap)
    nwin  = int(sr * window_width)

    # get the sinal from the pandas dataframe
    x = df_in.as_matrix()

    # Get all windows of x with length n as a single array, using strides to avoid data duplication
    #shape = (nfft, len(range(nfft, len(x), nstep)))
    shape = (nfft, ((x.shape[0] - nfft - 1)/nstep)+1)
    strides = (x.itemsize, nstep*x.itemsize)
    x_wins = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

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

    return pd.DataFrame(fft_mat_lm), meta
    
    
def simple_spectogram(df, meta):
    import numpy as np
    import pandas as pd
    from matplotlib.mlab import specgram
    
    
    def merge_two_dicts(x, y):
        z = x.copy()
        z.update(y)
        return z
    
    # extract numpy array from pandas.DataFrame
    data = np.asarray(df)
    
    spectrum, freqs, t = specgram(data.squeeze())    
    
    # Return value must be of a sequence of pandas.DataFrame
    df = pd.DataFrame(spectrum)
    

    # updata meta
    new_meta = {'spectogram_out': {'frequencies': freqs.tolist(),
                                   'time_points': t.tolist()}}
    
    d = merge_two_dicts(meta.to_dict(), new_meta)
    meta = pd.DataFrame(d)   
    
    return df, meta

