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