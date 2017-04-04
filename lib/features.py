import numpy as np

def compute_ACI(spectro,j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.
    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer
    the singing activity of an avian community: the Acoustic Complexity Index (ACI).
    Ecological Indicators, 11, 868-873.

    Ported from the soundecology R package.
    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)
    """

    #times = range(0, spectro.shape[1], j_bin) # relevant time indices
    times = range(0, spectro.shape[1]-10, j_bin) # alternative time indices to follow the R code

    jspecs = [np.array(spectro[:,i:i+j_bin]) for i in times]  # sub-spectros of temporal size j

    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs] 	# list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values # return main (global) value, temporal values


def ACI_inspired_features(spec):
    return np.abs(np.diff(spec)).sum(axis=1)# / np.sum(spec, axis=1)


def compute_features(_spec, annot):
    spec = _spec.copy()
    spec -= np.median(spec, 1, keepdims=True)
    just_biotic_normed = spec[:, annot > 0.5]
    just_biotic_unnormed = _spec[:, annot > 0.5]

    if just_biotic_normed.size == 0:
        return np.zeros(33)
    X = []
    # X.append(just_biotic_normed.mean(1))
    # X.append(just_biotic_normed.std(1))
    # X.append(just_biotic_unnormed.mean(1))
    # X.append(just_biotic_unnormed.std(1))
    X.append(just_biotic_unnormed.shape[1])
    X.append(ACI_inspired_features(just_biotic_unnormed))
    tmp = np.hstack(X)
    return tmp


def compute_all_feats(specs, fnames, preds):
    X = []
    for fname in fnames:#, annots):
        X.append(compute_features(specs[fname], preds[fname][:, 1]))
#         print X[-1].shape
    return np.vstack(X)
