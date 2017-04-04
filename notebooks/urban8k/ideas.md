IDEAS
=====


2) More preprocessing:
    [Dynamic range compression, log(1 + cx) where c is a constant,
    Mel scale (could allow for more resulution in the time space),
    Whitening (see http://cs231n.github.io/neural-networks-2/, consider the smoothing parameter),
    Use the matplotlib spectrograms,
    data augmentation using PCA to vary the components, e.g. as in ImageNet Classification with Deep Convolutional
Neural Networks]

2) Save out the wav files and inspect for the problems which may be occuring

3) Debug - maybe there is a problem?
    - Bugs found:
        1. Was only using the first 2 seconds of audio (not sure this will be a massive difference)
        2. Was transposing the spectrograms, although this was after the

6) Try replicating the baselines in one method or another...

7) Plot failures against their original sampling frequency, to see if there is still a problem there...
    - I think this is ok, given that the sample frequencies are all the same and ok when loading in the wav files...

8) Similar to C3D and the urban8k papers, split audio into strips and train model using strips. We could whiten/normalise the strips, perhaps including whitened and unwhitened. Open question: Do we do any full-file normalisation first? The classification for all the strips are then aggregated - probably just take the modal.


LATER
=====
1) Augment training data using the other bits
    - I feel this should really be an extra, not a standard


DONE
====
8) Just train and test using the difficult classes, to speed up the cycles...
    - DONE! - Looks like it should be useful

4) Try to improve the training error. It seems that this is suspiciously low
    - Turning off augmentation helps, as expected
    - actually a lot of this seems to be dropout - turning it off means we can get a very low training error

5) Two layers of images for the normalisations
    - Didn't really seem to massively help