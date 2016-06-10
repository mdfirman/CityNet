import lasagne
import theano.tensor as T
import nolearn.lasagne
from ml_helpers import minibatch_generators as mbg
import numpy as np


def force_make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


class Log1Plus(lasagne.layers.Layer):
    def __init__(self, incoming, off=lasagne.init.Constant(1.0), mult=lasagne.init.Constant(1.0), **kwargs):
        super(Log1Plus, self).__init__(incoming, **kwargs)
        num_channels = self.input_shape[1]
        self.off = self.add_param(off, shape=(num_channels,), name='off')
        self.mult = self.add_param(mult, shape=(num_channels,), name='mult')

    def get_output_for(self, input, **kwargs):
        return T.log(self.off.dimshuffle('x', 0, 'x', 'x') + self.mult.dimshuffle('x', 0, 'x', 'x') * input)

    def get_output_shape_for(self, input_shape):
        return input_shape


class SpecSampler(object):

    def __init__(self, specs, labels, hww, do_aug, learn_log):
        self.do_aug = do_aug
        self.learn_log = learn_log
        self.hww = hww

        blank_spec = np.zeros((specs[0].shape[0], hww))
        self.specs = np.hstack([blank_spec] + specs + [blank_spec])[None, ...]

        blank_label = np.zeros(hww) - 1
        self.labels = np.hstack([blank_label] + labels + [blank_label])

        which_spec = [ii * np.ones_like(lab).astype(np.int32) for ii, lab in enumerate(labels)]
        self.which_spec = np.hstack([blank_label] + which_spec + [blank_label])

        if learn_log:
            self.medians = np.zeros((len(specs), specs[0].shape[0], hww*2))
            for idx, spec in enumerate(specs):
                self.medians[idx] = np.median(spec, axis=1, keepdims=True)

    def sample(self, num_per_class, seed=None):
        num_samples = num_per_class * 2
        channels = self.specs.shape[0]
        height = self.specs.shape[1]

        if seed is not None:
            np.random.seed(seed)

        X = np.zeros((num_samples, channels, height, self.hww*2), np.float32)
        y = np.zeros(num_samples) * np.nan
        if self.learn_log:
            X_medians = np.zeros((num_samples, channels, height, self.hww*2), np.float32)
        count = 0

        for cls in [0, 1]:
            possible_locs = np.where(self.labels==cls)[0]

            if len(possible_locs) >= num_per_class:
                sampled_locs = np.random.choice(possible_locs, num_per_class, replace=False)

                for loc in sampled_locs:
                    X[count] = self.specs[:, :, (loc-self.hww):(loc+self.hww)]
                    y[count] = cls
                    if self.learn_log:
                        which = self.which_spec[loc]
                        X_medians[count] = self.medians[which]
                    count += 1

        # doing augmentation
        if self.do_aug:
            if self.learn_log:
                X *= (1.0 + np.random.randn(num_samples, 1, 1, 1) * 0.1)
            else:
                X *= (1.0 + np.random.randn(num_samples, 1, 1, 1) * 0.1)
                X += np.random.randn(num_samples, 1, 1, 1) * 0.05

        if self.learn_log:
            xb = {'input': X.astype(np.float32), 'input_med': X_medians.astype(np.float32)}
            return xb, y.astype(np.int32)

        else:
            # remove ones we couldn't get
            return X.astype(np.float32), y.astype(np.int32)
