import numpy as np
from sklearn import metrics


def form_correct_shape_array(X):
    """
    Given a list of images each of the same size, returns an array ofthe shape
    required by Lasagne/Theano
    """
    temp = np.dstack(X).transpose((2, 0, 1))
    S = temp.shape
    return temp.astype(np.float32).reshape(S[0], 1, S[1], S[2])


def multiclass_auc(gt, preds):
    """
    Compute a class-weighted multiclass AUC, as defined in section 9.2 of [1].

    Parameters
    ----------
    gt : sequence
        A 1D vector of ground truth class labels
    preds : array
        An array of classifier predictions, where columns correspond to classes
        and rows to data instances.

    [1] https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
    """

    assert gt.shape[0] == preds.shape[0]
    assert gt.max() < preds.shape[1]

    # compute AUC for each class in turn
    aucs = []
    for class_id in range(preds.shape[1]):
        class_preds = preds[:, class_id]
        class_gt = gt == class_id
        aucs.append(metrics.roc_auc_score(class_gt, class_preds))

    # return the class-weighted mean of the AUCs
    return np.average(np.array(aucs), weights=np.bincount(gt))
