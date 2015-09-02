import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def eval_seconds(gt, pred, total_length):
    """
    Evaluates the overall prediction for a file
    Total length is in seconds
    """
    results = {}
    results['biotic'] = (float(gt.sum()) / gt.size) * total_length
    results['predicted'] = \
        (float(pred.sum()) / pred.size) * total_length
    results['predicted_hard'] = \
        (float((pred>0.5).sum()) / pred.size) * total_length
    results['soft_variance'] = np.var(pred)
    # np.hisatogram is a bit slow, so doing this instead
    results['soft_binning'] = \
        np.hstack([
            (pred < 0.1).sum(),
            np.logical_and(pred > 0.1, pred < 0.2).sum(),
            np.logical_and(pred > 0.2, pred < 0.3).sum(),
            np.logical_and(pred > 0.3, pred < 0.4).sum(),
            np.logical_and(pred > 0.4, pred < 0.5).sum(),
            np.logical_and(pred > 0.5, pred < 0.6).sum(),
            np.logical_and(pred > 0.6, pred < 0.7).sum(),
            np.logical_and(pred > 0.7, pred < 0.8).sum(),
            np.logical_and(pred > 0.8, pred < 0.9).sum(),
            np.logical_and(pred > 0.9, pred < 1.0).sum()
        ])
    results['length'] = gt.shape[0]
    return results


def plot_roc_curve(gt, pred, label="", plot_midpoint=True):
    """
    Plots a single roc curve with a dot
    """

    # evaluating and finding the curve midpoint
    fpr, tpr, thresh = metrics.roc_curve(gt, pred.ravel())
    roc_auc = metrics.auc(fpr, tpr)

    # plotting curve
    plt.plot(fpr, tpr , label='%s (area = %0.2f)' % (label, roc_auc))

    if plot_midpoint:
        mid_idx = np.argmin(np.abs(thresh-0.5))
        plt.plot(fpr[mid_idx], tpr[mid_idx], 'bo')

    # labels and legends
    plt.legend(loc='best')
    plt.xlabel('FPR')
    plt.ylabel('TPR')


def normalised_accuracy(gt, pred, threshold=0.5):
    accs = 0
    unique_classes = np.unique(gt)

    for target in unique_classes:
        idxs = gt == target
        accs += metrics.accuracy_score(gt[idxs], pred[idxs] > threshold)

    return accs / len(unique_classes)
