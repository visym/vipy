import numpy as np
import matplotlib.pyplot as plt
from vipy.util import seq, groupby, try_import
from scipy.interpolate import interp1d
try_import('sklearn', 'scikit-learn'); import sklearn.metrics


def cumulative_match_characteristic(similarityMatrix, gtMatrix):
    """CMC curve for probe x gallery similarity matrix (larger is more similar) and ground truth match matrix (one +1 per row, rest zeros)"""
    n_categories = gtMatrix.shape[1]
    n_probe = gtMatrix.shape[0]
    rank = range(1,n_categories + 1)

    for i in range(0,n_probe):
        k = np.argsort(-similarityMatrix[i,:])  # index of sorted rows in descending order
        similarityMatrix[i,:] = similarityMatrix[i,k]  # reorder columns in similarityOrder
        gtMatrix[i,:] = gtMatrix[i,k]  # reorder ground truth in same order

    # Given ground truth matrix, if a row has exactly one "1" then there is a mate.  If a row has all zeros, then the mate does not exist in the gallery
    # if a row has nan, then there is a mate in the gallery, but this was not found in the top-k
    n_pos = np.sum(np.array(np.logical_or((np.sum(gtMatrix, axis=1) == 1.0), np.isnan(np.sum(gtMatrix, axis=1)))).astype(np.float32))
    gtMatrix = np.nan_to_num(gtMatrix)  # convert nans to zeros
    recall = [np.sum(np.max(gtMatrix[:,0:r], axis=1)) / n_pos for r in rank]
    return (rank, recall)


def plot_cmc(rank=None, tdr=None, similarityMatrix=None, truthMatrix=None, label=None, title=None, outfile=None, logscale=True, logy=False, figure=None, style=None, fontsize=None, xlabel='Rank', ylabel='Correct Retrieval Rate', legendSwap=False, errorbars=None, miny=0.0, color=None):
    """Generate cumulative match characteristic (CMC) plot"""

    if rank is None and tdr is None:
        (rank, tdr) = cumulative_match_characteristic(similarityMatrix, truthMatrix)

    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()
        plt.clf()

    if style is None:
        p = plt.plot(rank, tdr, label=label, color=color)
    else:
        p = plt.plot(rank, tdr, style, label=label, color=color)

    if errorbars is not None:
        (x,y,yerr) = zip(*errorbars)  # [(x,y,yerr), (x,y,yerr), ...]
        plt.gca().errorbar(x, y, yerr=yerr, fmt='none', ecolor=plt.getp(p[0], 'color'))  # HACK: force error bars to have same color as plot

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.ylim([miny, 1.0])
    plt.xlim([0.95 if not logscale else 0.95, len(rank)])
    if logscale:
        plt.gca().set_xscale('log')
    if logy:
        plt.gca().set_yscale('log')

    if title is not None:
        plt.title('%s' % (title))
    legendLoc = "lower left" if legendSwap else "lower right"
    if fontsize is None:
        plt.legend(loc=legendLoc)
    else:
        plt.legend(loc=legendLoc, prop={'size':fontsize})
    plt.grid(True)

    # Font size
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    # plt.tight_layout()
    plt.gcf().set_tight_layout(True)

    if outfile is not None:
        print('[vipy.metric.plot_cmc]: saving "%s"' % outfile)
        plt.savefig(outfile)

    else:
        plt.show()


def tdr_at_rank(rank=None, tdr=None, y_true=None, y_pred=None, numGallery=None, at=10):
    """Janus metric for correct retrieval (true detection rate) within a specific rank"""

    if rank is None and tdr is None:
        if y_true is not None and y_pred is not None:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            if numGallery is not None:
                truthMatrix = y_true.reshape((len(y_true) / numGallery, numGallery))
                similarityMatrix = y_pred.reshape((len(y_pred) / numGallery, numGallery))
            elif np.min(y_true.shape) > 1:
                truthMatrix = y_true
                similarityMatrix = y_pred
            else:
                raise ValueError('(y,yhat) must be reshaped into (numProbe x numGallery) of numGallery provided as input')
            (rank, tdr) = cumulative_match_characteristic(similarityMatrix, truthMatrix)
        else:
            raise ValueError('either (rank,tdr) or (y,yhat) required')

    if at > np.max(rank):
        raise ValueError('Selected operating point rank=%d must be less than maximum rank=%d' % (at, np.max(rank)))

    f = interp1d(rank, tdr)
    return f(at)


def auroc(y, yhat):
    return sklearn.metrics.roc_auc_score(y, yhat)


def roc(y_true, y_pred):
    (fpr, tpr, thresholds) = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=1)
    return (fpr, tpr)


def roc_per_image(y_true, y_pred, k_imgindex):
    (fpr, tpr, thresholds) = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=1)
    n_images = len(set(k_imgindex))
    n_fp = len(y_true) - np.sum(y_true)  # total number of false positives
    return (np.array(fpr) * (float(n_fp) / float(n_images)), tpr)  # renormalize false positives


def roc_eer(y_true=None, y_pred=None, fpr=None, tpr=None):
    if (fpr is None) and (tpr is None):
        (fpr, tpr) = roc(y_true, y_pred)
    tnr = 1.0 - np.array(fpr)
    k = np.argmin(np.square(np.array(tnr) - np.array(tpr)))
    eer = fpr[k]
    return eer


def tpr_at_fpr(y_true, y_pred, at=0.01):
    """Janus metric for true positive rate at a specific false positive rate"""
    (fpr, tpr) = roc(y_true, y_pred)
    f = interp1d(fpr, tpr)  # FIXME: kind='cubic' is singular?
    return f(at)


def fpr_at_tpr(y_true, y_pred, at=0.85):
    """Janus metric for false positive rate at a specific true positive rate"""
    (fpr, tpr) = roc(y_true, y_pred)
    f = interp1d(tpr, fpr)  # FIXME: kind='cubic' is singular?
    return f(at)


def plot_roc(y_true=None, y_pred=None, fpr=None, tpr=None, label=None, title=None, outfile=None, figure=None, logx=False, style=None, fontsize=None, xlabel='False Positive Rate', ylabel='True Positive Rate', legendSwap=False, errorbars=None):
    """http://scikit-learn.org/stable/auto_examples/plot_roc.html"""
    if (fpr is None) and (tpr is None):
        (fpr, tpr) = roc(y_true, y_pred)

    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()

    if style is None:
        # Use plot defaults to increment plot style when holding
        p = plt.plot(fpr, tpr, label=label)
    else:
        p = plt.plot(fpr, tpr, style, label=label)

    if errorbars is not None:
        (x,y,yerr) = zip(*errorbars)  # [(x,y,yerr), (x,y,yerr), ...]
        plt.gca().errorbar(x, y, yerr=yerr, fmt='none', ecolor=plt.getp(p[0], 'color'))  # HACK: force error bars to have same color as plot

    if logx is False:
        plt.plot([0, 1], [0, 1], 'k--', label="_nolegend_")
    if logx is True:
        plt.xscale('log')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legendLoc = "upper left" if legendSwap else "lower right"
    if fontsize is None:
        plt.legend(loc=legendLoc)
    else:
        plt.legend(loc=legendLoc, prop={'size':fontsize})
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.autoscale(tight=True)

    if title is not None:
        plt.title(title)

    # Font size
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    plt.gcf().set_tight_layout(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    if outfile is not None:
        print('[vipy.metric.plot_roc]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()


def mean_average_precision(ap):
    """numpy wrapper for mean"""
    return np.mean(ap)


def average_precision(y_true, y_pred):
    """sklearn wrapper"""
    return sklearn.metrics.average_precision_score(y_true, y_pred)


def f1_score(y_true, y_pred):
    """sklearn wrapper"""
    return sklearn.metrics.f1_score(y_true, y_pred)


def confusion_matrix(truthMatrix, similarityMatrix):
    y = np.argmax(truthMatrix, axis=1)
    yhat = np.argmax(similarityMatrix, axis=1)
    return sklearn.metrics.confusion_matrix(y, yhat)


def plot_confusion_matrix(truthMatrix=None, similarityMatrix=None, y=None, yhat=None, figure=None, fontsize=None, xlabel='Predicted Label', ylabel='True Label', outfile=None, normalized=False, classes=None):
    y = np.argmax(truthMatrix, axis=1) if y is None else y
    yhat = np.argmax(similarityMatrix, axis=1) if yhat is None else yhat
    cm = sklearn.metrics.confusion_matrix(y, yhat)
    if normalized:
        cm = np.float32(cm) / (1E-9 + np.float32(np.sum(cm, axis=1).reshape(cm.shape[0], 1)))

    if figure is not None:
        plt.figure(figure)
        plt.clf()
        plt.matshow(cm, fignum=figure)
    else:
        plt.figure()
        plt.clf()
        plt.matshow(cm)

    plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.yticks(tick_marks, classes)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Font size
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    if outfile is not None:
        quietprint('[vipy.metric.plot_confusion_matrix]: saving "%s"' % outfile)
        plt.savefig(outfile, dpi=600)
    else:
        plt.show()

    return cm


def categorization_report(Y_true, Y_pred, labels):
    return sklearn.metrics.classification_report(Y_true, Y_pred, target_names=labels)


def precision_recall(y_true, y_pred):
    (precision, recall, thresholds) = sklearn.metrics.precision_recall_curve(y_true, y_pred)
    return (precision, recall)


def plot_pr(precision, recall, title=None, label='Precision-Recall', outfile=None, figure=None):
    """Plot precision recall curve using matplotlib, with optional figure save"""

    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()
        plt.clf()

    plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if title is not None:
        plt.title('%s' % (title))
    plt.legend(loc="lower left")
    plt.grid(True)

    if outfile is not None:
        quietprint('[vipy.metric.plot_pr]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()


def plot_ap(ap, categories, title=None, outfile=None):
    """Plot Average-Precision bar chart using matplotlib, with optional figure save"""
    plt.bar(range(1,len(ap) + 1), height=ap, width=0.8, bottom=None)
    plt.gca().set_xticks(seq(1.4,len(ap) + 1))
    plt.gca().set_xticklabels(categories, rotation=45)
    plt.ylim([0.0, 1.1])
    plt.ylabel('Average Precision')
    plt.autoscale(tight=True)
    if title is not None:
        plt.title('%s' % (title))
    if outfile is not None:
        quietprint('[vipy.metric.plot_ap]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()


def histogram(freq, categories, barcolors=None, title=None, outfile=None, figure=None, ylabel='Frequency', xrot='vertical'):
    """Plot histogram bar chart using matplotlib with vertical axis labels on x-axis,, with optional figure save"""
    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure(1)
        plt.clf()

    x = range(1, len(categories)+1)
    plt.bar(x, height=freq, width=0.8, bottom=None, color=barcolors)
    plt.xticks(x, list(categories), rotation=xrot)
    plt.autoscale(tight=True)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.subplots_adjust(bottom=0.75)  # tweak
    if title is not None:
        plt.title('%s' % (title))
    plt.tight_layout()            
    if outfile is not None:
        plt.savefig(outfile)
        plt.clf()
        return outfile
    else:
        plt.show()

        
def pie(sizes, labels, explode=None, outfile=None, shadow=False):
    """Generate a matplotlib style pie chart with wedges with specified size and labels, with an optional outfile"""
    plt.figure(1)
    plt.clf()
    
    # pie = plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=shadow, startangle=0)
    pie = plt.pie(sizes, explode=explode, shadow=shadow, startangle=0)
    plt.legend(labels)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout()        
    if outfile is not None:
        plt.savefig(outfile)
        plt.clf()        
        return outfile
    else:
        plt.show()
