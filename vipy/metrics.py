import numpy as np
import matplotlib.pyplot as plt
from vipy.util import seq, groupby, try_import, temppng
from vipy.math import interp1d
from vipy.globals import log
from vipy.image import Image, owl
from vipy.math import gaussian


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


def cmc_curve(rank=None, tdr=None, similarityMatrix=None, truthMatrix=None, label=None, title=None, outfile=None, logscale=True, logy=False, figure=None, style=None, fontsize=None, xlabel='Rank', ylabel='Correct Retrieval Rate', legendSwap=False, errorbars=None, miny=0.0, color=None):
    """Plot cumulative match characteristic (CMC) curve """

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
        log.info('[vipy.metric.plot_cmc]: saving "%s"' % outfile)
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


def receiver_operating_curve(y_true=None, y_pred=None, fpr=None, tpr=None, label=None, title=None, outfile=None, figure=None, logx=False, style=None, fontsize=None, xlabel='False Positive Rate', ylabel='True Positive Rate', legendSwap=False, errorbars=None):
    """Plot ROC: http://scikit-learn.org/stable/auto_examples/plot_roc.html"""
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
        log.info('[vipy.metric.plot_roc]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()



def confusion_matrix_plot(cm, outfile=None, figure=None, fontsize=5, xlabel=None, ylabel=None, classes=None, colorbar=False, figsize=None):
    """Generate a confusion matrix plot for a confusion matrix cm"""

    outfile = outfile if outfile is not None else temppng()
    figure = 1 if figure is None else figure
    
    if figsize:
        plt.figure(figure, figsize=figsize)
    else:
        plt.figure(figure)
    plt.clf()
    plt.matshow(cm, fignum=figure)

    if colorbar:
        plt.colorbar()
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.yticks(tick_marks, classes)
        plt.xticks(tick_marks, classes, rotation='vertical')

    xl = plt.xlabel(xlabel) if xlabel is not None else None
    yl = plt.ylabel(ylabel) if ylabel is not None else None

    # Font size
    ax = plt.gca()
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    plt.savefig(outfile, bbox_extra_artists=(yl,) if yl is not None else None, bbox_inches='tight', dpi=600)

    return outfile
    

def precision_recall_curve(precision, recall, title=None, label='Precision-Recall', outfile=None, figure=None, fontsize=8, loc='upper right'):
    """Plot precision recall curve using matplotlib, with optional figure save.  Call this multiple times with same figure number to plot multiple curves."""

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
    plt.legend(loc=loc, fontsize=fontsize)
    plt.grid(True)

    if outfile is not None:
        log.info('[vipy.metric.plot_pr]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()


def average_precision_chart(ap, categories, title=None, outfile=None):
    """Plot Average-Precision bar chart using matplotlib, with optional figure save"""
    plt.bar(range(1,len(ap) + 1), height=ap, width=0.8, bottom=None)
    plt.gca().set_xticks(seq(1,len(ap)))
    plt.gca().set_xticklabels(categories, rotation=45)
    plt.ylim([0.0, 1.1])
    plt.ylabel('Average Precision')
    plt.autoscale(tight=True)
    if title is not None:
        plt.title('%s' % (title))
    if outfile is not None:
        log.info('[vipy.metric.plot_ap]: saving "%s"' % outfile)
        plt.savefig(outfile)
    else:
        plt.show()


def histogram(freq, categories, barcolors=None, title=None, outfile=None, figure=None, ylabel='Frequency', xrot='vertical', xlabel=None, fontsize=10, xshow=True):
    """Plot histogram bar chart using matplotlib with vertical axis labels on x-axis,, with optional figure save.
       
       Inputs:
          -freq:  the output of (freq, categories) = np.histogram(..., bins=n)
          -categories [list]:  a list of category names that must be length n, or the output of (f,c) = np.histogram(...) and categories=c[:-1]
          -xrot ['vertical'|None]:  rotate the xticks
          -barcolors [list]:  list of named colors equal to the length of categories
    """
    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure(1)
        plt.clf()

    x = range(1, len(categories)+1)
    plt.bar(x, height=freq, width=0.8, bottom=None, color=barcolors)
    if xshow:
        plt.xticks(x, list(categories), rotation=xrot, fontsize=fontsize)
    plt.autoscale(tight=True)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
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

    return outfile


def pie(sizes, labels, explode=None, outfile=None, shadow=False, legend=True, fontsize=10, rotatelabels=False):
    """Generate a matplotlib style pie chart with wedges with specified size and labels, with an optional outfile"""
    plt.figure(1)
    plt.clf()
    
    # pie = plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=shadow, startangle=0)
    if legend:
        pie = plt.pie(sizes, explode=explode, shadow=shadow, startangle=0,  textprops={'fontsize': fontsize}, rotatelabels=rotatelabels)
        plt.legend(labels)
    else:
        pie = plt.pie(sizes, explode=explode, shadow=shadow, startangle=0, labels=labels,  textprops={'fontsize': fontsize}, rotatelabels=rotatelabels)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout()        
    if outfile is not None:
        plt.savefig(outfile)
        plt.clf()        
        return outfile
    else:
        plt.show()

        
def scatterplot(X, labels, outfile=None):
    """Generate a scatterplot of 2D points in an Nx2 matrix (X) with provided category labels in list of length N (labels).  Each label will be assigned a unique color.  Scatterplot saved to outfile (if provided).""" 
    assert isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] == 2
    assert len(X) == len(labels)
    import vipy.show
    
    plt.clf()
    #plt.figure()
    plt.grid(True)
    colors = vipy.show.colorlist()
    d_label_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(set(labels))}
    plt.axis('equal')  
    for y in sorted(set(labels)):
        x = np.array([xi for (xi,yi) in zip(X, labels) if yi == y])
        plt.scatter(x[:,0], x[:,1], c=d_label_to_color[y], label=y)
    plt.axis([np.min(X), np.max(X), np.min(X), np.max(X)])                
    plt.legend()
    plt.gca().set_axisbelow(True)  # grid behind

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        plt.clf()        
        return outfile
    else:
        plt.ion()
        plt.show()
        plt.pause(0.001)

        
def ascii_bar_chart(soft_labels, bar_width=40, min_conf=0, max_conf=1):
    """Given a list of soft_labels = [(label, confidence), ...], return an ascii horizontal bar chart for each label sorted by confidence.

    Confidences are specied for the provide range (min_conf, max_conf)
    The bar_width controls how wide the overall bars are in characters
    
    >>> print(vipy.metrics.ascii_bar_chart([('A',1), ('B',0.5), ('C',0.1)]))
    [████████████████████████████████████████]  A (1.000)
    [████████████████████....................]  B (0.500)
    [████....................................]  C (0.100)

    """
    cmin = min_conf if min_conf is not None else min(c for (l,c) in soft_labels)
    cmax = max_conf if max_conf is not None else max(c for (l,c) in soft_labels)
    num_blocks = lambda c: int(round(((c-cmin)/(cmax-cmin)) * bar_width))      
    num_dots   = lambda c: int(round((1 - ((c - cmin)/(cmax-cmin))) * bar_width))
    return '\n'.join(["[%s]  %s (%1.3f)" % ('█' * num_blocks(c) + '.' * num_dots(c), lbl,c) for (lbl,c) in sorted(soft_labels, key=lambda x: x[1], reverse=True)])        


class SSIM():
    """Structural similarity (SSIM) index """
    """Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, "Image quality assessment: from error visibility to structural similarity". IEEE Transactions on Image Processing. 13 (4): 600–612"""

    def __init__(self, do_alignment=True, min_matches_for_alignment=10, num_matches_for_alignment=500, K1=0.01, K2=0.03):        
        self.do_alignment = do_alignment
        self.min_matches_for_alignment = min_matches_for_alignment
        self.num_matches_for_alignment = num_matches_for_alignment
        self.K1 = K1
        self.K2 = K2

    def __repr__(self):
        return str('<vipy.ssim: do_alignment=%s, min_matches_for_alignment=%d, num_matches_for_alignment=%d, K1=%f, K2=%f>' % (str(self.do_alignment), self.min_matches_for_alignment, self.num_matches_for_alignment, self.K1, self.K2))

    def match(self, img1, img2):
        """Return a set of matching points in img1 (MxN uint8 numpy) and img2 (MxN uint8 numpy) in the form suitable for homography estimation"""
        try_import('cv2', 'opencv-python'); import cv2 
        
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        # Match descriptors.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x:x.distance)[:self.num_matches_for_alignment]

        img1_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        img2_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        return (img1_pts, img2_pts)

    def warp(self, src_pts, dst_pts, im_src):
        """Warp an image im_src with points src_pts to align with dst_pts"""

        try_import('cv2', 'opencv-python'); import cv2
        
        if src_pts.shape[0] < self.min_matches_for_alignment:
            raise ValueError('Invalid number of inliers')
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

    def align(self, img1, img2):
        """Return an image which is the warped version of img1 (MxN uint8 numpy) that aligns with img2 (MxN uint8 numpy)"""
        (p1, p2) = self.match(img1, img2)
        return self.warp(p1, p2, img1)

    def rgb2gray(self, I):
        """Convert RGB image to grayscale; accesory function"""
        R = I[:,:,0]
        G = I[:,:,1]
        B = I[:,:,2]
        return 0.299 * R + 0.587 * G + 0.114 * B

    def similarity(self, I1, I2, returnMap=True):
        """Compute the Structural Similarity Index (SSIM) score of two images
        Inputs:
        1) I1, image array
        2) I2, image array
        3) K1, float (optional, default=0.01)
        - constant
        4) K2, float (optional, default=0.03)
        - constant
        Outputs:
        1) out; float
        - SSIM score
        2) ssim_map; 2-D image array
        - SSIM map"""

        I1 = self.rgb2gray(I1) if I1.ndim == 3 else I1
        I2 = self.rgb2gray(I2) if I2.ndim == 3 else I2

        C1 = np.power(self.K1 * 255,2)
        C2 = np.power(self.K2 * 255,2)

        w = gaussian(11,1.5)
        f = np.zeros((11,11))
        for k in range(len(w)):
            for k2 in range(len(w)):
                f[k,k2] = np.multiply(w[k],w[k2])
        f = np.true_divide(f,np.sum(f))

        try_import('scipy.signal', 'scipy'); from scipy.signal import convolve2d        
        ux = convolve2d(I1,f,mode='same')
        uy = convolve2d(I2,f,mode='same')

        # Compute SSIM constants
        ux_sq = np.power(ux,2)
        uy_sq = np.power(uy,2)
        ux_uy = np.multiply(ux,uy)

        sig_x = convolve2d(np.power(I1,2),f,mode='same') - ux_sq
        sig_y = convolve2d(np.power(I2,2),f,mode='same') - uy_sq
        sig_xy = convolve2d(np.multiply(I1,I2),f,mode='same') - ux_uy

        # Core SSIM Equation
        ssim_map = np.divide(np.multiply(2 * ux_uy + C1, 2 * sig_xy + C2),
                             np.multiply(ux_sq + uy_sq + C1, sig_x + sig_y + C2))

        out = np.mean(ssim_map)

        return (out, ssim_map) if returnMap else out

    
    def ssim(self, im_reference, im_degraded, returnAligned=False):
        """Return structural similarity score when aligning im_degraded to im_reference

        >>> (ssim, im_aligned) = vipy.ssim.SSIM(do_alignment=True).ssim(vipy.image.squareowl(), vipy.image.squareowl().rotate(0.01), returnAligned=True)
        >>> print(ssim)
        >>> im_aligned.show(figure=1)
        >>> vipy.image.squareowl().rotate(0.01).show(figure=2)
        
        """
        assert isinstance(im_reference, np.ndarray) or isinstance(im_reference, Image)
        assert isinstance(im_degraded, np.ndarray) or isinstance(im_degraded, Image)
        
        img_degraded = im_degraded.lum().numpy() if isinstance(im_degraded, Image) else im_degraded
        img_reference = im_reference.lum().numpy() if isinstance(im_reference, Image) else im_reference
        
        img_degraded_aligned = self.align(img_degraded, img_reference) if self.do_alignment else im_degraded
        ssim = self.similarity(img_degraded_aligned, img_reference, returnMap=False)
        return (ssim, Image(array=img_degraded_aligned, colorspace='lum')) if returnAligned else ssim

    @staticmethod
    def demo(im=None):
        """Synthetically rotate an image by 4 degrees, and compute structural similarity with and without alignment, return images
        
        >>> (image, degraded_image, aligned_image) = vipy.ssim.demo(vipy.image.Image(filename='/path/to/image.jpg')))
        
        """
        assert im is None or isinstance(im, Image)
        im = owl().centersquare() if im is None else im
        
        # Synthetic degradation: 1-channel uint8
        (im, im_degraded) = (im.lum(), im.clone().rotate(4*(np.pi/180.0)).lum())
    
        # SSIM
        (ssim_aligned, im_aligned) = SSIM(do_alignment=True).ssim(im.numpy(), im_degraded.numpy(), returnAligned=True)
        (ssim_unaligned) = SSIM(do_alignment=False).ssim(im.numpy(), im_degraded.numpy())    
        log.info('Structural similarity score (aligned): %f' % ssim_aligned)
        log.info('Structural similarity score (unaligned): %f' % ssim_unaligned)
        return (im.show(figure=1),
                im_degraded.show(figure=2),
                im_aligned.show(figure=3))
