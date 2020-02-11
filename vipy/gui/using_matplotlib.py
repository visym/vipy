import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from vipy.util import islist, temppng


FIGHANDLE = {}
matplotlib.rcParams['toolbar'] = 'None'


# Optional latex strings in captions
try:
    from distutils.spawn import find_executable
    if not find_executable('latex'):
        raise
    matplotlib.rc('text', usetex=True)  # requires latex installed
except:
    pass  # ignored if latex is not installed


def flush():
    plt.pause(0.001)

    
def show(fignum):
    plt.ion()
    plt.draw()
    plt.show()


def noshow(fignum):
    closeall()
    plt.ioff()


def savefig(filename=None, fignum=None, pad_inches=0, dpi=None, bbox_inches='tight', format=None):
    if fignum is not None:
        plt.figure(fignum)
    if filename is None:
        filename = temppng()
    plt.savefig(filename, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches, format=format)
    return filename


def figure(fignum=None):
    if fignum is not None:
        plt.figure(fignum)
    else:
        plt.figure()
    return plt


def close(fignum):
    global FIGHANDLE
    if fignum in FIGHANDLE:
        FIGHANDLE.pop(fignum, None)
        return plt.close(fignum)
    else:
        return None


def closeall():
    global FIGHANDLE
    FIGHANDLE = {}
    return plt.close('all')


def _imshow_tight(img, fignum=None):
    """Helper function to show an image in a figure window"""
    dpi = 100.0
    fig = plt.figure(fignum, dpi=dpi, figsize=(img.shape[1] / dpi, img.shape[0] / dpi))
    plt.clf()

    # Tight axes
    ax = plt.Axes(fig, [0., 0., 1.0, 1.0], frameon=False)
    fig.add_axes(ax)
    for a in plt.gcf().axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    imh = plt.imshow(img, animated=True, interpolation='nearest', aspect='equal')
    return (fig.number, imh)


def imshow(img, fignum=None):
    """Show an image in a figure window (optionally visible), reuse previous figure if it is the same shape"""
    global FIGHANDLE
    if fignum in plt.get_fignums() and fignum in FIGHANDLE and FIGHANDLE[fignum].get_size() == img.shape[0:2]:
        # Delete all polygon and text overlays from previous drawing
        FIGHANDLE[fignum].set_data(img)
        for c in plt.gca().get_children():
            if 'Text' in c.__repr__() or 'Polygon' in c.__repr__() or 'Circle' in c.__repr__() or 'Line' in c.__repr__() or 'Patch' in c.__repr__():
                try:
                    c.remove()
                except:
                    pass
    else:
        if fignum in plt.get_fignums() and fignum in FIGHANDLE:
            close(fignum)
        (fignum, imh) = _imshow_tight(img, fignum=fignum)
        FIGHANDLE[fignum] = imh
    return fignum


def boundingbox(img, xmin, ymin, xmax, ymax, bboxcaption=None, fignum=None, bboxcolor='green', facecolor='white', facealpha=0.5, textcolor='black', textfacecolor='white', fontsize=10, captionoffset=(0,0)):
    """Draw a captioned bounding box on a previously shown image"""
    plt.figure(fignum)
    plt.axvspan(xmin, xmax, ymin=1.0 - np.float32(float(ymax) / float(img.shape[0])), ymax=1 - np.float32(float(ymin) / float(img.shape[0])), edgecolor=bboxcolor, facecolor=facecolor, linewidth=3, fill=True, alpha=facealpha, label=None, capstyle='round', joinstyle='bevel', clip_on=True)

    # Text string
    if bboxcaption is not None:
        # clip_on clips anything outside the image
        plt.text(xmin + captionoffset[0], ymin + captionoffset[1], bboxcaption, color=textcolor, bbox=dict(facecolor=textfacecolor, edgecolor=textcolor, alpha=1, boxstyle='round'), fontsize=fontsize, clip_on=True)

    return fignum


def imdetection(img, detlist, fignum=None, bboxcolor='green', do_caption=True, facecolor='white', facealpha=0.5, textcolor='green', textfacecolor='white', captionlist=None, fontsize=10, captionoffset=(0,0)):
    """Show bounding boxes from a list of vipy.object.Detections on the same image, plotted in list order with optional captions """

    # Create image
    fignum = imshow(img, fignum=fignum)

    # Valid detections
    for (k,det) in enumerate(detlist):
        if do_caption and captionlist is not None:
            bboxcaption = str(captionlist[k])
        elif do_caption:
            bboxcaption = str(det.category())
        else:
            bboxcaption = None

        if islist(bboxcolor):
            bboxcolor_ = bboxcolor[k]
        else:
            bboxcolor_ = bboxcolor

        if islist(textcolor):
            textcolor_ = textcolor[k]
        else:
            textcolor_ = textcolor

        boundingbox(img, xmin=det.xmin(), ymin=det.ymin(), xmax=det.xmax(), ymax=det.ymax(), bboxcaption=bboxcaption,
                    fignum=fignum, bboxcolor=bboxcolor_, facecolor=facecolor, facealpha=facealpha, textcolor=textcolor_, textfacecolor=textfacecolor, fontsize=fontsize, captionoffset=captionoffset)

    return fignum


def imframe(img, fr, color='b', markersize=10, label=None, figure=None):
    """Show a scatterplot of fr=[[x1,y1],[x2,y2]...] 2D points overlayed on an image"""
    if figure is not None:
        fig = plt.figure(figure)
    else:
        fig = plt.figure()

    figure = plt.gcf().number

    plt.clf()

    ax = plt.Axes(fig, [0., 0., 1., 1.], frameon=False)
    fig.add_axes(ax)

    plt.axis('off')
    ax.set_axis_off()
    for a in plt.gcf().axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    plt.autoscale(tight=True)
    plt.plot(fr[:,0],fr[:,1],'%s.' % color, markersize=markersize, axes=ax)

    if label is not None:
        for ((x,y),lbl) in zip(fr, label):
            ax.text(x, y, lbl, color='white')

    return plt


def frame(fr, im=None, color='b.', markersize=10, figure=None, caption=None):
    """Show a scatterplot of fr=[[x1,y1],[x2,y2]...] 2D points"""
    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()
        plt.clf()

    plt.axes([0,0,1,1])
    plt.plot(fr[:,0],fr[:,1],color)
    plt.axis('off')
    plt.draw()


def colorlist():
    """Return a list of named colors"""
    colorlist = [str(name) for (name, hex) in matplotlib.colors.cnames.items()]
    primarycolorlist = ['green','blue','red','cyan','orange', 'yellow','violet']
    return primarycolorlist + [c for c in colorlist if c not in primarycolorlist]
