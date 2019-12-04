import matplotlib
import sys

# if sys.platform == 'linux' or sys.platform == 'linux2':
#     try:
#         plt.switch_backend('tkagg')  # linux display
#     except:
#         matplotlib.use('Agg')
#         pass  # linux headless

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from vipy.util import islist, temppng, mat2gray
import os
import sys


FIGHANDLE = {}

matplotlib.rcParams['toolbar']= 'None'
plt.ion()
plt.show()
#matplotlib.rc('text', usetex=True)  # requires latex installed

def savefig(filename=None, handle=None, pad_inches=0, dpi=None, bbox_inches='tight'):
    if handle is not None:
        plt.figure(handle)
    if filename is None:
        filename = temppng()
    plt.savefig(filename, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches)
    return filename

def figure(handle=None):
    if handle is not None:
        plt.figure(handle)
    else:
        plt.figure()
    return plt

def closeall():
    global FIGHANDLE;  FIGHANDLE = {};
    return plt.close('all')

def imagesc(img, title=None, colormap=None):
    imshow(mat2gray(img), title, colormap=colormap)

def _imshow(img, title=None, colormap=None, do_updateplot=True, figure=None):
    # Short pause needed to show fig
    # See http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive
    if figure is not None:
        fig = plt.figure(figure)
        plt.clf()
    else:
        fig = plt.gcf()
        plt.clf()

    ax = plt.Axes(fig, [0., 0., 1., 1.], frameon=False)
    fig.add_axes(ax)

    plt.axis('off')
    ax.set_axis_off()
    for a in plt.gcf().axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    if colormap in ['grey', 'gray', 'grayscale', 'greyscale']:
        cmap=cm.Greys_r
        imh = plt.imshow(img, animated=True, interpolation='nearest', cmap=cmap)
    elif colormap is not None:
        imh = plt.imshow(np.uint8(255.0*img), animated=True, interpolation='nearest', cmap=plt.get_cmap(colormap))
    else:
        imh = plt.imshow(img, animated=True, interpolation='nearest')
    plt.autoscale(tight=True)
    plt.axis('image')
    #plt.gcf().set_tight_layout(True)  # gives warning
    if title is not None:
        plt.title(title)

    if do_updateplot:
        #plt.pause(0.00001)
        try:
            plt.gcf().canvas.flush_events()
        except:
            pass
        plt.draw()
        plt.show()

    return imh

def imshow(img, title=None, colormap=None, do_updateplot=True, figure=None):
    global FIGHANDLE
    if figure in FIGHANDLE.keys() and FIGHANDLE[figure].get_size() == img.shape[0:2]:
        # Delete all polygon and text overlays from previous drawing
        FIGHANDLE[figure].set_data(img)
        for c in plt.gca().get_children():
            if 'Text' in c.__repr__() or 'Polygon' in c.__repr__() or 'Circle' in c.__repr__() or 'Line' in c.__repr__() or  'Patch' in c.__repr__():
                try:
                    c.remove()
                except:
                    pass
        if do_updateplot:
            plt.draw()
            plt.show()
    else:
        FIGHANDLE[figure] = _imshow(img, title=title, colormap=colormap, do_updateplot=do_updateplot, figure=figure)
    pause(0.00001)  # flush
    return figure


def pause(sec=0.0001):
    plt.pause(sec)

def tracks(im, bbox, bboxcolor='green', caption=None, captioncolor='red'):
    plt.clf()
    # Short pause needed to show fig
    # See http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive
    plt.pause(0.0001)
    plt.imshow(im)
    plt.autoscale(tight=True)
    plt.axis('image')
    plt.set_cmap('gray')
    #plt.hold(True)

    if caption is None:
        caption = ['det']*len(bbox)
    elif not islist(caption):
        caption = [caption]*len(bbox)

    for (bb, cap) in zip(bbox, caption):
        # (x,y) bounding box is right and down, swap to right and up for plot
        xmin = bb[0]
        ymin = bb[1]
        xmax = bb[0]+bb[2]
        ymax = bb[1]+bb[3]
        plt.axvspan(xmin, xmax, ymin=1-np.float32(float(ymax)/float(im.shape[0])), ymax=1-np.float32(float(ymin)/float(im.shape[0])), edgecolor='g', facecolor='white', linewidth=3, fill=True, alpha=0.5, label='test')

        if cap is not None:
            plt.text(xmin, ymin, cap, bbox=dict(facecolor='white', edgecolor='g',alpha=1), fontsize=10)

    plt.draw()





def imbbox(img, xmin, ymin, xmax, ymax, bboxcaption=None, figure=None, bboxcolor='green', facecolor='white', facealpha=0.5, textcolor='black', textfacecolor='white', do_updateplot=True, do_imshow=True, colormap='gray', fontsize=10, captionoffset=(0,0)):
    """Draw bounding box"""
    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()
    figure = plt.gcf().number

    # Short pause needed to show fig
    # See http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive

    if do_imshow == True:
        #plt.clf()
        imshow(img, colormap=colormap, figure=figure, do_updateplot=False)
        #plt.hold(True)

    # (x,y) bounding box is right and down, swap to right and up for plot
    # clip_on clips anything outside the image
    plt.axvspan(xmin, xmax, ymin=1.0-np.float32(float(ymax)/float(img.shape[0])), ymax=1-np.float32(float(ymin)/float(img.shape[0])), edgecolor=bboxcolor, facecolor=facecolor, linewidth=3, fill=True, alpha=facealpha, label=None, capstyle='round', joinstyle='bevel', clip_on=True)

    if bboxcaption is not None:
        # clip_on clips anything outside the image
        plt.text(xmin+captionoffset[0], ymin+captionoffset[1], bboxcaption, color=textcolor, bbox=dict(facecolor=textfacecolor, edgecolor=textcolor, alpha=1, boxstyle='round'), fontsize=fontsize, clip_on=True)

    # Update plot only for final bbox if displaying a lot
    if do_updateplot == True:
        #plt.pause(0.00001)
        try:
            plt.gcf().canvas.flush_events()
        except:
            pass

        #plt.draw()
        #plt.show()


    pause(0.00001)
    return plt.gcf().number

def imdetection(img, imdetlist, figure=None, bboxcolor='green', colormap=None, do_caption=True, facecolor='white', facealpha=0.5, textcolor='green', textfacecolor='white', captionlist=None, fontsize=10, captionoffset=(0,0)):
    """Show bounding boxes from a list of ImageDetections on the same image, plotted in list order with optional captions """

    # try imdetlist.sort(key=(lambda im: im.attributes['RawDetectionProbability']))
    # try captionlist = [im.attributes['RawDetectionProbability'] for im in imdetlist]
    # color = {'face':'red', 'vehicle':'blue', 'person':'green'}
    # bboxcolor = [color[im.category()] for im in detlist]

    # Empty?
    if len(imdetlist) == 0:
        imshow(img, figure=figure, colormap=colormap, do_updateplot=True)
        return figure

    # Valid detections
    fig = figure
    for (k,im) in enumerate(imdetlist):
        do_imshow = True if k==0 else False  # first image only
        do_updateplot = True if k==(len(imdetlist)-1) else False  # last image only
        if do_caption and captionlist is not None:
            bboxcaption = str(captionlist[k])
        elif do_caption:
            bboxcaption = str(im.category())
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

        fig = imbbox(img=img, xmin=im.bbox.xmin, ymin=im.bbox.ymin, xmax=im.bbox.xmax, ymax=im.bbox.ymax, bboxcaption=bboxcaption, do_imshow=do_imshow, do_updateplot=do_updateplot, figure=fig, colormap=colormap, bboxcolor=bboxcolor_, facecolor=facecolor, facealpha=facealpha, textcolor=textcolor_, textfacecolor=textfacecolor, fontsize=fontsize, captionoffset=captionoffset)
    #plt.hold(False)
    pause(0.00001)
    return fig


def precision_recall(y_precision, x_recall, title=None):
    plt.clf()
    plt.plot(x_recall, y_precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    if title is not None:
        plt.title(title)
    plt.legend(loc="lower left")
    plt.draw()



def imframe(img, fr, color='b', markersize=10, label=None, figure=None):
    if figure is not None:
        fig = plt.figure(figure)
        #plt.hold(True)
    else:
        fig = plt.figure()
        #plt.hold(True)

    figure = plt.gcf().number

    plt.pause(0.00001)
    #fig = plt.figure(np.floor(np.random.rand()*50))
    plt.clf()

    #height = float(im.shape[0])
    #width = float(im.shape[1])
    #fig = plt.gcf()
    #fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.], frameon=False)
    #ax.set_axis_off()
    fig.add_axes(ax)


    if img is not None:
        b = ax.imshow(img, cmap=cm.gray)
    #plt.autoscale(tight=True)
    #plt.axis('image')
    #plt.axis('off')


    plt.axis('off')
    ax.set_axis_off()
    for a in plt.gcf().axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    plt.autoscale(tight=True)
    #plt.hold(True)
    plt.plot(fr[:,0],fr[:,1],'%s.' % color, markersize=markersize, axes=ax)
    #plt.hold(False)
    #ax = plt.axes([0,0,1,1])
    #ax.axis('off')
    if label is not None:
        for ((x,y),lbl) in zip(fr, label):
            #plt.text(x, y, lbl, bbox=dict(facecolor='white', edgecolor='g',alpha=1))
            ax.text(x, y, lbl, color='white')

    #plt.draw()

    pause(0.00001)
    return plt


def frame(fr, im=None, color='b.', markersize=10, figure=None, caption=None):
    if figure is not None:
        fig = plt.figure(figure)
        #plt.hold(True)
    else:
        fig = plt.figure()
        plt.clf()
        #plt.hold(True)

    ax = plt.axes([0,0,1,1])
    #b = plt.imshow(im, cmap=cm.gray)
    #plt.hold(True)
    plt.plot(fr[:,0],fr[:,1],color)
    #plt.hold(False)
    plt.axis('off');
    plt.draw()
    #return plt


def colorlist():
    """Return a list of named colors"""
    colorlist = [str(name) for (name, hex) in matplotlib.colors.cnames.items()]
    primarycolorlist = ['green','blue','red','cyan','orange', 'yellow','violet']
    return primarycolorlist + [c for c in colorlist if c not in primarycolorlist]
