import os
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np
from vipy.util import islist, temppng
import sys
from vipy.globals import log
from vipy.version import Version
import numpy as np


FIGHANDLE = {}
matplotlib.rcParams['toolbar'] = 'None'
matplotlib_version_at_least_3p3 = Version.from_string(matplotlib.__version__) >= '3.3'


PRIMARY_COLORLIST = ['green','blue','red','cyan','orange','yellow','violet','white'] + ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
COLORLIST = PRIMARY_COLORLIST + [str(name) for (name, hex) in mcolors.cnames.items() if str(name) not in PRIMARY_COLORLIST]  # use primary colors first
COLORLIST_LUMINANCE = [(np.array(mcolors.to_rgb(name)) * np.array([0.2126, 0.7152, 0.0722])).sum() for name in COLORLIST]
DARK_COLORLIST = tuple(c for (c,l) in zip(COLORLIST, COLORLIST_LUMINANCE) if l>=0.5)  # colors that show well on dark backgrounds
LIGHT_COLORLIST = tuple(c for (c,l) in zip(COLORLIST, COLORLIST_LUMINANCE) if l<=0.5) # colors that show well on light backgrounds


# Optional latex strings in captions
try:
    import vipy.globals
    if vipy.globals.GLOBAL['LATEX'] is not None:
        from distutils.spawn import find_executable
        if not find_executable('latex'):
            raise
        matplotlib.rc('text', usetex=True)  # requires latex installed
except:
    pass  # ignored if latex is not installed or not wanted


def escape_to_exit(event):
    if event.key == 'escape' or event.key == 'q' or event.key == 'ctrl+c':
        import vipy.globals            
        vipy.globals._user_hit_escape(True)
    
def flush():
    if matplotlib.get_backend().lower() == 'macosx':
        plt.draw()  # YUCK: to flush buffer on video play, will be slow
    plt.pause(0.001)

    
def imflush():
    if matplotlib.get_backend().lower() == 'macosx':
        plt.draw()  # YUCK: to flush buffer on video play, will be slow
    plt.pause(0.001)

    # this is necessary for imshow only, maybe to trigger remove() of child?
    # However, this breaks jupyter notebook image show if we use gca()
    [a.annotate('', (0,0)) for i in plt.get_fignums() for a in plt.figure(i).axes]
    
    #plt.gca().annotate('', (0,0))  

    
def show(fignum):
    fig = plt.figure(fignum) 
    fig.canvas.draw()    
    plt.ion()
    plt.show()

    
def noshow(fignum):
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
        plt.close(fignum)
        FIGHANDLE.pop(fignum, None)
        return None
    else:
        return None


def closeall():
    global FIGHANDLE
    FIGHANDLE = {}
    return plt.close('all')


def _imshow_tight(img, fignum=None, keypress=True):
    """Helper function to show an image in a figure window"""
    dpi = 100.0
    fig = plt.figure(fignum, dpi=dpi, figsize=(img.shape[1] / dpi, img.shape[0] / dpi)) if not plt.fignum_exists(fignum) else plt.figure(fignum)
    plt.clf()

    # Tight axes
    ax = plt.Axes(fig, [0., 0., 1.0, 1.0], frameon=False)
    fig.add_axes(ax)
    for a in plt.gcf().axes:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    imh = plt.imshow(img, animated=True, interpolation='nearest', aspect='equal')

    if keypress:
        fig.canvas.mpl_connect('key_press_event', escape_to_exit)
        import vipy.globals    
        vipy.globals._user_hit_escape(False)

    return (fig.number, imh)


def imshow(img, fignum=None):
    """Show an image in a figure window (optionally visible), reuse previous figure if it is the same shape"""
    global FIGHANDLE

    if fignum in plt.get_fignums() and fignum in FIGHANDLE and FIGHANDLE[fignum].get_size() == img.shape[0:2]:
        # Do not delete and recreate the figure, just change the pixels 
        FIGHANDLE[fignum].set_data(img)

        # Delete all polygon and text overlays from previous drawing so that they can be overwritten on current frame
        for c in plt.gca().get_children():
            desc = c.__repr__()
            if 'Text' in desc or 'Polygon' in desc or 'Circle' in desc or 'Line' in desc or 'Patch' in desc or 'PathCollection' in desc:
                try:
                    c.remove()
                except:
                    # MacOSX fails with the error: NotImplementedError: cannot remove artist
                    # fallback on closing figure completely
                    (fignum, imh) = _imshow_tight(img, fignum=fignum) 
                    FIGHANDLE[fignum] = imh
                    return fignum

    else:
        # Jupyter notebook does not respect fignum.  It is always one when inspecting plt.get_fignums()
        if fignum in plt.get_fignums() and fignum in FIGHANDLE:
            close(fignum)
            #pass  # don't close unless user requests, this is faster, but window size does not change with image resolution
        (fignum, imh) = _imshow_tight(img, fignum=fignum)
        FIGHANDLE[fignum] = imh
    return fignum


def text(caption, xmin, ymin, fignum=None, textcolor='black', textfacecolor='white', textfacealpha=1.0, fontsize=10, linewidth=3, facecolor='white', facealpha=0.5, alpha=1.0, pad=0.5, captionoffset=0):
    plt.figure(fignum) if fignum is not None else plt.gcf()
    lw = linewidth  # pull in the boxes by linewidth so that they do not overhang the figure

    newlines = caption.count('\n')
    
    # MatplotlibDeprecationWarning: The 's' parameter of annotate() has been renamed 'text' since Matplotlib 3.3            
    handle = plt.annotate(**{'text' if matplotlib_version_at_least_3p3 else 's': caption},
                          alpha=alpha, xy=(xmin,ymin), xytext=(xmin, ymin+captionoffset), xycoords='data',
                          color=textcolor, bbox=None if textfacecolor is None else dict(facecolor=textfacecolor, edgecolor=None, alpha=textfacealpha, boxstyle='square', pad=pad),
                          fontsize=fontsize, clip_on=True)

    return fignum

def boundingbox(img, xmin, ymin, xmax, ymax, bboxcaption=None, fignum=None, bboxcolor='green', facecolor='white', facealpha=0.5, textcolor='black', textfacecolor='white', textfacealpha=1.0, fontsize=10, captionoffset=(0,0), linewidth=3):
    """Draw a captioned bounding box on a previously shown image"""
    plt.figure(fignum)
    ax = plt.gca()
    lw = linewidth  # pull in the boxes by linewidth so that they do not overhang the figure
    (H,W) = (img.shape[0], img.shape[1])
    
    ymin_frac = 1.0 - (min(ymax, H - lw/2) / float(H))
    ymax_frac = 1.0 - (max(ymin, lw/2) / float(H))
    
    y0, y1 = ax.get_ylim()                 # data‐coordinates of bottom and top of the axes
    rect = Rectangle((xmin, ymin), xmax - xmin,  (ymax_frac - ymin_frac) * abs(y1 - y0), 
                 linewidth=lw,
                 edgecolor=(*mcolors.to_rgb(bboxcolor),0.4),
                 facecolor=(*mcolors.to_rgb(facecolor),facealpha),
                 #alpha=facealpha,
                 transform=ax.transData, capstyle='round', joinstyle='bevel', clip_on=True)
    ax.add_patch(rect)

    # Text string
    if bboxcaption is not None:
        # a) Compute x‐fraction: (xmin − x0)/(x1 − x0)
        x0, x1 = ax.get_xlim()
        xmin_frac = ((xmin - x0) / (x1 - x0)) #+ (captionoffset[0]/H)
        
        # b) y1_frac is already the top of the box in axes‐fraction; add vertical offset
        ymin_frac_text = ymin_frac #+ (captionoffset[1]/W)
        ymax_frac = 1.0 - ((max(ymin, lw/2)) / float(H)) 
        
        newlines = bboxcaption.count('\n')

        captionoffset = captionoffset if ymin > 20 else (captionoffset[0], captionoffset[1]+(20*(1)))  # move down a bit if near top of image, shift once per newline in caption

        # MatplotlibDeprecationWarning: The 's' parameter of annotate() has been renamed 'text' since Matplotlib 3.3   
        plt.annotate(**{'text' if matplotlib_version_at_least_3p3 else 's': bboxcaption},
                     xy=(xmin_frac, ymax_frac),
                     xytext=(captionoffset[0], -captionoffset[1]),
                     textcoords='offset pixels',
                     xycoords='axes fraction',
                     color=textcolor,
                     fontsize=fontsize,
                     bbox=dict(
                         facecolor=textfacecolor,
                         edgecolor=textcolor,
                         alpha=textfacealpha,
                         boxstyle='square',
                     ),
                     horizontalalignment='left',
                     verticalalignment='top',
                     clip_on=True
                     )

    return fignum


def imdetection(img, detlist, fignum=None, bboxcolor='green', do_caption=True, facecolor='white', facealpha=0.5, textcolor='green', textfacecolor='white', textfacealpha=1.0, fontsize=10, captionoffset=(0,0)):
    """Show bounding boxes from a list of vipy.object.Detections on the same image, plotted in list order with optional captions """

    # Create image
    #fignum = imshow(img, fignum=fignum)  # this can fail on MacOSX, go the slow route instead
    fignum = _imshow_tight(img, fignum=fignum)     

    # A better way? https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
    
    # Valid detections
    for (k,det) in enumerate(detlist):
        if do_caption and det.category() is not None and len(det.category()) != 0 and (det.category()[0:2] != '__'):  # prepending category with '__' will disable caption
            bboxcaption = det.category()
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
                    fignum=fignum, bboxcolor=bboxcolor_, facecolor=facecolor, facealpha=facealpha, textcolor=textcolor_, textfacecolor=textfacecolor, fontsize=fontsize, captionoffset=captionoffset, textfacealpha=textfacealpha)

    return fignum


def imkeypoints(img, kplist, fignum=None, bordercolor='green', do_caption=True, facecolor='white', facealpha=0.5, textcolor='green', textfacecolor='white', textfacealpha=1.0, fontsize=10, captionoffset=(0,0)):
    """Show `vipy.object.Keypoint2d` on the same image, plotted in list order with optional captions """

    # Create image
    fignum = imshow(img, fignum=fignum) 

    # A better way? https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
    
    # Valid keypoints
    (x,y,size,color,caption,captioncolor) = ([],[],[],[],[],[])
    for (k,p) in enumerate(kplist):
        x.append(p.x)
        y.append(p.y)
        size.append(max(p.r, 1)**2)  # size is pts squared
        
        if do_caption and p.category() is not None and len(p.category()) != 0 and (p.category()[0:2] != '__'):  # prepending category with '__' will disable caption
            caption.append(p.category())
        else:
            caption.append(None)

        if islist(bordercolor):
            color.append(mcolors.to_hex(bordercolor[k]) + 'BB')  # with alpha
        else:
            color.append(mcolors.to_hex(bordercolor) + 'BB')  # with alpha
            
        if islist(textcolor):
            captioncolor.append(textcolor[k])
        else:
            captioncolor.append(textcolor)

    plt.figure(fignum)
    plt.scatter(x, y, c=color, s=size)
    return fignum


def imobjects(img, objlist, fignum=None, bordercolor='green', do_caption=True, facecolor='white', facealpha=0.5, textcolor='green', textfacecolor='white', textfacealpha=1.0, fontsize=10, captionoffset=(0,0), kp_alpha=0.8):
    """Show `vipy.object.Keypoint2d` on the same image, plotted in list order with optional captions """

    # Create image
    fignum = imshow(img, fignum=fignum) 
    plt.gca().set_autoscale_on(False)


    # A better way? https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html

    # Valid detections
    detlist = [p for p in objlist if isinstance(p, vipy.object.Detection)]
    if len(detlist) > 0:
        for (k,det) in enumerate(detlist):
            if do_caption and det.category() is not None and len(det.category()) != 0 and (det.category()[0:2] != '__'):  # prepending category with '__' will disable caption
                bboxcaption = det.category()
            else:
                bboxcaption = None

            if islist(bordercolor):
                bboxcolor_ = bordercolor[k]
            else:
                bboxcolor_ = bordercolor

            if islist(textcolor):
                textcolor_ = textcolor[k]
            else:
                textcolor_ = textcolor

            boundingbox(img, xmin=det.xmin(), ymin=det.ymin(), xmax=det.xmax(), ymax=det.ymax(), bboxcaption=bboxcaption,
                        fignum=fignum, bboxcolor=bboxcolor_, facecolor=bboxcolor_, facealpha=facealpha, textcolor=textcolor_, textfacecolor=textfacecolor, fontsize=fontsize, captionoffset=captionoffset, textfacealpha=textfacealpha)
    
    # Valid keypoints
    kplist = [p for p in objlist if isinstance(p, vipy.object.Keypoint2d)]    
    plt.gca().set_autoscale_on(False)

    if len(kplist) > 0:
        (x,y,size,colors,r,patches) = ([],[],[],[],[],[])
        for (k,p) in enumerate(kplist):
            x.append(p.x)
            y.append(p.y)
            r.append(p.r)
            
            c = mcolors.to_hex(bordercolor[k] if islist(bordercolor) else bordercolor)
            colors.append(c + format(int(255*kp_alpha), '02X'))  # with alpha
            patches.append(Circle((p.x, p.y), p.r))

        plt.gca().add_collection(PatchCollection(patches, facecolors=colors, edgecolors='silver', clip_on=True, linewidths=0.5))  # scale radius with window size
    return fignum


def imframe(img, fr, color='b', markersize=10, label=None, figure=None):
    """Show a scatterplot of fr=[[x1,y1],[x2,y2]...] 2D points overlayed on an image, all the same color"""
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
    """Show a scatterplot of fr=[[x1,y1],[x2,y2]...] 2D points, all the same color"""
    if figure is not None:
        plt.figure(figure)
    else:
        plt.figure()
        plt.clf()

    plt.axes([0,0,1,1])
    plt.plot(fr[:,0],fr[:,1],color)
    plt.axis('off')
    plt.draw()


def colorlist(theme=None):
    """Return a list of named colors that are higher contrast with a white background if light_mode, else named colors that are higher contrast with a dark background if dark_mode"""        
    return DARK_COLORLIST if theme=='dark' else (LIGHT_COLORLIST if theme=='light' else COLORLIST)


def edit():
    import matplotlib.pyplot as plt 
    from matplotlib.widgets import Slider, Button, RadioButtons   
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    ax2 = plt.axes([0.1, 0.325, 0.5, 0.05])

    ax2 = plt.axes([0, 0, 0.5, 0.05])
    ax2.patch.set_alpha(0.5) 
    Slider(ax2, 'Reset2', facecolor='red', alpha=0.5, valmin=0, valmax=10)
    ax2.patch.alpha = 0.5 

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print ('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print ('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()


class DraggableRectangle(object):
    def __init__(self, rect):
        self.rect = rect
        self.press = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        log.info('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        #log.info('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        self.rect.figure.canvas.draw()


    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

        
    #a = Annotate()
    #plt.show()    

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #rects = ax.bar(range(10), 20*np.random.rand(10))
    #drs = []
    #for rect in rects:
    #    dr = DraggableRectangle(rect)
    #    dr.connect()
    #    drs.append(dr)
    #
    #plt.show()


class DraggableRectangleFast(object):
    lock = None  # only one can be animated at a time
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return
        if DraggableRectangle.lock is not None: return
        contains, attrd = self.rect.contains(event)
        if not contains: return
        log.info('event contains', self.rect.xy)
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata
        DraggableRectangle.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        self.rect.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.rect)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if DraggableRectangle.lock is not self:
            return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        canvas = self.rect.figure.canvas
        axes = self.rect.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.rect)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableRectangle.lock is not self:
            return

        self.press = None
        DraggableRectangle.lock = None

        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        # redraw the full figure
        self.rect.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)    
