import os
import matplotlib

# Matplotlib Backend
# - Specify this environment variable VIPY_BACKEND
# - Headless operation: 'Agg'
# - Linux (X11-forwarded): 'TkAgg', 'GTK3Cairo', 'QtAgg'
# - Valid backends: https://matplotlib.org/stable/users/explain/backends.html
#
# Installing TkAgg
# sh> export VIPY_BACKEND='TkAgg'
# sh> sudo apt install python3.12-tk  # replace with your python version
# sh> pip install tk   # virtualenv
# >>> vipy.image.owl().show()

# Installing QtAgg
# sh> export VIPY_BACKEND='QtAgg'
# sh> sudo apt install python3-pyqt5   # replace with your python version
# sh> pip install pyqt5 opencv-python-headless 
# >>> vipy.image.owl().show()

if 'VIPY_BACKEND' in os.environ and 'DISPLAY' in os.environ:
    matplotlib.use(os.environ['VIPY_BACKEND'])  # 'Agg' is required unless there is a DISPLAY set

    
import importlib
BACKEND = importlib.import_module('vipy.gui.using_matplotlib')

try:
    import matplotlib.style as mplstyle
    mplstyle.use('fast')
except:
    pass


def figure(fignum=None):
    return BACKEND.figure(fignum)


def close(fignum=None):
    return BACKEND.close(fignum)


def closeall():
    return BACKEND.closeall()


def show(fignum):
    return BACKEND.show(fignum)


def noshow(fignum):
    return BACKEND.noshow(fignum)


def imshow(im, fignum=None, nowindow=False, timestamp=None, timestampcolor='black', timestampfacecolor=None, flush=False):
    """Show an image in the provided figure number"""
    if nowindow:
        noshow(fignum)
    h = BACKEND.imshow(im, fignum=fignum)
    if timestamp is not None:
        text(str(timestamp), 4, 15, fontsize=10, alpha=0.8 if timestampfacecolor is not None else 1.0, textfacealpha=0.6, facealpha=0.6, textcolor=timestampcolor, textfacecolor=timestampfacecolor)
    if not nowindow:
        show(fignum)
        if flush:
            BACKEND.imflush()
    return h


def imbbox(img, xmin, ymin, xmax, ymax, bboxcaption=None, fignum=None, nowindow=False):
    if nowindow:
        noshow(fignum)
    h = BACKEND.imshow(img, fignum=fignum)
    h = BACKEND.boundingbox(img, xmin, ymin, xmax, ymax, bboxcaption, fignum=h)
    if not nowindow:
        show(fignum)
        BACKEND.flush()
    return h


def imdetection(img, detlist, fignum=None, bboxcolor='green', facecolor='white', facealpha=0.5, do_caption=True, fontsize=10, textcolor='green', textfacecolor='white', textfacealpha=1.0, captionoffset=(0,0), nowindow=False, timestamp=None, timestampcolor='black', timestampfacecolor=None, timestampoffset=(0,0)):
    """Show a list of vipy.object.Detections overlayed on img.  Image must be RGB"""

    if nowindow:
        noshow(fignum)
    h = BACKEND.imdetection(img, detlist, fignum=fignum, bboxcolor=bboxcolor, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset, textfacecolor=textfacecolor, textfacealpha=textfacealpha)
    if timestamp is not None:
        text(str(timestamp), 10+timestampoffset[0], 21+timestampoffset[1], fontsize=fontsize, textfacealpha=0.6, facealpha=0.6, alpha=0.8 if timestampfacecolor is not None else 1.0, textcolor=timestampcolor, textfacecolor=timestampfacecolor, pad=0.75)       
    if not nowindow:
        show(fignum)
        BACKEND.flush() if len(detlist)>0 else BACKEND.imflush()               
    return h

def imkeypoints(img, kplist, fignum=None, bordercolor='green', facecolor='white', facealpha=0.5, do_caption=True, fontsize=10, textcolor='green', textfacecolor='white', textfacealpha=1.0, captionoffset=(0,0), nowindow=False, timestamp=None, timestampcolor='black', timestampfacecolor=None, timestampoffset=(0,0)):

    if nowindow:
        noshow(fignum)
    h = BACKEND.imkeypoints(img, kplist, fignum=fignum, bordercolor=bordercolor, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset, textfacecolor=textfacecolor, textfacealpha=textfacealpha)
    if timestamp is not None:
        text(str(timestamp), 10+timestampoffset[0], 21+timestampoffset[1], fontsize=fontsize, textfacealpha=0.6, facealpha=0.6, alpha=0.8 if timestampfacecolor is not None else 1.0, textcolor=timestampcolor, textfacecolor=timestampfacecolor, pad=0.75)       
    if not nowindow:
        show(fignum)
        BACKEND.flush() if len(kplist)>0 else BACKEND.imflush()               
    return h

def imobjects(img, objlist, fignum=None, bordercolor='green', facecolor='white', facealpha=0.5, do_caption=True, fontsize=10, textcolor='green', textfacecolor='white', textfacealpha=1.0, captionoffset=(0,0), nowindow=False, timestamp=None, timestampcolor='black', timestampfacecolor=None, timestampoffset=(0,0), timestamp_alpha=0.6, kp_alpha=0.7):
    if nowindow:
        noshow(fignum)
    h = BACKEND.imobjects(img, objlist, fignum=fignum, bordercolor=bordercolor, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset, textfacecolor=textfacecolor, textfacealpha=textfacealpha, kp_alpha=kp_alpha)
    if timestamp is not None:
        text(str(timestamp), 10+timestampoffset[0], 21+timestampoffset[1], fontsize=fontsize, textfacealpha=0.6, facealpha=0.6, alpha=timestamp_alpha, textcolor=timestampcolor, textfacecolor=timestampfacecolor, pad=0.75)       
    if not nowindow:
        show(fignum)
        BACKEND.flush() if len(objlist)>0 else BACKEND.imflush()               
    return h


def impoints(img, objlist, fignum=None, bboxcolor='green', facecolor='white', facealpha=0.5, do_caption=True, fontsize=10, textcolor='green', textfacecolor='white', textfacealpha=1.0, captionoffset=(0,0), nowindow=False, timestamp=None, timestampcolor='black', timestampfacecolor=None, timestampoffset=(0,0)):
    pass

def frame(fmr, im=None, color='b.', caption=False, markersize=10, fignum=1):
    return BACKEND.frame(fr, im=im, color=color, caption=caption, markersize=markersize, fignum=fignum)


def imframe(img, fr, color='b', markersize=20, label=None, fignum=None):
    return BACKEND.imframe(img=img, fr=fr, color=color, markersize=markersize, label=label, fignum=fignum)


def savefig(filename=None, fignum=None, pad_inches=0, bbox_inches='tight', dpi=None, format=None):
    return BACKEND.savefig(filename, fignum, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches, format=format)


def colorlist(theme=None):
    assert theme in [None, 'dark', 'light'], "invalid color theme '%s' - choose from [None, 'dark', 'light']" % theme
    return BACKEND.colorlist(theme=theme)


def text(caption, xmin, ymin, fignum=None, textcolor='black', textfacecolor=None, textfacealpha=1.0, fontsize=10, linewidth=3, facecolor='white', facealpha=0.5, alpha=1.0, pad=0.5):
    return BACKEND.text(caption, xmin, ymin, fignum, textcolor, textfacecolor, textfacealpha, fontsize, linewidth, facecolor, facealpha, alpha, pad=pad)
    

def array(img, mindim=512, figure=1):
    """Fast visualization of a numpy array img
        
    ```python
    vipy.show.array(np.random.rand(16,16,3))
    ```

    """
    from vipy.util import isnumpy
    from vipy.image import Image
    
    assert isnumpy(img)
    return Image(array=np.array(img).astype(np.float32), colorspace='float').mindim(mindim, interp='nearest').show(figure=figure)
