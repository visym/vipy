import os
import matplotlib
if 'VIPY_BACKEND' in os.environ:
    matplotlib.use(os.environ['VIPY_BACKEND'])
    
import importlib
BACKEND = importlib.import_module('vipy.gui.using_matplotlib')


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


def imshow(im, fignum=None, nowindow=False):
    """Show an image in the provided figure number"""
    if nowindow:
        noshow(fignum)
    h = BACKEND.imshow(im, fignum=fignum)
    if not nowindow:
        show(fignum)
        BACKEND.flush()
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


def imdetection(img, detlist, fignum=None, bboxcolor='green', facecolor='white', facealpha=0.5, do_caption=True, fontsize=10, textcolor='green', textfacecolor='white', textfacealpha=1.0, captionoffset=(0,0), nowindow=False):
    """Show a list of vipy.object.Detections overlayed on img.  Image must be RGB"""
    if nowindow:
        noshow(fignum)
    h = BACKEND.imdetection(img, detlist, fignum=fignum, bboxcolor=bboxcolor, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset, textfacecolor=textfacecolor, textfacealpha=textfacealpha)
    if not nowindow:
        show(fignum)
        BACKEND.flush()                
    return h


def frame(fmr, im=None, color='b.', caption=False, markersize=10, fignum=1):
    return BACKEND.frame(fr, im=im, color=color, caption=caption, markersize=markersize, fignum=fignum)


def imframe(img, fr, color='b', markersize=20, label=None, fignum=None):
    return BACKEND.imframe(img=img, fr=fr, color=color, markersize=markersize, label=label, fignum=fignum)


def savefig(filename=None, fignum=None, pad_inches=0, bbox_inches='tight', dpi=None, format=None):
    return BACKEND.savefig(filename, fignum, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches, format=format)


def colorlist():
    return BACKEND.colorlist()

