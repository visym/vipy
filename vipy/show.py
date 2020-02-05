import importlib

BACKEND = importlib.import_module('vipy.gui.using_matplotlib')

def figure(fignum=None):
    return BACKEND.figure(fignum)

def close(fignum=None):
    return BACKEND.close(fignum)

def closeall():
    return BACKEND.closeall()
    
def imshow(im, figure=None):
   """Show an image in the provided figure number"""
   return BACKEND.imshow(im, figure=figure, do_updateplot=True)

def imbbox(im, xmin, ymin, xmax, ymax, bboxcaption=None, figure=None, do_updateplot=True):
    return BACKEND.imbbox(im, xmin, ymin, xmax, ymax, bboxcaption, figure=figure, do_updateplot=do_updateplot)
            
def frame(fmr, im=None, color='b.', caption=False, markersize=10, figure=1):
    return BACKEND.frame(fr, im=im, color=color, caption=caption, markersize=markersize, figure=figure)

def imframe(img, fr, color='b', markersize=20, label=None, figure=None):
    return BACKEND.imframe(img=img, fr=fr, color=color, markersize=markersize, label=label, figure=figure)
    
def savefig(filename=None, figure=None, pad_inches=0, bbox_inches='tight', dpi=None, format=None):
    return BACKEND.savefig(filename, figure, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches, format=format)

def imdetection(img, detlist, figure=None, bboxcolor='green', facecolor='white', facealpha=0.5, do_caption=True, captionlist=None, fontsize=10, textcolor='green', captionoffset=(0,0)):
    """Show a list of vipy.object.Detections overlayed on img.  Image must be RGB (not BGR!) or grayscale"""
    return BACKEND.imdetection(img, detlist, figure=figure, bboxcolor=bboxcolor, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, captionlist=captionlist, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset)

def colorlist():
    return BACKEND.colorlist()

