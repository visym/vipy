import importlib

BACKEND = importlib.import_module('vipy.gui.using_matplotlib')
#BACKEND = importlib.import_module('bobo.gui.using_opencv')  # segfaults with multiprocessing.  why?

def backend(using='pygame'):
    global BACKEND
    BACKEND = importlib.import_module('bobo.gui.using_%s' % using)
    if using == 'pygame':
        figure()  # need figure before opencv camera otherwise segfault. Why?
    
def using(using_='pygame'):
    backend(using_)

def pause(sec=1):
    BACKEND.pause(sec)
    
def figure(handle=None):
    return BACKEND.figure(handle)

def close(title=None):
    BACKEND.close(title)

def closeall():
    return BACKEND.closeall()
    
#def fullscreen():
#    BACKEND.fullscreen()
    
def imshow(im, handle=None, colormap=None, figure=None, do_updateplot=True):    
    return BACKEND.imshow(im, handle, colormap, figure=figure, do_updateplot=do_updateplot)

def imagesc(im, handle=None, colormap=None):    
    return BACKEND.imagesc(im, handle, colormap)
    
def imbbox(im, xmin, ymin, xmax, ymax, bboxcaption=None, colormap=None, figure=None, do_updateplot=True):    
    return BACKEND.imbbox(im, xmin, ymin, xmax, ymax, bboxcaption, colormap=colormap, figure=figure, do_updateplot=do_updateplot)
            
def rectangle(bbox, color='green', caption=None, filled=False, linewidth=1):
    BACKEND.rectangle(bbox, color, caption, filled, linewidth)

def ellipse(bbox, color='green', caption=None, filled=False, linewidth=1):
    BACKEND.ellipse(bbox, color, caption, filled, linewidth)
    
def circle(center, radius, color='green', caption=None, filled=False, linewidth=1):
    BACKEND.circle(center, radius, color, caption, filled, linewidth)
    
def boundingbox(bbox, caption, color='green'):
    rectangle(bbox, caption=caption, filled=False, linewidth=1, color=color)

def tracks(im, bbox, bboxcolor='green', caption=None, captioncolor='red'):
    BACKEND.tracks(im, bbox, bboxcolor, caption, captioncolor)
    
def frame(fr, im=None, color='b.', caption=False, markersize=10, figure=1):
    return BACKEND.frame(fr, im=im, color=color, caption=caption, markersize=markersize, figure=figure)

def imframe(img, fr, color='b', markersize=20, label=None, figure=None):
    return BACKEND.imframe(img=img, fr=fr, color=color, markersize=markersize, label=label, figure=figure)
    
def scatter(fr, im=None, color='green'):
    return BACKEND.scatter(fr, im, color)

def text(ij, caption, color):    
    BACKEND.text(ij, caption, color)

def savefig(filename=None, figure=None, pad_inches=0, bbox_inches='tight', dpi=None):
    return BACKEND.savefig(filename, figure, pad_inches=pad_inches, dpi=dpi, bbox_inches=bbox_inches)

def opticalflow(im, flow):
    return BACKEND.opticalflow(im, flow)

def sparseflow(im, flow):
    return BACKEND.sparseflow(im, flow)

def disparity(disp, maxdisparity=None):
    return BACKEND.disparity(disp, maxdisparity)

def impolygon(im, poly, color='green'):
    return BACKEND.impolygon(im, poly, color)

def imdetection(img, imdetlist, figure=None, bboxcolor='green', facecolor='white', facealpha=0.5, colormap=None, do_caption=True, captionlist=None, fontsize=10, textcolor='green', captionoffset=(0,0)):
    """Show a list of ImageDetections overlayed on img.  Image must be RGB (not BGR!) or grayscale"""
    return BACKEND.imdetection(img, imdetlist, figure=figure, bboxcolor=bboxcolor, colormap=colormap, do_caption=do_caption, facecolor=facecolor, facealpha=facealpha, captionlist=captionlist, fontsize=fontsize, textcolor=textcolor, captionoffset=captionoffset)

def colorlist():
    return BACKEND.colorlist()

