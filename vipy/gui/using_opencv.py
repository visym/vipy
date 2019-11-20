import numpy as np
import cv2
import multiprocessing
import sys
from time import sleep
import signal
import matplotlib.cm as cm 
from bobo.util import temppng


# FIXME: rewrite me to remove broken multiprocessing event queues
raise ValueError('OpenCV backend not supported')


WINDOWSTATE = {'focus':None, 'windows':{}}

def _num_windows():
    return len(WINDOWSTATE['windows'])

def _handle(h=None):
    if h is None and WINDOWSTATE['focus'] is not None:
        h = WINDOWSTATE['focus']
    elif h is None: 
        h = 'Figure %d' % int(_num_windows()+1)
    elif type(h) is int:
        h = 'Figure %d' % int(h)
    elif type(h) is not str:
        raise ValueError('window handle must be a string')
    return h

def _eventloop(handle, drawqueue):
    def _sigint_handler(signum, frame):
        cv2.destroyWindow(handle)        
        sys.exit()
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)    
    
    cv2.namedWindow(handle)
    cv2.resizeWindow(handle, 320, 240)
    while cv2.waitKey(10) != 27:  # ESC
        if not drawqueue.empty():
            cv2.imshow(handle, drawqueue.get())
    cv2.destroyWindow(handle)

def _flip(im, handle=None):
    if im.ndim == 2:
        imbgr = cv2.cvtColor(im, cv2.cv.CV_GRAY2BGR)
    else:
        imbgr = im
        
    (x, drawprocess, drawqueue) = WINDOWSTATE['windows'][_handle(handle)]
    while drawqueue.empty() == False:
        sleep(0.01)  
    drawqueue.put(imbgr)
    WINDOWSTATE['windows'][_handle(handle)][0] = imbgr.copy()
    
def figure(handle=None):
    global WINDOWSTATE
    handle = _handle(handle)
    WINDOWSTATE['focus'] = handle
    
    if handle not in WINDOWSTATE['windows'].keys() or WINDOWSTATE['windows'][handle][1].is_alive() == False:
        drawqueue = multiprocessing.Queue()
        drawprocess = multiprocessing.Process(target=_eventloop, args=(handle, drawqueue))
        drawprocess.start()        
        WINDOWSTATE['windows'][handle] = [np.zeros((240,320)), drawprocess, drawqueue] 
    return handle

def close(handle=None):
    global WINDOWSTATE
    handle = _handle(handle)
    if handle in WINDOWSTATE['windows'].keys():
        (im, drawprocess, drawqueue) = WINDOWSTATE['windows'][handle]
        drawprocess.terminate()
        del WINDOWSTATE['windows'][handle]
    
def closeall():
    for h in WINDOWSTATE['windows'].iterkeys():
        (im, drawprocess, drawqueue) = WINDOWSTATE['windows'][h]
        drawprocess.terminate()
    WINDOWSTATE['windows'] = {}

def imshow(im, handle=None):    
    _flip(im, figure(handle))
    
def imbbox(im, xmin, xmax, ymin, ymax, bboxcaption=None):
    pass

def rectangle(bbox, color='green', caption=None, filled=False, linewidth=1, flip=False):
    global WINDOWSTATE
    im = WINDOWSTATE['windows'][WINDOWSTATE['focus']][0]
    cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), cv2.cv.Scalar(color[0],color[1],color[2])), 
    WINDOWSTATE['windows'][WINDOWSTATE['focus']][0] = im
    if flip:
        imshow(im)

def ellipse(bbox, color='green', caption=None, filled=False, linewidth=1):
    pass


def circle(center, radius, color, caption, filled=False, linewidth=1):
    global WINDOWSTATE
    im = WINDOWSTATE['windows'][WINDOWSTATE['focus']][0]
    cv2.circle(im, center, radius, cv2.cv.Scalar(color[0],color[1],color[2]))
    WINDOWSTATE['windows'][WINDOWSTATE['focus']][0] = im
    imshow(im)

def _color(c):
    if type(c) is str:
        if c == 'green':
            c = cv2.cv.Scalar(0,255,0)
        elif c == 'red':
            c = cv2.cv.Scalar(0,0,255)            
        elif c == 'blue':
            c = cv2.cv.Scalar(255,0,0)            
        elif c == 'yellow':
            c = cv2.cv.Scalar(0,255,255)            
        else:
            print 'undefined color %s' % c
            c = cv2.cv.Scalar(0,0,0)            
    else:    
        pass
    return c

def _im2bgr(im):
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.cv.CV_GRAY2BGR)
    return im    

def frame(fr, im, color, caption):
    im = _im2bgr(im)
    c = _color(color)
    for xysr in fr:
        bbox = (xysr[0], xysr[1], 10, 10)
        s = xysr[2]
        th = xysr[3]
        R = np.mat([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        x = np.mat([ [-10, +10, +10, -10], [-10, -10, +10, +10] ])
        y = R*((s/10)*x)
        v = [(int(xysr[0]+y[0,0]),int(xysr[1]+y[1,0])), (int(xysr[0]+y[0,1]),int(xysr[1]+y[1,1])), (int(xysr[0]+y[0,2]),int(xysr[1]+y[1,2])), (int(xysr[0]+y[0,3]),int(xysr[1]+y[1,3]))]
        cv2.line(im, v[0], v[1], color=c, thickness=1, lineType=cv2.CV_AA) 
        cv2.line(im, v[1], v[2], color=c, thickness=1, lineType=cv2.CV_AA) 
        cv2.line(im, v[2], v[3], color=c, thickness=1, lineType=cv2.CV_AA) 
        cv2.line(im, v[3], v[0], color=c, thickness=1, lineType=cv2.CV_AA)                                     
    _flip(im, figure('frame'))

def tracks(im, bbox, bboxcolor='green', caption=None, captioncolor='red'):
    im = _im2bgr(im)
    if type(bboxcolor) not in (list, tuple):
        bboxcolorlist = [_color(bboxcolor)]*len(bbox)
    else:
        bboxcolorlist = [_color(c) for c in bboxcolor]
    if type(captioncolor) not in (list, tuple):
        captioncolorlist = [_color(captioncolor)]*len(bbox)
    else:
        captioncolorlist = [_color(c) for c in captioncolor]
    for (bb,bbcolor,cap,capcolor) in zip(bbox, bboxcolorlist, caption, captioncolorlist):
        cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[0])+int(bb[2]), int(bb[1])+int(bb[3])), color=bbcolor), 
        if cap is not None:
            cv2.putText(im, cap, (int(bb[0]),int(bb[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=capcolor, thickness=1, lineType=cv2.CV_AA)
    _flip(im, figure('tracks')) # update the display                
    
def scatter(fr, im, color):
    im = _im2bgr(im)
    #c = _color(color)
    n_frame = fr.shape[1]
    cmap = cm.get_cmap('jet', n_frame) 
    rgb = np.uint8(255*cmap(np.arange(n_frame)))
    for (i,xysr) in enumerate(fr.transpose()):
        cv2.circle(im, (int(xysr[0]),int(xysr[1])), radius=1, color=cv2.cv.Scalar(int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])))
    _flip(im, figure('scatter')) # update the display                

def text(ij, caption, color):
    global WINDOWSTATE
    im = WINDOWSTATE['windows'][WINDOWSTATE['focus']][0]
    cv2.putText(im, caption, ij, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=_color(color), thickness=1, lineType=cv2.CV_AA)
    # getTextSize() for computing ij offsets
    WINDOWSTATE['windows'][WINDOWSTATE['focus']][0] = im
    imshow(im)
    
def savefig(handle=None, filename=None):
    handle = _handle(handle)
    if filename is None:
        filename = temppng()
    cv2.imwrite(filename, WINDOWSTATE['windows'][WINDOWSTATE['focus']][0])
    return filename
    
def opticalflow(im, uv_flow):
    # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    (mag, ang) = cv2.cartToPolar(uv_flow[...,0], uv_flow[...,1])
    hsv = np.zeros((im.shape[0], im.shape[1], 3))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    imbgr = cv2.cvtColor(np.float32(hsv), cv2.COLOR_HSV2BGR)
    imshow(imbgr)
    # FIXME: this is broken

def sparseflow(im, ijuv_flow, maxflow=16):
    im = _im2bgr(im)
    (mag, ang) = cv2.cartToPolar(ijuv_flow[:,2], ijuv_flow[:,3])
    hsv = np.zeros( (1, ijuv_flow.shape[0], 3) )
    hsv[:,:,1:2] = 255
    hsv[:,:,0:1] = ang*180/np.pi/2
    hsv[:,:,2:3] = cv2.normalize(np.clip(mag,0,maxflow), None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)
    for (k, ijuv) in enumerate(ijuv_flow):
        c = cv2.cv.Scalar( int(bgr[0,k,0]), int(bgr[0,k,1]), int(bgr[0,k,2]))
        cv2.circle(im, (int(ijuv[0]),int(ijuv[1])), radius=2, color=c, thickness=-1)
    _flip(im, figure('sparse optical flow')) # update the display 

    
def disparity(disp, maxdisparity=None):
    if maxdisparity is None:
        maxdisparity = np.percentile(disp[np.nonzero(disp)], 95)  # robust colormap
    imdisp = np.mat((255*np.clip(np.float32(disp), 0, maxdisparity)) / maxdisparity, dtype=np.uint8)
    imdisp = cv2.applyColorMap(imdisp, cv2.COLORMAP_JET)
    _flip(imdisp, figure('disparity')) # update the display 
    
def impolygon(im, poly, color='green'):
    im = _im2bgr(im)
    c = _color(color)
    for p in poly:
        cv2.polylines(im, p, isClosed=True, color=c, thickness=1, lineType=cv2.CV_AA)
    _flip(im, figure('polygon')) # update the display                
    
