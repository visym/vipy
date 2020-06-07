from vipy.util import mat2gray, try_import
try_import('cv2', 'opencv-python opencv-contrib-python'); import cv2
import vipy.image
from vipy.math import cartesian_to_polar
import numpy as np
import scipy.interpolate
import vipy.object


class Image(vipy.image.Image):
    """vipy.flow.Image() class"""
    def __init__(self, array):
        (self._minflow, self._maxflow) = (np.min(array), np.max(array))   # normalization coefficients 
        array = mat2gray(np.pad(array, ((0,0),(0,0),(0,1)))).astype(np.float32)  # normalize flow [0,1] three channel float32 for image representation
        super(Image, self).__init__(array=array, colorspace='float')
                
    def __repr__(self):
        return str('<vipy.flow: height=%d, width=%d, minflow=%1.2f, maxflow=%1.2f>' % (self.height(), self.width(), self._minflow, self._maxflow))

    def colorflow(self, minmag=None, maxmag=None):
        """Flow visualization image (HSV: H=flow angle, V=flow magnitude), returns vipy.image.Image()"""
        flow = self.flow()
        (r, t) = cartesian_to_polar(flow[:,:,0], flow[:,:,1])
        hsv = np.zeros( (self.height(), self.width(), 3), dtype=np.uint8)
        hsv[:,:,0] = (((t+np.pi) * (180 / np.pi))*(255.0/360.0))
        hsv[:,:,1] = 255
        hsv[:,:,2] = 255*mat2gray(r, min=minmag, max=maxmag)  
        return vipy.image.Image(array=np.uint8(hsv), colorspace='hsv').rgb()
        
    def warp(self, im):
        """Warp vipy.image.Image() input using computed flow from imprev to imnext updating objects"""
        (H, W) = self.shape()
        flow = -self.flow()        
        flow[:,:,0] += np.arange(W)
        flow[:,:,1] += np.arange(H)[:,np.newaxis]
        return (im.clone()
                  .array( cv2.remap(im.numpy(), flow, None, cv2.INTER_LINEAR) )
                  .objectmap(lambda bb: bb.int().offset(dx=np.mean(self.flow()[bb.ymin():bb.ymax(), bb.xmin():bb.xmax(), 0]),
                                                        dy=np.mean(self.flow()[bb.ymin():bb.ymax(), bb.xmin():bb.xmax(), 1]))))
    
    def _convert(self, to):
        raise ValueError("Colorspace conversion on vipy.flow will result in flow scaling to be incorrect.  Use self.colorflow() instead.")

    def resize(self, cols=None, rows=None, width=None, height=None, interp='bilinear'):
        """Isotropic resize of flow, scaling the flow vectors appropriately"""
        (r, d) = (self.aspectratio(), float(self.mindim()))
        super(Image, self).resize(cols, rows, width, height, interp)
        assert np.isclose(r, self.aspectratio(), atol=1E-2), "Anisotropic resizing not supported for flow images"
        (self._minflow, self._maxflow) = ((self.mindim()/d)*self._minflow, (self.mindim()/d)*self._maxflow)        
        return self

    def rescale(self, scale=1, interp='bilinear'):
        """Isotropic resize of flow, scaling the flow vectors appropriately """        
        d = float(self.mindim())
        super(Image, self).rescale(scale, interp)
        (self._minflow, self._maxflow) = ((self.mindim()/d)*self._minflow, (self.mindim()/d)*self._maxflow)
        return self
    
    def flow(self):
        """Return flow MxNx2 flow array, scaled appropriately as flow vectors"""
        return ((self._maxflow - self._minflow)*self.numpy() + self._minflow)[:,:,0:2]

    def dx(self):
        """Return dx (horizontal) component of flow"""
        return self.flow()[:,:,0]

    def dy(self):
        """Return dy (vertical) component of flow"""        
        return self.flow()[:,:,1]

    def show(self, figure=None, nowindow=False):
        self.colorflow().show(figure, nowindow)
    
    def saveas(self, filename):
        raise ValueError('Use vipy.util.save() for saving vipy.flow images')

    
class Video(vipy.video.Video):
    """vipy.flow.Video() class"""
    def __init__(self, array):
        (self._minflow, self._maxflow) = (np.min(array), np.max(array))
        array = mat2gray(np.pad(array, ((0,0),(0,0),(0,0),(0,1)))).astype(np.float32)  # normalize flow [0,1] three channel float32 for image representation        
        super(Video, self).__init__(array=array, colorspace='float')

    def __repr__(self):
        return str('<vipy.flow: frames=%d, height=%d, width=%d, minflow=%1.2f, maxflow=%1.2f>' % (len(self), self.height(), self.width(), self._minflow, self._maxflow))        

    def __getitem__(self, k):
        return Image(self.flow()[k])
        
    def colorflow(self):
        """Flow visualization video"""
        (minmag, maxmag) = (np.min(self.magnitude()), np.max(self.magnitude()))  # scaled over video
        return vipy.video.Video(array=np.stack([Image(f).colorflow(minmag=minmag, maxmag=maxmag).numpy() for f in self.flow()]), colorspace='rgb')

    def flow(self):
        return ((self._maxflow - self._minflow)*self.numpy() + self._minflow)[:,:,:,0:2]

    def magnitude(self):
        return np.stack([cartesian_to_polar(f[:,:,0], f[:,:,1])[0] for f in self.flow()])
    
    def show(self):
        return self.colorflow().show()

    def play(self):
        return self.show()
    
    def resize(self):
        raise NotImplementedError()

    
class Flow(object):
    def __init__(self):
        self._mindim = 256  # small displacements only

    def __call__(self, imprev=None, imnext=None, dt=1, step=1):
        return self._videoflow(imprev, dt, step) if imnext is None else self._imageflow(imprev, imnext)
    
    def _imageflow(self, imprev, imnext):
        """Default opencv dense flow.  This should be overloaded"""        
        assert isinstance(imprev, vipy.image.Image) and isinstance(imnext, vipy.image.Image)
        imp = imprev.clone().mindim(self._mindim).luminance() if imprev.channels() != 1 else imprev.clone().mindim(self._mindim)
        imn = imnext.clone().mindim(self._mindim).luminance() if imnext.channels() != 1 else imnext.clone().mindim(self._mindim)
        flow = cv2.calcOpticalFlowFarneback(imp.numpy(), imn.numpy(), None, 0.5, 7, 16, 8, 5, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)  # parameterization for mindim=256
        return Image(flow).resize_like(imprev, interp='nearest')  # flow only, no objects
        
    def _videoflow(self, v, dt=1, step=1):
        assert isinstance(v, vipy.video.Video)
        imf = [self._imageflow(v[k], v[k+dt]) for k in range(0, len(v.load()), step) if k+dt < len(v.load())]
        return Video(np.stack([im.flow() for im in imf]))  # flow only, no objects
        
    def euclidean(self, imprev, imnext, border=0.1, contrast=(16.0/255.0), smooth=None, verbose=True, dilate=1.5):
        """Euclidean flow field, uses procrustes analysis for global rotation and translation"""
        flow = self.__call__(imprev, imnext).flow()
        (H,W) = (imprev.height(), imprev.width())

        # Rotation and translation estimation
        m = imprev.dilate(dilate).rectangular_mask()  # ignore foreground regions
        b = imprev.border_mask(int(border*min(W,H)))  # ignore borders
        w = np.uint8(np.sum(np.abs(np.gradient(imprev.clone().greyscale().numpy())), axis=0) < contrast)  # ignore low contrast regions
        bk = np.nonzero((m+b) == 0)  # indexes for valid flow regions
        (dx, dy) = (np.mean(flow[bk][:,0]), np.mean(flow[bk][:,1]))  # global background translation
        (X,Y) = np.meshgrid(np.arange(0, imprev.width()), np.arange(0, imprev.height()))
        (fx, fy) = (flow[:,:,0].flatten(), flow[:,:,1].flatten())  # flow
        (x1, y1) = ((X.flatten() - (W/2.0)), (Y.flatten() - (H/2.0)))  # source coordinates (point centered)
        (x2, y2) = (x1 + fx, y1 + fy)  # destination coordinates
        (x2, y2) = (x2 - np.mean(x2), y2 - np.mean(y2))  # destination coordinates (point centered)
        r = -np.arctan(np.sum(np.multiply(x2, y1) - np.multiply(y2, x1)) / np.sum(np.multiply(x2, x1) + np.multiply(y2, y1)))

        # Parameter smoother: smooth=None or smooth=0 for first iteration to clear history
        if smooth is None or not hasattr(self, '_r') or smooth == 0:
            (self._r, self._dx, self._dy) = ([], [], [])
        else:
            (self._r, self._dx, self._dy) = (self._r+[r], self._dx+[dx], self._dy+[dy])
            (r, dx, dy) = (np.mean(self._r[-smooth:]), np.mean(self._dx[-smooth:]), np.mean(self._dy[-smooth:]))
            
        # Euclidean flow field for global rotation and translation to align imprev to imnext (with tranposed rotation matrix for image coordinates)
        if verbose:
            print('[vipy.flow]: rot=%1.3f, tx=%1.2f, ty=%1.2f' % (r, dx, dy))
        flow = np.array(np.dstack((np.array(x1*np.cos(r) - y1*np.sin(r) - x1 + dx).reshape(H,W),
                                   np.array(x1*np.sin(r) + y1*np.cos(r) - y1 + dy).reshape(H,W))))
        return Image(flow)
        
    def stabilize(self, v, border=0.05, smooth=None, verbose=True, strict=True, dilate=1.5):
        """Rotation and translation stabilization"""
        assert isinstance(v, vipy.video.Video)

        # Compute euclidean flow on all frames to the middle frame
        k_middle = len(v.load())//2
        imflow = [self.euclidean(v[k], v[k_middle], verbose=verbose, border=border, dilate=dilate, smooth=smooth if k>0 else 0).print('[vipy.flow][%d/%d]: ' % (k,len(v)), verbose) for k in range(0, len(v.load()))]
        if strict:
            dt = np.max(np.abs(np.gradient([(np.mean(imf.dx()), np.mean(imf.dy())) for imf in imflow], axis=0)))  # maximum framewise translation
            assert dt < border*v.mindim(), "Flow stabilization failed (minimum frame overlap violation) - %1.1f (max dt) > %1.1f (mindim*border)" % (dt, v.mindim()*border)
        
        # Stabilization to middle frame
        imwarp = [imf.warp(im) for (im, imf) in zip(v, imflow)]

        # Return stabilized video with spatially aligned tracks
        return (v.clone(flushfilter=True).nofilename().nourl()
                .array(np.stack([im.numpy() for im in imwarp]))
                .trackmap(lambda t: t.keyboxes([bb for im in imwarp for bb in im.objects() if bb.attributes['trackid'] == t.id()],
                                               keyframes=list(range(0,len(imwarp))))))
                                              
    
    def gpu(self):
        pass


def FlowNet(Flow):
    def __init__(self):
        pass

    def __call__(self, imprev, imnext):
        pass

    
