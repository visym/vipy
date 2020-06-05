from vipy.util import mat2gray, try_import
try_import('cv2', 'opencv-python opencv-contrib-python'); import cv2
import vipy.image
from vipy.math import cartesian_to_polar
import numpy as np
import scipy.interpolate


class Image(vipy.image.Image):
    def __init__(self, array):
        super(Image, self).__init__(array=array.astype(np.float32), colorspace='float')

    def __repr__(self):
        return str('<vipy.flow: height=%d, width=%d, channels=%d>' % (self.height(), self.width(), self.channels()))

    def colorflow(self, minflow=None, maxflow=None, maxpct=0.05):
        """Flow visualization image (HSV: H=flow angle, V=flow magnitude), returns vipy.image.Image()"""
        (r, t) = cartesian_to_polar(self._array[:,:,0], self._array[:,:,1])
        hsv = np.zeros( (self.height(), self.width(), 3), dtype=np.uint8)
        hsv[:,:,0] = (((t+np.pi) * (180 / np.pi))*(255.0/360.0))
        hsv[:,:,1] = 255
        hsv[:,:,2] = 255*mat2gray(r, min=minflow, max=maxflow if maxflow is not None else maxpct*max(self.shape()))
        return vipy.image.Image(array=np.uint8(hsv), colorspace='hsv').rgb()

    def warp(self, im):
        """Warp vipy.image.Image() using computed flow from imprev to imnext updating objects"""
        (h, w) = self.shape()
        flow = -self.numpy()        
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        return (im.clone()
                  .array( cv2.remap(im.numpy(), flow, None, cv2.INTER_LINEAR) )
                  .objectmap(lambda bb: bb.int().offset(dx=np.mean(flow[bb.ymin():bb.ymax(), bb.xmin():bb.xmax(), 0]),
                                                        dy=np.mean(flow[bb.ymin():bb.ymax(), bb.xmin():bb.xmax(), 1]))))

class Video(vipy.video.Video):
    def __init__(self, array):
        super(Video, self).__init__(array=array.astype(np.float32), colorspace='float')

    def __repr__(self):
        return str('<vipy.flow: height=%d, width=%d, channels=%d>' % (self.height(), self.width(), self.channels()))

    def colorflow(self):
        """Flow visualization video"""
        return vipy.video.Video(array=np.stack([Image(im.numpy()).colorflow().numpy() for im in self]), colorspace='rgb')
                
    
class Flow(object):
    def __init__(self):
        pass

    def __call__(self, imprev=None, imnext=None):
        return self._videoflow(imprev) if imnext is None else self._imageflow(imprev, imnext)
    
    def _imageflow(self, imprev, imnext):
        """Default opencv dense flow.  This should be overloaded"""        
        assert isinstance(imprev, vipy.image.Image) and isinstance(imnext, vipy.image.Image)
        imprev = imprev.clone().luminance() if imprev.channels() != 1 else imprev
        imnext = imnext.clone().luminance() if imnext.channels() != 1 else imnext
        flow = cv2.calcOpticalFlowFarneback(imprev.numpy(), imnext.numpy(), None, 0.5, 7, 127, 64, 7, 1.5, 0)                
        return Image(flow)  # flow only, no objects

    def _videoflow(self, v, dt=1):
        assert isinstance(v, vipy.video.Video)
        imf = [self._imageflow(v[k], v[k+dt]) for k in range(0, len(v.load())-1, dt)]
        return Video(np.stack([im.numpy() for im in imf]))  # flow only, no objects
        
    def euclidean(self, imprev, imnext, border=0.1, contrast=(5.0/255.0)):
        """Euclidean flow field"""
        flow = self.__call__(imprev, imnext).array()
        (H,W) = (imprev.height(), imprev.width())
        
        m = imprev.rectangular_mask()  # ignore foreground regions
        b = imprev.border_mask(int(border*min(W,H)))  # ignore borders
        w = np.uint8(np.sum(np.abs(np.gradient(imprev.clone().greyscale())), axis=0) < contrast)  # ignore low contrast regions
        bk = np.nonzero((m+b+w) == 0)  # indexes for valid flow regions
        (dx, dy) = (np.median(flow[bk][:,0]), np.median(flow[bk][:,1]))  # global background translation (robust)        
        
        (X,Y) = np.meshgrid(np.arange(0, imprev.width()), np.arange(0, imprev.height()))
        (fx, fy) = (flow[:,:,0].flatten(), flow[:,:,1].flatten())  # flow
        (x1, y1) = ((X.flatten() - (W/2.0)), (Y.flatten() - (H/2.0)))  # source coordinates (image centered)
        (x2, y2) = (x1 + fx - dx, y1 + fy - dy)  # destination coordinates (image centered, translated)
        r = np.median(np.array(np.arctan2(y2,x2) - np.arctan2(y1,x1)).reshape(H,W)[bk].flatten())  # global rotation prev -> next (robust)

        # Euclidean flow field equivalent to a global rotation and translation to align imprev to imnext
        flow = np.array(np.dstack((np.array(x1*np.cos(r) - y1*np.sin(r) - x1 + dx).reshape(H,W),
                                   np.array(x1*np.sin(r) + y1*np.cos(r) - y1 + dy).reshape(H,W))))
        return Image(flow)
        
    def stabilize(self, v, strict=False, t_stabilized=10, border=0.1):
        """Rotation and translation stabilization"""
        assert isinstance(v, vipy.video.Video)

        # Compute euclidean flow on all frames to the middle frame
        k_middle = len(v.load())//2
        imflow = [self.euclidean(v[k], v[k_middle], border=border).print('[vipy.flow][%d/%d]: ' % (k,len(v))) for k in range(0, len(v.load()))]

        # Stabilization to middle frame (with optional strict overlap check)
        imwarp = [imf.warp(im) for (im, imf) in zip(v, imflow)]
        if strict:
            assert all([np.median(img - v[k_middle].numpy()) < t_stabilized for img in imwarp]), "Stabilization failed - Frame alignment residual too high"

        # Return stabilized video with spatially aligned tracks
        return (v.clone(flushfilter=True).nofilename().nourl()
                .array(np.stack([im.numpy() for im in imwarp])))
                #.trackmap(lambda t: t.resample(dt=1).keyboxes([bb for im in imwarp for bb in im.objects() if bb.attributes['trackid'] == t.id()])))       

    def gpu(self):
        pass


def FlowNet(Flow):
    def __init__(self):
        pass

    def __call__(self, imprev, imnext):
        pass

    
