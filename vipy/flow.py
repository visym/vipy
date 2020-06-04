from vipy.util import mat2gray, try_import
try_import('cv2', 'opencv-python opencv-contrib-python'); import cv2
import vipy.image
from vipy.math import cartesian_to_polar
import numpy as np



class Image(vipy.image.Image):
    def __init__(self, array):
        super(Image, self).__init__(array=array, colorspace='float')

    def __repr__(self):
        return str('<vipy.flow: height=%d, width=%d, channels=%d>' % (self.height(), self.width(), self.channels()))

    def image(self):
        """Flow visualization image, returns vipy.image.Image()"""
        (r, t) = cartesian_to_polar(self._array[:,:,0], self._array[:,:,1])
        hsv = np.zeros( (self.height(), self.width(), 3), dtype=np.uint8)
        hsv[:,:,0] = (t * 180 / np.pi / 2)      
        hsv[:,:,1] = 255
        hsv[:,:,2] = mat2gray(r)*255.0
        return self.clone().array(np.uint8(hsv)).colorspace('hsv').rgb()

    def warp(self, im):
        """Warp vipy.image.Image() using computed flow from imprev to imnext"""
        (h, w) = self.shape()
        flow = -self.numpy()        
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        return im.clone().array( cv2.remap(im.numpy(), flow, None, cv2.INTER_LINEAR) )

    
class Flow(object):
    def __init__(self):
        pass

    def __call__(self, imprev, imnext):
        """Default opencv dense flow"""
        assert isinstance(imprev, vipy.image.Image) and isinstance(imnext, vipy.image.Image)
        imprev = imprev.clone().luminance() if imprev.channels() != 1 else imprev
        imnext = imnext.clone().luminance() if imnext.channels() != 1 else imnext
        flow = cv2.calcOpticalFlowFarneback(imprev.numpy(), imnext.numpy(), None, 0.5, 7, 127, 64, 7, 1.5, 0)                
        return Image(flow)

    def gpu(self):
        pass


