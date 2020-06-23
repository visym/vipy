from vipy.util import mat2gray, try_import, string_to_pil_interpolation, Stopwatch, isnumpy
try_import('cv2', 'opencv-python opencv-contrib-python'); import cv2
import vipy.image
from vipy.math import cartesian_to_polar
import numpy as np
import scipy.interpolate
import vipy.object
import PIL.Image
import copy
import vipy.geometry
import warnings


class Image(object):
    """vipy.flow.Image() class"""
    
    def __init__(self, array):
        assert array.ndim == 3 and array.shape[2] == 2, "Must be HxWx2 flow array"
        self._array = array
        
    def __repr__(self):
        return str('<vipy.flow: height=%d, width=%d, minflow=%1.2f, maxflow=%1.2f>' % (self.height(), self.width(), self.min(), self.max()))

    def __add__(self, imf):
        assert isinstance(imf, Image)
        return self.clone().flow( self.flow() + imf.flow() )

    def __sub__(self, imf):
        assert isinstance(imf, Image)
        return self.clone().flow( self.flow() - imf.flow() )
    
    def min(self, minflow=None):
        if minflow is None:
            return np.min(self._array)
        else:
            self._array = np.maximum(minflow, self._array)
            return self
            
    def max(self, maxflow=None):
        if maxflow is None:
            return np.max(self._array)
        else:
            self._array = np.minimum(maxflow, self._array)
            return self

    def scale(self, s):
        self._array *= s
        return self
        
    def width(self):
        return self._array.shape[1]

    def height(self):
        return self._array.shape[0]

    def shape(self):
        return (self.height(), self.width())
    
    def flow(self, array=None):
        if array is None:
            return self._array
        else:
            self._array = array
            return self
    
    def colorflow(self, minmag=None, maxmag=None):
        """Flow visualization image (HSV: H=flow angle, V=flow magnitude), returns vipy.image.Image()"""
        flow = self.flow()
        (r, t) = cartesian_to_polar(flow[:,:,0], flow[:,:,1])
        hsv = np.zeros( (self.height(), self.width(), 3), dtype=np.uint8)
        hsv[:,:,0] = (((t+np.pi) * (180 / np.pi))*(255.0/360.0))
        hsv[:,:,1] = 255
        hsv[:,:,2] = 255*mat2gray(r, min=minmag, max=maxmag)  
        return vipy.image.Image(array=np.uint8(hsv), colorspace='hsv').rgb()
        
    def warp(self, imfrom, imto=None):
        """Warp image imfrom=vipy.image.Image() to imto=vipy.image.Image() using flow computed as imfrom->imto, updating objects"""
        (H, W) = self.shape()
        flow = -self.flow().astype(np.float32)
        flow[:,:,0] += np.arange(W)
        flow[:,:,1] += np.arange(H)[:,np.newaxis]
        imwarp = (imfrom.clone()
                  .array( cv2.remap(imfrom.numpy(), flow, None, cv2.INTER_LINEAR, dst=imto._array if imto is not None else None, borderMode=cv2.BORDER_TRANSPARENT if imto is not None else cv2.BORDER_CONSTANT)))
        if isinstance(imwarp, vipy.image.Scene):
            imwarp.objectmap(lambda bb: bb.int().offset(dx=np.mean(self.dx()[bb.ymin():bb.ymax(), bb.xmin():bb.xmax()]),
                                                        dy=np.mean(self.dy()[bb.ymin():bb.ymax(), bb.xmin():bb.xmax()])))
        return imwarp

    def alphapad(self, pad=None, to=None, like=None):
        assert pad is not None or to is not None or like is not None
        pad_width = (pad, pad) if pad is not None else ((to[0]-self.height())//2, int(np.ceil((to[1] - self.width())/2))) if to is not None else ((like.height()-self.height())//2, int(np.ceil((like.width() - self.width())/2)))
        assert np.all([p >= 0 for p in pad_width])
        self._array = np.pad(self._array, pad_width=(pad_width, pad_width, (0,0)), mode='constant', constant_values=-100000)  # -inf
        return self
                
    def zeropad(self, pad=None, to=None, like=None):
        assert pad is not None or to is not None or like is not None
        pad_width = (pad, pad) if pad is not None else ((to[0]-self.height())//2, int(np.ceil((to[1] - self.width())/2))) if to is not None else ((like.height()-self.height())//2, int(np.ceil((like.width() - self.width())/2)))
        assert np.all([p >= 0 for p in pad_width])
        self._array = np.pad(self._array, pad_width=(pad_width, pad_width, (0,0)), mode='constant', constant_values=0)
        return self
                
    def dx(self):
        """Return dx (horizontal) component of flow"""
        return self.flow()[:,:,0]

    def dy(self):
        """Return dy (vertical) component of flow"""        
        return self.flow()[:,:,1]

    def shift(self, f):
        self._array += f
        return self
    
    def show(self, figure=None, nowindow=False):
        self.colorflow().show(figure, nowindow)
    
    def rescale(self, scale, interp='bicubic'):
        (height, width) = self.shape()
        return self.resize(int(np.round(scale * height)), int(np.round(scale * width)), interp)

    def resize_like(self, im, interp='bicubic'):
        """Resize flow buffer to be the same size as the provided vipy.image.Image()"""
        assert hasattr(im, 'width') and hasattr(im, 'height'), "Invalid input - Must be Image() object"        
        return self.resize(im.height(), im.width(), interp=interp) if self.shape() != im.shape() else self

    def resize(self, height, width, interp='bicubic'):
        (yscale, xscale) = (height/float(self.height()), width/float(self.width()))
        self._array = np.dstack((np.array(PIL.Image.fromarray(self.dx()*xscale).resize((width, height), string_to_pil_interpolation(interp))),
                                 np.array(PIL.Image.fromarray(self.dy()*yscale).resize((width, height), string_to_pil_interpolation(interp)))))                                 
        return self

    def magnitude(self):
        return cartesian_to_polar(self.dx(), self.dy())[0]

    def angle(self):
        return cartesian_to_polar(self.dx(), self.dy())[1]

    def clone(self):
        return copy.deepcopy(self)

    def print(self, outstring=None):
        print(outstring if outstring is not None else str(self))
        return self

    
class Video(vipy.video.Video):
    """vipy.flow.Video() class"""
    
    def __init__(self, array, flowstep, framestep):
        assert array.ndim == 4 and array.shape[3] == 2, "Must be NxHxWx2 flow array"        
        self._flowstep = flowstep 
        self._framestep = framestep
        self._array = array

    def __repr__(self):
        return str('<vipy.flow: frames=%d, height=%d, width=%d, keyframes=%d, framestep=%d, flowstep=%d, minflow=%1.2f, maxflow=%1.2f>' % (len(self), self.height(), self.width(), len(self._array), self._framestep, self._flowstep, self.min(), self.max()))        

    def __len__(self):
        return len(self._array)*self._framestep

    def __getitem__(self, k):
        assert k >= 0
        if self._flowstep == 1 and self._framestep == 1:
            return Image(self._array[k])
        else:
            # Flow interpolation
            (N,X,Y,F) = np.meshgrid(k, np.arange(self.height()), np.arange(self.width()), np.arange(2))
            xi = np.stack( [N.flatten(), X.flatten(), Y.flatten(), F.flatten()] ).transpose()
            x = scipy.interpolate.interpn( (np.arange(0, len(self), self._framestep), np.arange(self.height()), np.arange(self.width()), np.arange(2)),
                                           self.flow() / float(self._flowstep),
                                           xi,
                                           method='linear', bounds_error=False, fill_value=0)
            return Image(x.reshape( (self.height(), self.width(), 2) ))

    def __iter__(self):
        for k in np.arange(len(self)):
            yield self.__getitem__(k)        
        
    def min(self):
        return np.min(self._array)

    def max(self):
        return np.max(self._array)

    def width(self):
        return self._array.shape[2]

    def height(self):
        return self._array.shape[1]

    def flow(self):
        return self._array
    
    def colorflow(self):
        """Flow visualization video"""
        (minmag, maxmag) = (np.min(self.magnitude()), np.max(self.magnitude()))  # scaled over video
        return vipy.video.Video(array=np.stack([im.colorflow(minmag=minmag, maxmag=maxmag).numpy() for im in self]), colorspace='rgb')

    def magnitude(self):
        return np.stack([cartesian_to_polar(f[:,:,0], f[:,:,1])[0] for f in self.flow()])
    
    def show(self):
        return self.colorflow().show()

    def print(self, outstring=None):
        print(outstring if outstring is not None else str(self))
        return self

    
class Flow(object):
    """vipy.flow.Flow() class"""
    
    def __init__(self, flowiter=10):
        self._mindim = 256  # flow computation dimensionality, change the flow parameters if you change this
        self._flowiter = flowiter
        
    def __call__(self, im, imprev=None, flowstep=1, framestep=1):
        return self.videoflow(im, flowstep, framestep) if imprev is None else self.imageflow(im, imprev)

    def _numpyflow(self, img, imgprev):
        """Overload this method for custom flow classes"""        
        return Image(cv2.calcOpticalFlowFarneback(img, imgprev, None, 0.5, 3, 7, self._flowiter, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN))
        
    def imageflow(self, im, imprev):
        """Default opencv dense flow, from im to imprev.  This should be overloaded"""        
        assert isinstance(imprev, vipy.image.Image) and isinstance(im, vipy.image.Image)
        imp = imprev.clone().mindim(self._mindim).luminance() if imprev.channels() != 1 else imprev.clone().mindim(self._mindim)
        imn = im.clone().mindim(self._mindim).luminance() if im.channels() != 1 else im.clone().mindim(self._mindim)
        imflow = self._numpyflow(imn.numpy(), imp.numpy())
        return imflow.resize_like(im)  # flow only, no objects
        
    def videoflow(self, v, flowstep=1, framestep=1, keyframe=None):
        """Compute optical flow for a video framewise skipping framestep frames, compute optical flow acrsos flowstep frames, """
        assert isinstance(v, vipy.video.Video)
        imf = [self.imageflow(v[k], v[max(0, k-flowstep) if keyframe is None else keyframe]) for k in range(0, len(v.load())+framestep, framestep) if k < len(v.load())]
        return Video(np.stack([im.flow() for im in imf]), flowstep, framestep)  # flow only, no objects

    def keyflow(self, v, keystep=None):
        """Compute optical flow for a video framewise relative to keyframes separated by keystep"""
        assert isinstance(v, vipy.video.Video)
        imf = [(self.imageflow(v[min(len(v)-1, int(keystep*np.round(k/keystep)))], v[max(0, k-1)]) -
                self.imageflow(v[min(len(v)-1, int(keystep*np.round(k/keystep)))], v[k]))
               for k in range(0, len(v.load()))]
        return Video(np.stack([im.flow() for im in imf]), flowstep=1, framestep=1)  # flow only, no objects

    def affineflow(self, A, H, W):
        """Return a flow field of size (height=H, width=W) consistent with a 2x3 affine transformation A"""
        assert isnumpy(A) and A.shape == (2,3) and H > 0 and W > 0, "Invalid input"
        (X, Y) = np.meshgrid(np.arange(0, W,), np.arange(0, H))
        (x, y) = (X.flatten() - np.mean(X.flatten()), Y.flatten() - np.mean(Y.flatten()))
        (xf, yf) = np.dot(A, vipy.geometry.homogenize(np.vstack( (x, y))))
        return Image(np.dstack( ((x-xf).reshape(H,W), (y-yf).reshape(H,W))))

    def euclideanflow(self, R, t, H, W):
        """Return a flow field of size (height=H, width=W) consistent with an Euclidean transform parameterized by a 2x2 Rotation and 2x1 translation"""  
        return self.affineflow(np.array([[R[0,0], R[0,1], t[0]], [R[1,0], R[1,1], t[1]]]), H, W)
    
    def _correspondence(self, imflow, im, border=0.1, contrast=(16.0/255.0), dilate=1.0, validmask=None):
        (H,W) = (imflow.height(), imflow.width())
        m = im.clone().dilate(dilate).rectangular_mask() if (dilate  is not None and isinstance(im, vipy.image.Scene) and len(im.objects())>0) else 0  # ignore foreground regions
        b = im.border_mask(int(border*min(W,H))) if border is not None else 0  # ignore borders
        w = np.uint8(np.sum(np.abs(np.gradient(im.clone().greyscale().numpy())), axis=0) < contrast) if contrast is not None else 0  # ignore low contrast regions
        v = (1-np.float32(validmask)) if validmask is not None else 0  # ignore non-valid regions
        bk = np.nonzero((m+b+w+v) == 0)  # indexes for valid flow regions
        (X, Y) = np.meshgrid(np.arange(0, im.width()), np.arange(0, im.height()))        
        (fx, fy) = (imflow.dx()[bk].flatten(), imflow.dy()[bk].flatten())  # flow
        (x1, y1) = (X[bk].flatten(), Y[bk].flatten())  # image coordinates
        (x2, y2) = (x1 + fx, y1 + fy)  # destination coordinates
        return (np.stack((x1,y1)), np.stack((x2,y2)))
        
    def stabilize(self, v, keystep=20, padheight=32, padwidth=128, border=0.1, dilate=1.0, contrast=16.0/255.0, rigid=False, affine=True, verbose=True):
        """Affine stabilization to frame zero using multi-scale optical flow correspondence with foreground object keepouts.  

           * v [vipy.video.Scene]:  The input video to stabilize, should be resized to mindim=256
           * keystep [int]:  The local stabilization step between keyframes (should be <= 30)
           * padheight [int]:  The height padding to be applied to output video to allow for vertical stabilization
           * padwidth [int]:  The width padding to be applied to output video to allow for horizontal stabilization
           * border [float]:  The border keepout fraction to ignore during flow correspondence.  This should be proportional to the maximum frame to frame flow
           * dilate [float]:  The dilation to apply to the foreground object boxes to define a foregroun keepout for flow computation
           * contrast [float]:  The minimum gradient necessary for flow correspondence, to avoid flow on low contrast regions
           * rigid [bool]:  Euclidean stabilization
           * affine [bool]:  Affine stabilization
           * verbose [bool]:  This takes a while to run ...
        
           * NOTE: The remaining distortion after stabilization is due to: rolling shutter distortion, perspective distortion and non-keepout moving objects in background
           * NOTE: If the video contains objects, the object boxes will be transformed along with the stabilization 

        """        
        assert isinstance(v, vipy.video.Scene), "Invalid input - Must be vipy.video.Scene() with foreground object keepouts for background stabilization"
        if min(v.shape()) != 256:
            warnings.warn('You should resize the input video to v.mindim(256) prior to calling this function, otherwise it will take a while')
        
        # Prepare videos
        v = v.clone().cropeven()  # make even for zero pad
        vc = v.clone(flush=True).zeropad(padwidth, padheight).load().nofilename().nourl()       
        
        # Optical flow (three passes)        
        if verbose:
            print('[vipy.flow.stabilize]: Optical flow (3x)- %s' % v)
        vf = self.videoflow(v, framestep=1, flowstep=1)
        vfk1 = self.keyflow(v, keystep=keystep)
        vfk2 = self.keyflow(v, keystep=len(v)//2)        
        
        # Stabilization
        assert rigid is True or affine is True
        (A, T) = (np.array([ [1,0,0],[0,1,0],[0,0,1] ]).astype(np.float64), np.array([[1,0,padwidth],[0,1,padheight],[0,0,1]]).astype(np.float64))        
        f_estimate_coarse = ((lambda s, *args, **kw: np.vstack( (cv2.estimateAffinePartial2D(s, *args, **kw)[0], [0,0,1])).astype(np.float64)) if rigid else
                             (lambda s, *args, **kw: np.vstack( (cv2.estimateAffine2D(s, *args, **kw)[0], [0,0,1])).astype(np.float64)))
        f_estimate_fine = (lambda s, *args, **kw: cv2.findHomography(s, *args, **kw)[0]) if not (rigid or affine) else f_estimate_coarse 
        f_warp_coarse = cv2.warpAffine
        f_warp_fine = cv2.warpAffine if (rigid or affine) else cv2.warpPerspective
        f_transform_coarse = (lambda A: A[0:2,:])
        f_transform_fine = (lambda A: A[0:2,:]) if (rigid or affine) else (lambda A: A)
        imstabilized = v[0].clone().rgb().zeropad(padwidth, padheight)
        frames = []                        
        for (k, (im, imf, imfk1, imfk2)) in enumerate(zip(v, vf, vfk1, vfk2)): 
            if verbose and k==0:
                print('[vipy.flow.stabilize]: %s coarse to fine stabilization ...' % ('Euclidean' if rigid else 'Affine' if affine else 'Projective'))                
           
            # Coarse alignment 
            (xy_src_k0, xy_dst_k0) = self._correspondence(imf, im, border=border, dilate=dilate, contrast=contrast)
            (xy_src_k1, xy_dst_k1) = self._correspondence(imfk1, im, border=border, dilate=dilate, contrast=contrast)
            (xy_src_k2, xy_dst_k2) = self._correspondence(imfk2, im, border=border, dilate=dilate, contrast=contrast)
            (xy_src, xy_dst) = (np.hstack( (xy_src_k0, xy_src_k1, xy_src_k2) ).transpose(), np.hstack( (xy_dst_k0, xy_dst_k1, xy_dst_k2) ).transpose())
            M = f_estimate_coarse(xy_src, xy_dst, method=cv2.RANSAC, confidence=0.9999, ransacReprojThreshold=0.1, refineIters=16, maxIters=2000)

            # Fine alignment
            A = A.dot(M)  # update coarse reference frame
            imfine = im.clone().array(f_warp_coarse(im.clone().numpy(), dst=imstabilized.clone().numpy(), M=f_transform_coarse(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT), copy=True).objectmap(lambda o: o.projective(T.dot(A)))
            imfinemask = f_warp_coarse(np.ones_like(im.clone().greyscale().numpy()), dst=np.zeros_like(imstabilized.numpy()), M=f_transform_coarse(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT) > 0
            imfineflow = self.imageflow(imfine, imstabilized)
            (xy_src, xy_dst) = self._correspondence(imfineflow, imfine, border=None, dilate=dilate, contrast=contrast, validmask=imfinemask)
            F = f_estimate_fine(xy_src.transpose()-np.array([padwidth, padheight]), xy_dst.transpose()-np.array([padwidth, padheight]), method=cv2.RANSAC, confidence=0.9999, ransacReprojThreshold=0.1, refineIters=16, maxIters=2000)  
            
            # Render coarse to fine stabilized frame with aligned objects
            A = F.dot(A)  # update fine reference frame            
            f_warp_fine(im.numpy(), dst=imstabilized._array, M=f_transform_fine(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT)
            vc.addframe( im.array(imstabilized.array(), copy=True).objectmap(lambda o: o.projective(T.dot(A))), frame=k)
                            
        return vc
            

    
