from vipy.globals import log
from vipy.util import mat2gray, try_import, string_to_pil_interpolation, Stopwatch, isnumpy, clockstamp, tempMP4, premkdir
try_import('cv2', 'opencv-python opencv-contrib-python'); import cv2
import vipy.image
from vipy.math import cartesian_to_polar, even
import numpy as np
try_import('scipy.interpolate', 'scipy')
import scipy.interpolate
import vipy.object
import PIL.Image
import copy
import vipy.geometry
from vipy.geometry import homogenize



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

    def threshold(self, t):
        m = np.float32(self.magnitude() < t)
        self._array[:,:,0] = np.multiply(m, self._array[:,:,0])
        self._array[:,:,1] = np.multiply(m, self._array[:,:,1])                
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
        self.colorflow().show(figure=figure, nowindow=nowindow)
    
    def rescale(self, scale, interp='bicubic'):
        (height, width) = self.shape()
        return self.resize(int(np.round(scale * height)), int(np.round(scale * width)), interp)

    def resize_like(self, im, interp='bicubic'):
        """Resize flow buffer to be the same size as the provided vipy.image.Image()"""
        assert hasattr(im, 'width') and hasattr(im, 'height'), "Invalid input - Must be Image() object"        
        return self.resize(im.height(), im.width(), interp=interp) if self.shape() != im.shape() else self

    def resize(self, height, width, interp='bicubic'):
        assert height > 0 and width > 0, "Invalid input"
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
        log.info(outstring if outstring is not None else str(self))
        return self

    
class Video(vipy.video.Video):
    """vipy.flow.Video() class"""
    
    def __init__(self, array, flowstep, framestep):
        assert array.ndim == 4 and array.shape[3] == 2, "Must be NxHxWx2 flow array"        
        assert flowstep > 0, "Invalid flowstep"
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
        log.info(outstring if outstring is not None else str(self))
        return self

    
class Flow(object):
    """vipy.flow.Flow() class"""
    
    def __init__(self, flowiter=10, flowdim=256, gpu=None):
        self._mindim = flowdim
        self._levels = 3
        self._winsize = 7
        self._poly_n = 5
        self._poly_sigma = 1.2
        self._flowiter = flowiter
        self._gpu = gpu
        if gpu == True:
            try:
                f = cv2.cuda_FarnebackOpticalFlow
            except:
                log.warning('OpenCV not CUDA enabled - GPU acceleration is unavailable')
                self._gpu = None

        self._sparse_matcher = lambda x,y,m=cv2.BFMatcher(cv2.NORM_HAMMING).match: m(x,y)  # matcher on desc
        self._sparse_features = lambda img, f=cv2.ORB_create().detectAndCompute: f(img,None)  # returns (kp, desc), must be greyscale
        

    def __call__(self, im, imprev=None, flowstep=1, framestep=1):
        return self.videoflow(im, flowstep, framestep) if imprev is None else self.imageflow(im, imprev)

    def _numpyflow_gpu(self, img, imgprev):
        """Optical flow on GPU"""

        # To compile CUDA enabled opencv (YUCK):
        #
        #   sh> python3 -m venv /path/to/myvirtualenv
        #   sh> source /path/to/myvirtualenv/bin/activate   # for python bindings
        #   sh> git clone --recursive https://github.com/opencv/opencv-python.git
        #   sh> cd opencv-python
        #   sh> export ENABLE_CONTRIB=1 ENABLE_HEADLESS=1  # opencv packages
        #   sh> export CMAKE_ARGS="-DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DWITH_CUDA=ON -DCUDA_ARCH_BIN=5.2 -DCMAKE_CXX_STANDARD=11 -DPYTHON3_EXECUTABLE=$(which python) -DINSTALL_PYTHON_EXAMPLES=OFF -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_FLAGS=\"-std=c++11\""
        #   sh> pip wheel . --verbose
        #   sh> pip install /path/to/opencv_contrib_python_headless-4.5.3+c1cc7e5-cp36-cp36m-linux_x86_64.whl   # path output at compile time
        #   sh> cd ..  # do not import from within-source
        #   sh> python
        #   >>> import cv2
        #   >>> cv2.cuda_FarnebackOpticalFlow
        #
        # See also: 
        #   https://github.com/opencv/opencv-python#manual-builds
        #   https://learnopencv.com/getting-started-opencv-cuda-module/
        #   https://developer.nvidia.com/blog/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/

        (gpu_img, gpu_imgprev) = (cv2.cuda_GpuMat(), cv2.cuda_GpuMat())
        gpu_img.upload(img)
        gpu_imgprev.upload(imgprev)
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(self._levels, 0.5, False, self._winsize, self._flowiter, self._poly_n, self._poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_img, gpu_imgprev, None)
        flow = gpu_flow.download()        
        return vipy.flow.Image(flow)

    def _numpyflow_cpu(self, img, imgprev):
        """Overload this method for custom flow classes"""        
        return Image(cv2.calcOpticalFlowFarneback(img, imgprev, None, 0.5, self._levels, self._winsize, self._flowiter, self._poly_n, self._poly_sigma, cv2.OPTFLOW_FARNEBACK_GAUSSIAN))

    def _numpyflow(self, img, imgprev):
        """Overload this method for custom flow classes"""        
        f = self._numpyflow_cpu if self._gpu is None or self._gpu == False else self._numpyflow_gpu
        return f(img, imgprev)
        
    def imageflow(self, im, imprev):
        """Default opencv dense flow, from im to imprev.  This should be overloaded"""        
        assert isinstance(imprev, vipy.image.Image) and isinstance(im, vipy.image.Image)
        self._mindim = self._mindim if self._mindim is not None else im.mindim()
        imp = imprev.clone().mindim(self._mindim).luminance() if imprev.channels() != 1 else imprev.clone().mindim(self._mindim)
        imn = im.clone().mindim(self._mindim).luminance() if im.channels() != 1 else im.clone().mindim(self._mindim)
        imflow = self._numpyflow(imn.numpy(), imp.numpy())
        return imflow.resize_like(im, interp='nearest')  # flow only, no objects
        
    def videoflow(self, v, flowstep=1, framestep=1, keyframe=None):
        """Compute optical flow for a video framewise skipping framestep frames, compute optical flow acrsos flowstep frames, """
        assert isinstance(v, vipy.video.Video)
        imf = [self.imageflow(v[k], v[max(0, k-flowstep) if keyframe is None else keyframe]) for k in range(0, len(v.load())+framestep, framestep) if k < len(v.load())]
        return Video(np.stack([im.flow() for im in imf]), flowstep, framestep)  # flow only, no objects

    def videoflowframe(self, v, frame, duration, flowstep=1, framestep=1, keyframe=None):
        """Computer the videoflow for a single frame"""
        assert isinstance(v, vipy.video.Video)
        assert flowstep == 1 and framestep == 1
        imf = [self.imageflow(v.frame(k), v.frame(max(0, k-flowstep) if keyframe is None else keyframe)) for k in range(frame, frame+framestep, framestep) if k < duration]
        return imf[0]

    def keyflow(self, v, keystep=None):
        """Compute optical flow for a video framewise relative to keyframes separated by keystep"""
        assert isinstance(v, vipy.video.Video)
        imf = [(self.imageflow(v[min(len(v)-1, int(keystep*np.round(k/keystep)))], v[max(0, k-1)]) -
                self.imageflow(v[min(len(v)-1, int(keystep*np.round(k/keystep)))], v[k]))
               for k in range(0, len(v.load()))]
        return Video(np.stack([im.flow() for im in imf]), flowstep=1, framestep=1)  # flow only, no objects

    def keyflowframe(self, v, frame, duration, keystep=None):
        """Compute the keyflow for a single frame"""
        assert isinstance(v, vipy.video.Video)
        len_v = duration
        assert frame < len_v
        (ima, imb, imc) = (v.frame(min(len_v-1, int(keystep*np.round(frame/keystep)))), v.frame(max(0, frame-1)), v.frame(frame))
        return self.imageflow(ima, imb) - self.imageflow(ima, imc)

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
    
    def _correspondence(self, imflow, im, border=0.1, contrast=(16.0/255.0), dilate=1.0, validmask=None, maxflow=None, subsample=1):
        (H,W) = (imflow.height(), imflow.width())
        m = im.clone().dilate(dilate).rectangular_mask() if (dilate  is not None and isinstance(im, vipy.image.Scene) and len(im.objects())>0) else 0  # ignore foreground regions
        b = im.border_mask(int(border*min(W,H))) if border is not None else 0  # ignore borders
        w = np.uint8(np.sum(np.abs(np.gradient(im.clone().greyscale().numpy())), axis=0) < contrast) if contrast is not None else 0  # ignore low contrast regions
        v = (1-np.float32(validmask)) if validmask is not None else 0  # ignore non-valid regions
        x = np.float32(imflow.magnitude() > maxflow) if maxflow is not None else 0  # ignore maxflow region
        vf = (m+b+w+v+x)  # valid flow regions (non-zero elements)
        if subsample != 1:
            assert isinstance(subsample, int) and subsample > 1
            vf[::subsample, ::subsample] = 0  # zero out (ignore) neighboring flow
        bk = np.nonzero(vf == 0)  # indexes for valid flow regions
        (X, Y) = np.meshgrid(np.arange(0, im.width()), np.arange(0, im.height()))        
        (fx, fy) = (imflow.dx()[bk].flatten(), imflow.dy()[bk].flatten())  # flow
        (x1, y1) = (X[bk].flatten(), Y[bk].flatten())  # image coordinates
        (x2, y2) = (x1 + fx, y1 + fy)  # destination coordinates
        return (np.stack((x1,y1)), np.stack((x2,y2)))
        
    def _sparse_correspondence(self, img1, img2, radius=32):
        (kp1, desc1) = self._sparse_features(img1)
        (kp2, desc2) = self._sparse_features(img2)
        good_matches = [m for m in self._sparse_matcher(desc1, desc2) if (np.abs(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0]) < radius and
                                                                          np.abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) < radius)]
        return (np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2).transpose(),
                np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2).transpose())

    def stabilize(self, v, keystep=20, padheightfrac=0.125, padwidthfrac=0.25, padheightpx=None, padwidthpx=None, border=0.1, dilate=1.0, contrast=16.0/255.0, rigid=False, affine=True, verbose=True, strict=True, residual=False, maxflow=None, outfile=None, preload=True, framerate=5): 
        """Affine stabilization to frame zero using multi-scale optical flow correspondence with foreground object keepouts.  

        Recommended usage:  use the `vipy.video.Scene.stabilize` method on a `vipy.video.Video` object.

        ```python
        v = vipy.video.Scene(filename='/path/to/my/video.mp4').stabilize()
        ```

        Args:

            v: [`vipy.video.Scene`]:  The input video to stabilize, should be resized to mindim=256
            keystep: [int]  The local stabilization step between keyframes (should be <= 30)
            padheightfrac: [float] The height padding (relative to video height) to be applied to output video to allow for vertical stabilization
            padwidthfrac: [float]  The width padding (relative to video width) to be applied to output video to allow for horizontal stabilization
            padheightpx: [int]  The height padding to be applied to output video to allow for vertical stabilization.  Overrides padheight.
            padwidthpx: [int]  The width padding to be applied to output video to allow for horizontal stabilization.  Overrides padwidth.
            border: [float]  The border keepout fraction to ignore during flow correspondence.  This should be proportional to the maximum frame to frame flow
            dilate: [float]  The dilation to apply to the foreground object boxes to define a foregroun keepout for flow computation
            contrast: [float]  The minimum gradient necessary for flow correspondence, to avoid flow on low contrast regions
            rigid: [bool]  Euclidean stabilization
            affine: [bool]  Affine stabilization
            verbose: [bool]  This takes a while to run so show some progress ...
            strict: [bool]  If true, throw an exception on error, otherwise return the original video and set v.hasattribute('unstabilized'), useful for large scale stabilization
            outfile: [str] the file path to the stabilized output video
            preload [bool]: If true, load the input video into memory before stabilizing.  Faster, but requires video to fit into memory.
            framerate [float]: The framerate at which to compute the stabilization.  Videos will be stabilized at the native framerate of the input video, but will be linearly interpolated between keyframes aligned at this framerate

        Returns:

            A cloned `vipy.video.Scene` with filename=outfile, such that pixels and tracks are background stabilized.

        .. notes::
            - The remaining distortion after stabilization is due to: rolling shutter distortion, perspective distortion and non-keepout moving objects in background
            - If the video contains objects, the object boxes will be transformed along with the stabilization 
            - This requires loading videos entirely into memory.  Be careful with stabilizing long videos.
            - The returned video has the attribute 'stabilize' which contains the mean and median residual of the flow field relative to the motion model. This can be used for stabilization quality filtering.
            - Higher framerates result in more accurate stabilization, but take significantly longer. 

        """
        vc = v.clone()  # clone to avoid memory leaks in distributed processing
        vc = vc.saveas(tempMP4()) if vc.isloaded() else vc  # dump to temp file if loaded
        
        assert isinstance(vc, vipy.video.Scene), "Invalid input - Must be vipy.video.Scene() with foreground objects which provide keepouts for background stabilization"
        assert framerate>0 and framerate <= vc.framerate(), "Invalid framerate"
        vc = vc.framerate(framerate)  # resample to lower framerate

        # Prepare videos
        vv = vc.cropeven()  # make even for zero pad
        (padwidth, padheight) = (int(vv.width()*padwidthfrac) if padwidthpx is None else padwidthpx, int(vv.height()*padheightfrac) if padheightpx is None else padheightpx)  # width() height() triggers single frame fetch
        outfile = premkdir(outfile if outfile is not None else tempMP4())
        vs = vv.clone(flushforward=True, flushfilter=True).filename(outfile if outfile is not None else tempMP4()).nourl().cleartracks().framerate(v.framerate())   # stabilized video, does not trigger load, at input framerate
        vic = v.clone().cropeven()  # input video 
        s = vv.mindim() / float(self._mindim)  # for upsample
        vvd = vv.clone().mindim(self._mindim)  # downsampled for flow correspondence        
        if preload:
            (vv, vvd) = (vv.load(), vvd.load())  # Faster for random frame access, but requires lots of memory
        assert preload, "preload=True is required for now"

        # Stabilization parameters
        assert rigid is True or affine is True, "Projective stabilization is disabled"
        (A, T) = (np.array([ [1,0,0],[0,1,0],[0,0,1] ]).astype(np.float64), np.array([[1,0,padwidth],[0,1,padheight],[0,0,1]]).astype(np.float64))        
        f_estimate_coarse = ((lambda s, *args, **kw: np.vstack( (cv2.estimateAffinePartial2D(s, *args, **kw)[0], [0,0,1])).astype(np.float64)) if rigid else
                             (lambda s, *args, **kw: np.vstack( (cv2.estimateAffine2D(s, *args, **kw)[0], [0,0,1])).astype(np.float64)))
        f_estimate_fine = (lambda s, *args, **kw: cv2.findHomography(s, *args)[0]) if not (rigid or affine) else f_estimate_coarse 
        f_warp_coarse = cv2.warpAffine
        f_warp_fine = cv2.warpAffine if (rigid or affine) else cv2.warpPerspective
        f_transform_coarse = (lambda A: A[0:2,:])
        f_transform_fine = (lambda A: A[0:2,:]) if (rigid or affine) else (lambda A: A)
        imstabilized = vv.preview(0).rgb().zeropad(padwidth, padheight)  # single frame fetch
        duration = len(vv)  # requires preload, duration computed at stabilization framerate
        if duration < keystep:
            log.warning('[vipy.flow.stabilize]: video not long enough for stabilization, returning original video "%s"' % str(v))
            return v.clone().setattribute('unstabilized')
        r_coarse = []
        frames = []                        
        vs.setattribute('stabilize', {})

        # Stabilization
        for k in range(0, duration):  
            if verbose and k==0:
                log.info('[vipy.flow.stabilize]: %s coarse to fine stabilization ...' % ('Euclidean' if rigid else 'Affine' if affine else 'Projective'))                

            # Optical flow (3x): use downsampled video, do not precompute to save on memory, requires random access to downsampled video
            im = vv.frame(k)  # native resolution
            imf = self.videoflowframe(vvd, k, duration=duration, framestep=1, flowstep=1)
            imfk1 = self.keyflowframe(vvd, k, duration=duration, keystep=keystep)
            imfk2 = self.keyflowframe(vvd, k, duration=duration, keystep=duration//2)
            
            # Coarse alignment 
            imd = im.clone().rescale(1.0 / s)  # downsample
            (xy_src_k0, xy_dst_k0) = self._correspondence(imf, imd, border=border, dilate=dilate, contrast=contrast, maxflow=maxflow)
            (xy_src_k1, xy_dst_k1) = self._correspondence(imfk1, imd, border=border, dilate=dilate, contrast=contrast, maxflow=maxflow)
            (xy_src_k2, xy_dst_k2) = self._correspondence(imfk2, imd, border=border, dilate=dilate, contrast=contrast, maxflow=maxflow)
            (xy_src, xy_dst) = (np.hstack( (xy_src_k0, xy_src_k1, xy_src_k2) ).transpose(), np.hstack( (xy_dst_k0, xy_dst_k1, xy_dst_k2) ).transpose())  # Nx3
            try:            
                M = f_estimate_coarse(s*xy_src, s*xy_dst, method=cv2.RANSAC, confidence=0.99999, ransacReprojThreshold=0.1, refineIters=16, maxIters=3000)   # upsampled correspondences
                r_coarse.append(np.mean(np.sqrt(np.sum(np.square(M.dot(homogenize(xy_src[::8].transpose())) - homogenize(xy_dst[::8].transpose())), axis=0))) if (residual and len(xy_src)>8) else 0)
            except Exception as e:
                if not strict:
                    log.warning('[vipy.flow.stabilize]: coarse alignment failed with error "%s", returning original video "%s"' % (str(e), str(v)))
                    return v.clone().setattribute('unstabilized')  # for provenance
                raise

            # Fine alignment
            A = A.dot(M)  # update coarse reference frame
            imfine = im.clone().array(f_warp_coarse(im.numpy(), dst=np.zeros_like(imstabilized.numpy()), M=f_transform_coarse(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT), copy=True).objectmap(lambda o: o.projective(T.dot(A)))
            imfinemask = f_warp_coarse(np.ones_like(im.clone().greyscale().numpy()), dst=np.zeros_like(imstabilized.numpy()), M=f_transform_coarse(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT) > 0
            imfineflow = self.imageflow(imfine, imstabilized)
            (xy_src, xy_dst) = self._correspondence(imfineflow, imfine, border=None, dilate=dilate, contrast=contrast, validmask=imfinemask)
            try:
                F = f_estimate_fine(xy_src.transpose()-np.array([padwidth, padheight]), xy_dst.transpose()-np.array([padwidth, padheight]), method=cv2.RANSAC, confidence=0.99999, ransacReprojThreshold=0.1, refineIters=64, maxIters=3000)  
            except Exception as e:
                if not strict:
                    log.warning('[vipy.flow.stabilize]: fine alignment failed with error "%s", returning original video "%s"' % (str(e), str(v)))                    
                    return v.clone().setattribute('unstabilized')  # for provenance
                else:
                    raise ValueError('[vipy.flow.stabilize]: ERROR - fine alignment failed due to correspondence error')
        
            # Transform for interpolated rendering 
            A = F.dot(A)
            f_warp_fine(im.numpy(), dst=imstabilized._array, M=f_transform_fine(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT)
            vs.attributes['stabilize'][k] = A.copy()  # at stabilization framerate


        # Rendering: export video at source framerate
        with vs.stream(overwrite=True) as vss:      # Create write stream for stabilized video to avoid pre-allocating large video in memory
            transforms = list(vs.attributes['stabilize'].values())    # transform matrices at stabilization framerate
            imstabilized = vv.preview(0).rgb().zeropad(padwidth, padheight)  # single frame fetch
            imref = vv.preview(0).rgb().zeropad(padwidth, padheight)  # single frame fetch
            f_interpolate = lambda fi,R: (1-(fi-int(fi)))*R[int(fi)] + (fi-int(fi))*R[min(len(R)-1, int(fi)+1)]  # linear interpolation between transform matrices
            kref = None

            for (k,im) in enumerate(vic):
                ki = framerate*(k / vic.framerate())  # interpolated frame
                kr = min(int(round(ki)), len(transforms)-1)  # reference stabilization frame
                A = transforms[kr]  # reference stabilization transform

                # Refined alignment at source framerate
                if vic.framerate() != framerate and ki != kr:
                    if kref != kr:
                        f_warp_fine(vv.frame(kr).numpy(), dst=imref._array, M=f_transform_fine(T.dot(A)), dsize=(imref.width(), imref.height()), borderMode=cv2.BORDER_TRANSPARENT) 
                        kref = kr  # to avoid rewarping
                    Ai = f_interpolate(ki, transforms)  # interpolated transform matrix at source framerate
                    imfine = im.clone().array(f_warp_coarse(im.numpy(), dst=np.zeros_like(imref.numpy()), M=f_transform_coarse(T.dot(Ai)), dsize=(imref.width(), imref.height()), borderMode=cv2.BORDER_TRANSPARENT), copy=True).objectmap(lambda o: o.projective(T.dot(Ai)))
                    imfinemask = f_warp_coarse(np.ones_like(im.clone().greyscale().numpy()), dst=np.zeros_like(imref.numpy()), M=f_transform_coarse(T.dot(Ai)), dsize=(imref.width(), imref.height()), borderMode=cv2.BORDER_TRANSPARENT) > 0
                    imfineflow = self.imageflow(imfine, imref)  
                    (xy_src, xy_dst) = self._correspondence(imfineflow, imfine, border=None, dilate=dilate, contrast=contrast, validmask=imfinemask)
                    F = f_estimate_fine(xy_src.transpose()-np.array([padwidth, padheight]), xy_dst.transpose()-np.array([padwidth, padheight]), method=cv2.RANSAC, confidence=0.99999, ransacReprojThreshold=0.1, refineIters=64, maxIters=3000)  
                    A = F.dot(Ai)  # alignment of source frame to reference stabilization
                    
                f_warp_fine(im.numpy(), dst=imstabilized._array, M=f_transform_fine(T.dot(A)), dsize=(imstabilized.width(), imstabilized.height()), borderMode=cv2.BORDER_TRANSPARENT)
                im = im.objectmap(lambda o: o.projective(T.dot(A)))  # apply object transformation
                if any([not o.isvalid() for o in im.objects()]):  
                    if not strict:
                        log.warning('[vipy.flow.stabilize]: object alignment returned degenerate bounding box, returning original video "%s"' % str(v))
                        return v.clone().setattribute('unstabilized')  # for provenance
                    else:
                        raise ValueError('[vipy.flow.stabilize]: ERROR - object alignment returned degenerate bounding box for video "%s"' % str(v))                    
                vss.write( im.array(imstabilized.array()) )  # assign detections to tracks in stabilized video (vs) output

        return vs.setattribute('stabilize', {'mean residual':float(np.mean(r_coarse)), 'median residual':float(np.median(r_coarse))}) if residual else vs
