import vipy
import numpy as np
import copy
import math

vipy.util.try_import('torch')
import torch



class GaussianPyramid():
    """vipy.pyramid.GaussianPyramid() class"""
    def __init__(self, im=None, tensor=None):
        assert im is not None or tensor is not None
        assert im is None or isinstance(im, vipy.image.Image)
        assert tensor is None or (torch.is_tensor(tensor) and tensor.ndim == 4)
        
        g = ((1.0/np.sqrt(2*np.pi))*np.exp(-0.5*(np.array([-2,-1,0,1,2])**2))).astype(np.float32)
        G = torch.from_numpy(np.outer(g,g)).repeat( (3,1,1,1) )
        
        self._G = G
        self._band = []
        self._pad = torch.nn.ReflectionPad2d(2)
        
        x = im.float().torch(order='NCHW') if im is not None else tensor
        for k in range(int(np.log2(min(x.shape[2], x.shape[3]))-2)):
            y = torch.nn.functional.conv2d(self._pad(x), self._G, groups=3)  # anti-aliasing
            self._band.append(y)  # lowpass
            x = y[:,:,::2,::2]  # downsample
        self._band.append(x)  # lowpass

    def __len__(self):
        return len(self._band)

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]
    
    def __getitem__(self, k):
        assert k>=0 and k<len(self)
        return vipy.image.Image.fromtorch(self._band[k])

    def band(self, k):
        return self[k]

    def show(self, mindim=256):
        return vipy.visualize.montage([im.maxsquare() for im in self], mindim, mindim).show()

    
class LaplacianPyramid():
    """vipy.pyramid.LaplacianPyramid() class"""    
    def __init__(self, im, pad='zero'):
        
        g = (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*(np.array([-2,-1,0,1,2])**2))
        G = torch.from_numpy(np.outer(g,g).astype(np.float32))

        self._G = G.repeat( (3,1,1,1) )
        self._Ce = torch.sum(G[::2,::2])  # filter coefficient sum, even elements only
        self._Co = torch.sum(G[1::2,1::2]) # filter coefficient sum, odd elements only
        self._band = []
        self._im = im
        
        if pad == 'zero':
            self._pad = torch.nn.ZeroPad2d(2)  # introduces lowpass corner boundary artifact on reconstruction
            self._gain = 1.6  # rescale to approximately correct boundary artifact
        elif pad == 'reflect':
            self._pad = torch.nn.ReflectionPad2d(2)  # introduces lowpass gain boundary artifact on reconstruction
            self._gain = 1.4  # rescale to approximately correct boundary artifact
        else:
            raise ValueError('unknown padding "%s" - must be ["zero", "reflect"]' % pad)

        if isinstance(im, vipy.image.Image):
            x = im.float().torch(order='NCHW')
            for k in range(int(np.log2(im.mindim())-1)):
                y = torch.nn.functional.conv2d(self._pad(x), self._G, groups=3)  # anti-aliasing
                self._band.append(x-y)  # bandpass
                x = y[:,:,::2,::2]  # even downsample
            self._band.append(x)  # lowpass
        elif isinstance(im, list):
            assert all([torch.is_tensor(b) for b in im])
            self._band = im
            self._im = self[0].clone().zeros()  # no source image available
        else:
            raise ValueError('invalid input')

    def __repr__(self):
        return '<vipy.pyramid.LaplacianPyramid: scales=%d, channels=%d, height=%d, width=%d>' % (self.scales(), self.channels(), self.height(), self.width())
    
    def __len__(self):
        return len(self._band)

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]
            
    def __getitem__(self, k):
        assert k>=0 and k<len(self)
        return vipy.image.Image.fromtorch(self._band[k])

    def band(self, k):
        return self[k]

    def reconstruct(self):
        x = self._band[-1]
        for b in reversed(self._band[0:-1]):
            xz = torch.zeros(1, b.shape[1], b.shape[2], b.shape[3])
            xz[:,:,::2,::2] = x  # odd zero interpolate (even signal)
            xu = torch.nn.functional.conv2d(self._pad(xz), (1.0/self._Ce)*self._G, groups=3)  # upsample, rescale using sum of non-zeroed kernel elements for even locations
            xu[:,:,1::2,1::2] *= (self._Ce/self._Co)  # upsample, rescale using sum of non-zeroed kernel elements for odd locations
            x = xu + b  # reconstruction
        im = vipy.image.Image.fromtorch((x * self._gain).clamp(0, 255))  # final rescale due to padding boundary artifact (can also use im.mat2gray)
        return im.array(im.numpy().astype(np.uint8)).colorspace('rgb')

    def show(self, mindim=256):
        return vipy.visualize.montage([im.maxsquare().resize(mindim, mindim, interp='nearest') for im in self], mindim, mindim).show()

    def height(self):
        return self._im.height()
    def width(self):
        return self._im.width()
    def channels(self):
        return self._im.channels()
    def scales(self):
        return len(self)
    
    def tensor(self, interp='nearest'):
        return torch.stack([im.resize(self.height(), self.width(), interp=interp).torch(order='CHW') for im in self])  # scales() x channels() x height() x width()

    @staticmethod
    def fromtensor(bands):
        """Convert a S*CxHxW torch tensor back to LaplacianPyramid"""
        assert torch.is_tensor(bands) and bands.ndim == 4
        (S,C,H,W) = bands.shape
        return LaplacianPyramid([vipy.image.Image.fromtorch(b).resize(H//(2**i), W//(2**i), interp='nearest').torch(order='NCHW') for (i,b) in enumerate(bands)])

    
class Foveation(LaplacianPyramid):
    def __init__(self, im, mode='log-circle', s=None):        
        super().__init__(im)

        (H,W) = (im.height(), im.width())
        allowable_modes = ['gaussian', 'linear-circle', 'linear-square', 'log-circle']
        if mode == 'gaussian':
            s = s*(W/2) if s is not None else W/2            
            G = vipy.math.gaussian2d([W,H], [s,s], 2*H, 2*W)
            M = np.repeat(G[:,:,np.newaxis], 3, axis=2)
            thresholds = np.arange(0, float(np.max(G)), np.max(G)/len(self))
            masks = [vipy.image.Image(array=255*np.array(M>=t).astype(np.uint8)).blur(16).mat2gray(min=0) for t in thresholds]
        elif mode == 'linear-circle':
            s = s*2.0 if s is not None else 2.0
            masks = [vipy.calibration.imcircle(W, H, s*(d/2), 2*W, 2*H, 3).rgb().blur(16).mat2gray(min=0) for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        elif mode == 'log-circle':
            s = s*0.125 if s is not None else 0.125                        
            masks = [vipy.calibration.imcircle(W, H, (s*(d/2))**2, 2*W, 2*H, 3).rgb().blur(16).mat2gray(min=0) for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        elif mode == 'linear-square':
            s = s*2.0 if s is not None else 2.0                        
            masks = [vipy.image.Image(array=vipy.calibration.square(W,H,s*(d/2),2*W,2*H,3).astype(np.float32), colorspace='float') for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        else:
            raise ValueError('invalid mode "%s" - must be in %s' % (mode, str(allowable_modes)))
        self._immasks = masks
        self._masks = [m.torch(order='NCHW') for m in masks]

    def __call__(self, tx=0, ty=0, sx=1.0, sy=1.0):
        (C, H,W) = (self.channels(), self.height(), self.width())        
        theta = torch.FloatTensor([[sx,0,tx],[0,sy,ty]]).repeat( (1, 1, 1) )
        G = torch.nn.functional.affine_grid(theta, (1, C, H, W), align_corners=False)
        blend = [GaussianPyramid(tensor=torch.nn.functional.grid_sample(m, G, align_corners=False)) for m in self._masks]        

        pyr = copy.deepcopy(self)
        pyr._band[:-1] = [torch.mul(w[k].torch(order='NCHW'), b) for (k, (w,b)) in enumerate(zip(reversed(blend), pyr._band[:-1]))]
        return pyr.reconstruct()
        
    def foveate(self, tx=0, ty=0):
        """Foveate the input image at location (tx, ty) in scaled image coordinates where (0,0) is the center and (1,1) is the upper left"""        
        return self.__call__(tx=tx, ty=ty, sx=1.0, sy=1.0)

    def visualize(self):
        """Show the fovea density"""
        imgmask = self._immasks[0].clone().mat2gray().numpy()
        for (im,c) in zip(self._immasks, range(0, 255, len(self))):
            img = im.clone().mat2gray(min=0).numpy()
            imgmask += c*img
        return vipy.image.Image(array=imgmask)

    
class SteerablePyramid():
    def __init__(self, im):
        vipy.util.try_import('pyrtools')
        import pyrtools
        
        assert isinstance(im, vipy.image.Image)
        assert im.mindim() >= 32        
        self._channels = [pyrtools.pyramids.SteerablePyramidFreq(imc.load().array().astype(float), height='auto', order=3) for imc in im.channel()]                
        
    @property
    def num_scales(self):
        return self._channels[0].num_scales

    @property
    def num_orientations(self):
        return self._channels[0].num_orientations

    @property
    def num_channels(self):
        return len(self._channels)           
        
    def bandpass(self, channel):
        return [self._channels[channel].pyr_coeffs[(s,o)] for s in range(self.num_scales) for o in range(self.num_orientations)]
    
    def lowpass(self, channel):
        return self._channels[channel].pyr_coeffs['residual_lowpass']

    def highpass(self, channel):
        return self._channels[channel].pyr_coeffs['residual_highpass']        
    
    def synthesis(self):        
        return vipy.image.Image(array=np.stack([pyr.recon_pyr() for pyr in self._channels], axis=2).astype(np.float32)).mat2gray()

    def multichannel(self):
        """a multichannel image is an image of the same shape as the input, but with channels from pyramid decomposition.  Coefficients are resized using bilinear interpolation."""
        (H,W) = self.highpass(0).shape
        resizer = lambda x: np.array(torch.nn.functional.interpolate(torch.tensor(x).view(1,1,x.shape[0],x.shape[1]), size=(H,W), mode='bilinear')).squeeze()            
        return vipy.image.Image(array=np.stack([resizer(x) for c in range(self.num_channels) for x in self.bandpass(c)+[self.lowpass(c),self.highpass(c)]], axis=2).astype(np.float32))

    def montage(self):
        """scales by row, orientations by col, channels merged back into color image, last image is lowpass"""
        imlist = [vipy.image.Image(array=np.stack([self._channels[c].pyr_coeffs[(s,o)] for c in range(self.num_channels)], axis=2).astype(np.float32)) for s in range(self.num_scales) for o in range(self.num_orientations)]
        imlist += [vipy.image.Image(array=np.stack([self.lowpass(c) for c in range(self.num_channels)], axis=2).astype(np.float32))]
        imlist += [vipy.image.Image(array=np.stack([self.highpass(c) for c in range(self.num_channels)], axis=2).astype(np.float32))]        
        return vipy.visualize.montage([im.mat2gray().rgb() for im in imlist], gridrows=self.num_scales+1, gridcols=self.num_orientations)

    

class BatchSteerablePyramid():
    def __init__(self, height, width, channels, device='cpu', orientations=4):
        vipy.util.try_import('plenoptic')
        import plenoptic as po

        assert height >= 32 and width >= 32        
        self._pyr = po.simulate.SteerablePyramidFreq(height='auto', image_shape=[height, width], twidth=1, order=orientations-1, downsample=False, is_complex=True).to(device)
        self._pyr.eval()
        self._imheight = height
        self._imwidth = width
        self._imchannels = channels
        self._device = device
        self._pyr_info = None
        self._tensor = None
        
    def device(self, d=None):
        if d is not None:
            self._device = d  # check device in self._pyr._buffers['lo0mask'], or just store here
            self._pyr = self._pyr.cpu() if d == 'cpu' else self._pyr.to(d)
            return self
        return self._device
                
    @property
    def num_scales(self):
        return self._pyr.num_scales

    @property
    def num_orientations(self):
        return self._pyr.num_orientations

    @property
    def num_channels(self):
        return (self._pyr.num_orientations*self._pyr.num_scales + 2)*self._imchannels

    def to(self, dev):
        return self.device(dev)    

    def forward(self, x):
        assert (torch.is_tensor(x) and x.ndim == 4) or isinstance(x, vipy.image.Image) or (isinstance(x, (list, tuple)) and all([isinstance(im, vipy.image.Image) for im in x]))

        x = torch.stack([im.torch('CHW') for im in vipy.util.tolist(x)]) if not torch.is_tensor(x) else x
        assert x.shape[2] == self._imheight and x.shape[3] == self._imwidth, "wrong input shape"
        assert x.shape[1] == self._imchannels, "wrong input channels"

        with torch.no_grad():        
            (self._tensor, self._pyr_info) =  self._pyr.convert_pyr_to_tensor(self._pyr.forward(x.to(self.device())))
        return self

    def forwarded(self):
        return self._tensor is not None
    
    def tensor(self):
        """a pyramid tensor is an NxCxHxW tensor (same shape as the input), but with channels from complex (even, odd) pyramid coefficients
        The first channel will be the residual highpass and the last will be the residual lowpass. Each band is then a separate channel in (scale, orientation) order

        input is NxCxHxW tensor
        """
        assert self.forwarded() 
        return self._tensor

    def band(self, scale, orientation):
        assert scale <= self.num_scales and orientation <= self.num_orientations
        return self.tensor()[:, 1+scale*self.num_orientations + orientation,:,:]

    def magnitude(self):
        """return magnitude component of complex steerable pyramid"""
        return self.tensor().abs()

    def phase(self):
        """return phase component of complex steerable pyramid"""
        return self.tensor().angle()        
    
    def montage(self, x):
        """return a montage visualization of the pyramid, scales by row, orientations by col, channels merged back into color image, last row is highpass and lowpass"""
        assert self.forwarded()
        assert torch.is_tensor(x) and x.ndim == 4 and x.shape == self._tensor.shape
        
        (C,N) = (x.shape[1], x.shape[1]//self._imchannels)
        return vipy.visualize.montage([vipy.image.Image.fromtorch(x[0,j::N,:,:]).mat2gray().rgb() for j in list(range(1,N-1))+[0,N-1]],
                                       gridrows=self.num_scales+1, gridcols=self.num_orientations)                

        
    def synthesis(self, x):
        """Generate a synthesis of the multichannel tensor x"""
        assert self.forwarded()
        assert torch.is_tensor(x) and x.ndim == 4 and x.shape == self._tensor.shape

        img = self._pyr.recon_pyr(self._pyr.convert_tensor_to_pyr(x, pyr_keys=self._pyr_info[2], num_channels=self._pyr_info[0], split_complex=self._pyr_info[1]))
        return [vipy.image.Image.fromtorch(t) for t in img]  # one image per batch element

    
    def phase_congruency(self, eps=1E-6):
        (M,P) = (self.magnitude()[:,1:-1,:,:], self.phase()[:,1:-1,:,:])
        return torch.sum( M * torch.cos(P - ((M*P).sum(dim=1, keepdims=True) / (eps + M.sum(dim=1, keepdims=True)))), dim=1, keepdims=True) / (eps + torch.sum(M, dim=1, keepdims=True))

    def zero_crossing(self, eps=1E-6):
        (M,P) = (self.magnitude()[:,1:-1,:,:], self.phase()[:,1:-1,:,:])
        return torch.sum( M*torch.nn.functional.relu(torch.cos(P)), dim=1, keepdims=True)
    
