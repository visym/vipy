import vipy
import numpy as np
import copy

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
