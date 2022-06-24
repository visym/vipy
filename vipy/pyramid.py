import vipy
import numpy as np
import copy
import os
import random
import dill
import time
import json

import vipy.util
from vipy.util import try_import
try_import('torch');

import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split


class GaussianPyramid(object):
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

    
class LaplacianPyramid(object):
    """vipy.pyramid.LaplacianPyramid() class"""    
    def __init__(self, im):

        g = ((1.0/np.sqrt(2*np.pi))*np.exp(-0.5*(np.array([-2,-1,0,1,2])**2))).astype(np.float32)
        G = torch.from_numpy(np.outer(g,g)).repeat( (3,1,1,1) )
        
        self._G = G
        self._band = []
        self._pad = torch.nn.ReflectionPad2d(2)
        self._im = im
        
        x = im.float().torch(order='NCHW')
        for k in range(int(np.log2(im.mindim())-2)):
            y = torch.nn.functional.conv2d(self._pad(x), self._G, groups=3)  # anti-aliasing
            self._band.append(x-y)  # bandpass
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
    
    def reconstruct(self):
        x = self._band[-1]
        for b in reversed(self._band[0:-1]):
            xu = torch.zeros(1, 3, b.shape[2], b.shape[3])
            xu[:,:,::2,::2] = x
            x = torch.nn.functional.conv2d(self._pad(xu), 4*self._G, groups=3) + b
            x[:,:,1::2,1::2] *= float(np.sqrt(5/4))  # compensate for odd padding
        im = vipy.image.Image.fromtorch(x.clamp(0, 255))
        return im.array(im.numpy().astype(np.uint8)).colorspace('rgb')

    def show(self, mindim=256):
        return vipy.visualize.montage([im.maxsquare().resize(mindim, mindim, interp='nearest') for im in self], mindim, mindim).show()
    

class Foveation(LaplacianPyramid):
    def __init__(self, im, s=0.125, mode='log-circle'):
        super().__init__(im)
        
        (H,W) = (im.height(), im.width())
        allowable_modes = ['gaussian', 'linear-circle', 'linear-square', 'log-circle']
        if mode == 'gaussian':
            G = np.repeat(vipy.math.gaussian2d([W,H], [s,s], 2*H, 2*W)[:,:,np.newaxis], 3, axis=2)
            masks = [vipy.image.Image(array=np.array(G>t).astype(np.float32), colorspace='float') for t in np.arange(0, np.max(G), np.max(G)/len(self))]
        elif mode == 'linear-circle':
            masks = [vipy.image.Image(array=vipy.calibration.circle(W,H,s*(d/2),2*W,2*H,3).astype(np.float32), colorspace='float') for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        elif mode == 'log-circle':
            masks = [vipy.image.Image(array=vipy.calibration.circle(W,H,(s*(d/2))**2,2*W,2*H,3).astype(np.float32), colorspace='float') for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        elif mode == 'linear-square':
            masks = [vipy.image.Image(array=vipy.calibration.square(W,H,s*(d/2),2*W,2*H,3).astype(np.float32), colorspace='float') for d in np.arange(max(H,W), 0, -max(H,W)/len(self))]
        else:
            raise ValueError('invalid mode "%s" - must be in %s' % (mode, str(allowable_modes)))
        self._immasks = masks
        self._masks = [m.torch(order='NCHW') for m in masks]

    def __call__(self, tx=0, ty=0, sx=1.0, sy=1.0):
        (H,W) = (self._im.height(), self._im.width())        
        theta = torch.FloatTensor([[sx,0,tx],[0,sy,ty]]).repeat( (1, 1, 1) )
        G = torch.nn.functional.affine_grid(theta, (1, 3, H, W), align_corners=False)
        blend = [GaussianPyramid(tensor=torch.nn.functional.grid_sample(m, G, align_corners=False)) for m in self._masks]        

        pyr = copy.deepcopy(self)
        pyr._band[:-1] = [torch.mul(w[k].torch(order='NCHW'), b) for (k, (w,b)) in enumerate(zip(reversed(blend), pyr._band[:-1]))]
        return pyr.reconstruct()
        
    def foveate(self, tx=0, ty=0, sx=1.0, sy=1.0):
        return self.__call__(tx=tx, ty=ty, sx=sx, sy=sy)
