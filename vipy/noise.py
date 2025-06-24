import vipy
import numpy as np
from io import BytesIO
import PIL
import random
import math


vipy.util.try_import('torchvision.transforms', 'torchvision'); import torchvision.transforms
vipy.util.try_import('torchvision.transforms.v2', 'torchvision'); import torchvision.transforms.v2
vipy.util.try_import('skimage', 'scikit-image'); import skimage.transform
vipy.util.try_import('cv2', 'opencv-python'); import cv2


# GEOMETRIC
def left_shear(im, s, border='zero'):
    assert isinstance(im, vipy.image.Image)
    assert s >= 0
    
    T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
    A = np.dot(vipy.geometry.affine_transform(ky=s), T)
    A = np.dot(np.linalg.inv(T), A)
    return im.load().affine_transform(A, border=border).mat2gray(0,255)

def right_shear(im, s, border='zero'):
    assert isinstance(im, vipy.image.Image)
    assert s >= 0
    
    T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
    A = np.dot(vipy.geometry.affine_transform(ky=-s), T)
    A = np.dot(np.linalg.inv(T), A)
    return im.load().affine_transform(A, border=border).mat2gray(0,255)

def rotate(im, rad=None, deg=None, border='zero'):
    assert isinstance(im, vipy.image.Image)
    assert rad is not None or deg is not None
    
    r = rad if rad is not None else (deg*np.pi/180.0)
    T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
    A = np.dot(vipy.geometry.affine_transform(r=r), T)
    A = np.dot(np.linalg.inv(T), A)
    return im.load().affine_transform(A, border=border).mat2gray(0,255)

def barrel(im):
    assert isinstance(im, vipy.image.Image)    
    (width, height) = (im.width(), im.height())
    distCoeff = np.zeros((4,1),np.float64)
    
    k1 = 0.1 + 0.05*np.random.randn()
    k2 = 0.1 + 0.05*np.random.randn()
    p1 = 0.1 + 0.05*np.random.randn()
    p2 = 0.1 + 0.05*np.random.randn()
    
    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;
    
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 600.        # define focal length x
    cam[1,1] = 600.        # define focal length y

    # here the undistortion will be computed
    dst = cv2.undistort(im.load().array(),cam,distCoeff)

    return im.array(dst)

def left_swirl(im, center=None, strength=2):
    assert isinstance(im, vipy.image.Image)        
    return im.array(np.array(skimage.transform.swirl(im.load().array(), rotation=0, strength=strength, center=center, radius=im.maxdim())).astype(np.float32)).mat2gray(0,1)

def right_swirl(im, strength=2, center=None):
    assert isinstance(im, vipy.image.Image)            
    return im.array(np.array(skimage.transform.swirl(im.load().fliplr().array(), rotation=0, center=center, strength=strength, radius=im.maxdim())).astype(np.float32)).mat2gray(0,1).fliplr()

def horizontal_mirror(im):
    assert isinstance(im, vipy.image.Image)            
    img = im.load().array()
    (i,j) = (im.width()//2 if vipy.math.iseven(im.width()) else (im.width()//2)+1, im.width()//2)    
    img[:,i:] = np.fliplr(img[:,0:j])
    return im.array(img)

def vertical_mirror(im):
    assert isinstance(im, vipy.image.Image)            
    img = im.load().array()
    (i,j) = (im.height()//2 if vipy.math.iseven(im.height()) else (im.height()//2)+1, im.height()//2)
    img[i:,:] = np.flipud(img[0:j, :])
    return im.array(img)

def vertical_motion_blur(im, kernel_size):
    assert isinstance(im, vipy.image.Image)
    kernel_v = np.zeros((kernel_size, kernel_size) )  
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)  
    kernel_v /= kernel_size
    return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_v)).mat2gray(0,255).colorspace('float')

def horizontal_motion_blur(im, kernel_size):
    assert isinstance(im, vipy.image.Image)    
    kernel_h = np.zeros((kernel_size, kernel_size) )  
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_h /= kernel_size
    return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_h)).mat2gray(0,255).colorspace('float')

def ghost(im, txr=5, tyr=5, txg=-5, tyg=-5, txb=5, tyb=-5):
    assert isinstance(im, vipy.image.Image)
    (imc,imz) = (im.clone().load(), im.clone().load().zeros())
    img = imz.numpy()
    img[:,:,0] = translate(imc.red(), txr, tyr).load().array()
    img[:,:,1] = translate(imc.green(), txg, tyg).load().array()
    img[:,:,2] = translate(imc.blue(), txb, tyb).load().array()
    return imz

def crop(im, s=0.8):
    return im.crop(vipy.geometry.BoundingBox(xmin=random.randint(0, int((1-s)*im.width())), ymin=random.randint(0, int((1-s)*im.height())), width=math.ceil(s*im.width()), height=math.ceil(s*im.height()))).resize(height=im.height(), width=im.width())

def fliplr(im):
    return im.load().fliplr()

def flipud(im):
    return im.load().flipud()

def rot90cw(im):
    return im.load().rot90cw()

def rot90ccw(im):
    return im.load().rot90ccw()

def translate(im, dx, dy): 
    """Translate by (dx,dy) pixels, with zero border handling"""
    assert isinstance(im, vipy.image.Image)
    return im.load().padcrop(im.imagebox().translate(dx,dy))

def isotropic_scale(im, s, border='zero'):
    assert isinstance(im, vipy.image.Image)
    assert s > 0

    (W,H) = (im.width(), im.height())
    return im.load().rescale(s).padcrop(vipy.geometry.BoundingBox(centroid=(im.width()//2, im.height()//2), width=W, height=H))

def zoom(im, s, zx=None, zy=None, border='zero'):
    assert isinstance(im, vipy.image.Image)
    assert s > 0

    (zx, zy) = ((-im.width()/2) if zx is None else -zx, (-im.height()/2) if zy is None else -zy)    
    return isotropic_scale(translate(im, zx, zy), s)


# <PHOTOMETRIC>
def blur(im, sigma):
    assert isinstance(im, vipy.image.Image)
    imblur = im.rgb().blur(sigma=sigma)
    return imblur.greyscale().colorspace_like(im) if im.channels() == 1 else imblur.colorspace_like(im)

def salt_and_pepper(im, p):
    assert isinstance(im, vipy.image.Image)    
    assert p >= 0 and p <= 1
    
    W = np.array(np.random.rand( im.height(), im.width(), 1 ) > p).astype(np.float32)
    B = np.array(np.random.rand( im.height(), im.width(), 1 ) > p).astype(np.float32)    
    return im.load().float().mat2gray().map(lambda x: np.multiply(x, W) + 1-W).map(lambda x: np.multiply(x, B)).colorspace_like(im)

def jpeg_compression(im, quality):
    assert isinstance(im, vipy.image.Image)
    assert quality >= 0 and quality <= 95
    
    out = BytesIO()
    im.load().pil().save(out, format='jpeg', quality=quality)
    out.seek(0)
    return im.load().array(np.array(PIL.Image.open(out))).mat2gray(0,255)

def bit_depth(im, d):
    assert d>=1 and d<=8    
    return im.array(np.array(PIL.ImageOps.posterize(im.load().pil(), d)))

def solarize(im, t):
    assert t>=0 and t<=255    
    return im.array(np.array(PIL.ImageOps.solarize(im.load().pil(), t)))

def permute_color_channels(im, order):
    assert sorted(order) == [0,1,2]
    return im.array(np.array(im.load().rgb())[:,:,order])

def greyscale(im):
    assert isinstance(im, vipy.image.Image)    
    return im.clone().load().greyscale().colorspace_like(im)

def bgr(im):
    assert isinstance(im, vipy.image.Image)    
    return im.array(im.bgr().load().array()).colorspace('rgb')

def hot(im):
    assert isinstance(im, vipy.image.Image)    
    return im.hot().load().rgb()

def rainbow(im):
    assert isinstance(im, vipy.image.Image)    
    return im.rainbow().load().rgb()

def saturate(im, low, high):
    assert isinstance(im, vipy.image.Image)    
    return im.saturate(low, high)

def colorjitter(im):
    transform = torchvision.transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))
    return im.array(np.array(transform(im.pil())))

def sharpness(im, f):
    return im.array(np.array(torchvision.transforms.v2.functional.adjust_sharpness(im.load().torch('CHW'), f).permute(1,2,0)))

def gamma(im, f):
    return im.array(np.array(torchvision.transforms.v2.functional.adjust_gamma(im.load().torch('CHW'), f).permute(1,2,0)))

def autocontrast(im):
    return im.mat2gray().rgb()

def edge(im):
    assert isinstance(im, vipy.image.Image)    
    return im.array(np.array(im.luminance().pil().filter(PIL.ImageFilter.FIND_EDGES))).colorspace('lum').colorspace_like(im)

def emboss(im): 
    assert isinstance(im, vipy.image.Image)   
    return im.array(np.array(im.luminance().pil().filter(PIL.ImageFilter.EMBOSS))).colorspace('lum').colorspace_like(im)

def darken(im, g): 
    assert isinstance(im, vipy.image.Image)   
    return im.gain(g).mat2gray(0, 255).colorspace('float')

def negative(im):
    assert isinstance(im, vipy.image.Image)    
    return im.load().array(1-im.load().mat2gray(0,255).array()).colorspace('float')

def scan_lines(im):
    assert isinstance(im, vipy.image.Image)
    imc = im.load()
    imc._array[::2, :] = 0
    return imc

def additive_gaussian_noise(im, sigma=0.1):
    assert isinstance(im, vipy.image.Image)            
    return im.float().map(lambda img: img + max(sigma*im.maxpixel(), sigma)*np.random.randn(img.shape[0], img.shape[1], im.channels()).astype(np.float32)).saturate(0,255).mat2gray(0,255).colorspace('float')



class Noise():
    def __init__(self, magnitude=0.25, provenance=False, register=None, deregister=['hot','rainbow','vertical_mirror','flipud','rot90cw','rot90ccw']):

        n = magnitude
        (shear, dx, dy, rot, z, border, sigma, gain) = (n*0.25, n*0.15, n*0.15, n*25, n*0.2, 'replicate', 1.0, 0.5)
        
        self._provenance = provenance        
        self._registry = {'blur': lambda im: blur(im, sigma=n*random.uniform(im.mindim()*0.01, im.mindim()*0.05)),
                          'salt_and_pepper': lambda im: salt_and_pepper(im, n*0.1*random.random()),
                          'jpeg_compression': lambda im: jpeg_compression(im, random.randint(50,95-int(10*n))),
                          'greyscale': lambda im: greyscale(im),
                          'bgr': lambda im: bgr(im),
                          'hot': lambda im: hot(im),
                          'rainbow': lambda im: rainbow(im),
                          'saturate': lambda im: saturate(im, random.randint(0, 64), random.randint(255-64, 255)),
                          'edge': lambda im: edge(im),
                          'emboss': lambda im: emboss(im),
                          'darken': lambda im: darken(im, 1-n*random.random()),
                          'negative': lambda im: negative(im),
                          'scan_lines': lambda im: scan_lines(im),
                          'additive_gaussian_noise': lambda im: additive_gaussian_noise(im, np.random.uniform(n*0.01, n*0.5)),
                          'translate': lambda im, dx=dx, dy=dy, b=border: translate(im, np.random.uniform(-dx,dx), np.random.uniform(-dy,dy)),                          
                          'horizontal_motion_blur': lambda im: horizontal_motion_blur(im, random.randint(math.ceil(im.mindim()*0.05*n), math.ceil(im.mindim()*0.15*n))),
                          'vertical_motion_blur': lambda im: vertical_motion_blur(im, random.randint(math.ceil(im.mindim()*0.05*n), math.ceil(im.mindim()*0.15*n))),                          
                          'barrel': lambda im: barrel(im),
                          'left_swirl': lambda im: left_swirl(im, center=(random.randint(0, im.mindim()), random.randint(0,im.mindim())), strength=n),
                          'right_swirl': lambda im: right_swirl(im, center=(random.randint(0, im.mindim()), random.randint(0, im.mindim())), strength=n),
                          'horizontal_mirror': lambda im: horizontal_mirror(im),
                          'vertical_mirror': lambda im: vertical_mirror(im),
                          'left_shear': lambda im, s=shear, b=border: left_shear(im, np.random.uniform(0,s), border=b),
                          'right_shear': lambda im, s=shear, b=border: right_shear(im, np.random.uniform(0,s), border=b),
                          'rotate': lambda im, deg=rot, b=border: rotate(im, deg=np.random.uniform(-deg,deg), border=b),
                          'isotropic_scale': lambda im, z=z, b=border: isotropic_scale(im, np.random.uniform(1.0-z, 1.0+z)),
                          'ghost': lambda im: ghost(im,
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n)),
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n)),
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n)),
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n)),
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n)),
                                                    random.randint(-int(im.mindim()*0.05*n),int(im.mindim()*0.05*n))),
                          'bit_depth': lambda im: bit_depth(im, random.randint(1,7)),
                          'permute_color_channels': lambda im: permute_color_channels(im, random.sample([[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]], 1)[0]),
                          'fliplr': lambda im: fliplr(im),
                          'flipud': lambda im: flipud(im),
                          'rot90cw': lambda im: rot90cw(im),
                          'rot90ccw': lambda im: rot90ccw(im),
                          'solarize': lambda im: solarize(im, random.randint(32, 255-32)),
                          'colorjitter': lambda im: colorjitter(im),
                          'autocontrast': lambda im: autocontrast(im),
                          'sharpness': lambda im: sharpness(im, 10*random.random()),
                          'gamma': lambda im: gamma(im, random.random()+0.5),
                          'crop': lambda im: crop(im, 1-0.2*random.random()),
                          }

        if deregister is not None:
            self._registry = {k:v for (k,v) in self._registry.items() if k not in deregister}                    
        if register is not None:
            self._registry = {k:v for (k,v) in self._registry.items() if k in register}        

    def transformations(self):
        return list(self._registry.keys())

    def transform(self, im, transform):
        assert isinstance(im, vipy.image.Image)
        assert transform in self.transformations()

        imd = self._registry[transform](im.clone().load())
        if self._provenance:
            imd.setattribute('vipy.noise', transform)
        return imd

    def montage(self, im, num_transforms=None):
        """Return a montage of noise applied to the input image. This is useful for visualization of the types of noise applied to a given image

        Args:
           im [`vipy.image.Image`]: the input image
           num_transforms [None|int]: if None, return a montage where each element is one transform, if int, return a montage randomly selecting num_transforms trannsforms
        
        Returns:
           `vipy.image.Image` montage as returned from `vipy.visualize.montage`.  Try show() on this returned image.

        """
        transforms = self.transformations() if num_transforms is None else random.choices(self.transformations(), k=num_transforms)
        return vipy.visualize.montage([self.transform(im.clone().centersquare(), k) for k in transforms], 256, 256)
    
    def __call__(self, im):
        assert isinstance(im, vipy.image.Image), "vipy.image.Image required"
        return self.transform(im, random.choice(self.transformations()))


class RandomCrop(Noise):
    def __init__(self, magnitude=1, provenance=False):
        super().__init__(magnitude=magnitude, provenance=provenance, register=['translate', 'isotropic_scale', 'crop'])
    
class Geometric(Noise):
    def __init__(self, magnitude=0.25, provenance=False):
        super().__init__(magnitude=magnitude, provenance=provenance,
                         register=['translate', 'horizontal_motion_blur', 'vertical_motion_blur', 'barrel',
                                   'left_swirl', 'right_swirl','horizontal_mirror','left_shear','right_shear',
                                   'rotate','isotropic_scale','fliplr','crop'])
class Photometric(Noise):
    def __init__(self, magnitude=0.25, provenance=False):
        super().__init__(magnitude=magnitude, provenance=provenance,
                         register=['blur', 'salt_and_pepper', 'jpeg_compression', 'greyscale', 'bgr', 'hot','saturate','edge','emboss',
                                   'darken','negative','scan_lines','additive_gaussian_noise','bit_depth','permute_color_channels','solarize',
                                   'colorjitter','autocontrast', 'sharpness', 'gamma', 'ghost'])
class Perturbation(Noise):
    def __init__(self, magnitude=0.5, provenance=False):
        super().__init__(magnitude=magnitude, provenance=provenance, register=['translate', 'isotropic_scale', 'crop', 'blur', 'additive_gaussian_noise']) 
        

geometric = Geometric(provenance=True)
photometric = Photometric(provenance=True)
perturbation = Perturbation(provenance=True)
randomcrop = RandomCrop(provenance=True)

