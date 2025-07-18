import vipy
import numpy as np
from io import BytesIO
import PIL
import random
import math
import functools


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
    center = center if center is not None else (im.width()*center[0], im.height()*center[1])
    return im.array(np.array(skimage.transform.swirl(im.load().array(), rotation=0, strength=strength, center=center, radius=im.maxdim())).astype(np.float32)).mat2gray(0,1)

def right_swirl(im, strength=2, center=None):
    assert isinstance(im, vipy.image.Image)
    center = center if center is not None else (im.mindim()*center[0], im.mindim()*center[1])    
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
    kernel_size = int(im.mindim()*kernel_size)    
    kernel_v = np.zeros((kernel_size, kernel_size) )  
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)  
    kernel_v /= kernel_size
    return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_v)).mat2gray(0,255).colorspace('float')

def horizontal_motion_blur(im, kernel_size):
    assert isinstance(im, vipy.image.Image)
    kernel_size = int(im.mindim()*kernel_size)
    kernel_h = np.zeros((kernel_size, kernel_size) )  
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_h /= kernel_size
    return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_h)).mat2gray(0,255).colorspace('float')

def ghost(im, txr=0.01, tyr=0.01, txg=-0.01, tyg=-0.01, txb=0.01, tyb=-0.01):
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
    """Translate by (dx,dy) normalized pixels, with zero border handling"""
    assert isinstance(im, vipy.image.Image)    
    return im.load().padcrop(im.imagebox().translate(int(dx*im.mindim()), int(dy*im.mindim())))

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
def mask(im, num_masks=1, xywh_range=((0.1,0.9),(0,1,0.9),(0.3,0.5),(0.3,0.5)), mask_type='zeros', radius=7):
    """Introduce one or more rectangular masks filled by mask_type.

    The position and size of masks are uniformly sampled from the provided range of xywh = (xmin, ymin, width, height)
    
    xywh_range = ((xmin_lowerbound, xmin_upperbound), (ymin_lb,ymin_ub), (width_lb,width_ub), (height_lb,height_ub))) relative to the normalized height and width

    Allowable mask_types = ['zeros', 'inverse_zeros', 'mean', 'blur', 'pixelize', 'inverse_mean', 'inverse_blur']
    
    - zeros: all masks are replaced with zeros
    - inverse_zeros: all pixels outside masks are replaced with zeros
    - mean: all pixels inside masks are replaced with the mean pixel
    - inverse_mean: all pixels outside masks are replaced with the mean pixel    
    - blur: all pixels inside masks are replaced with blurred pixels with a given gaussian radius
    - inverse_blur: all pixels outside masks are replaced with blurred pixels with a given gaussian radius
    - pixelize: all pixels inside masks are replaced with low resolution pixels with a given downscale pixel radius

    Radius is specific to blur and pixelize masks only and ignored for others
    """
    (H,W) = (im.height(), im.width())
    masks = [vipy.object.Detection(category='mask%d' % k,
                                   xmin=np.random.randint(xywh_range[0][0]*W, xywh_range[0][1]*W),
                                   ymin=np.random.randint(xywh_range[1][0]*H, xywh_range[1][1]*H),
                                   width=np.random.randint(xywh_range[2][0]*W, xywh_range[2][1]*W),
                                   height=np.random.randint(xywh_range[3][0]*H, xywh_range[3][1]*H))
             for k in range(num_masks)]
    im = vipy.image.Scene.cast(im.clone()).objects(masks)

    if mask_type == 'zeros':
        im = im.fgmask()
    elif mask_type == 'inverse_zeros':
        im = im.bgmask()
    elif mask_type == 'mean':
        im = im.mean_mask()
    elif mask_type == 'inverse_mean':
        im = im.inverse_mean_mask()
    elif mask_type == 'blur':
        im = im.blur_mask(radius=radius)
    elif mask_type == 'inverse_blur':
        im = im.inverse_blur_mask(radius=radius)
    elif mask_type in ['pixel', 'pixelize', 'pixelate']:
        im = im.pixel_mask(radius=radius)
    else:
        raise ValueError("unknown mask type '%s'" % mask_type)
    return im


def blur(im, sigma):
    assert isinstance(im, vipy.image.Image)
    imblur = im.rgb().blur(sigma=sigma*im.mindim())
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
    """All pixels above threshold are inverted"""
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
        (shear, dx, dy, deg, z, border, sigma, gain) = (n*0.25, n*0.15, n*0.15, n*25, n*0.2, 'replicate', 1.0, 0.5)
        
        self._provenance = provenance        
        self._registry = {'blur': functools.partial(blur, sigma=n*random.uniform(0.01, 0.05)),
                          'salt_and_pepper': functools.partial(salt_and_pepper, p=n*0.1*random.random()),
                          'jpeg_compression': functools.partial(jpeg_compression, quality=random.randint(50,95-int(10*n))),
                          'greyscale': functools.partial(greyscale),
                          'bgr': functools.partial(bgr),
                          'hot': functools.partial(hot),
                          'rainbow': functools.partial(rainbow),
                          'saturate': functools.partial(saturate, low=random.randint(0, 64), high=random.randint(255-64, 255)),
                          'edge': functools.partial(edge),
                          'emboss': functools.partial(emboss),
                          'darken': functools.partial(darken, g=1-n*random.random()),
                          'negative': functools.partial(negative),
                          'scan_lines': functools.partial(scan_lines),
                          'additive_gaussian_noise': functools.partial(additive_gaussian_noise, sigma=np.random.uniform(n*0.01, n*0.5)),
                          'translate': functools.partial(translate, dx=np.random.uniform(-dx,dx), dy=np.random.uniform(-dy,dy)),                          
                          'horizontal_motion_blur': functools.partial(horizontal_motion_blur, kernel_size=random.uniform(0.05*n, 0.15*n)),
                          'vertical_motion_blur': functools.partial(vertical_motion_blur, kernel_size=random.uniform(0.05*n, 0.15*n)),                          
                          'barrel': functools.partial(barrel),
                          'left_swirl': functools.partial(left_swirl, center=(random.random(),random.random()), strength=n),
                          'right_swirl': functools.partial(right_swirl, center=(random.random(), random.random()), strength=n),
                          'horizontal_mirror': functools.partial(horizontal_mirror),
                          'vertical_mirror': functools.partial(vertical_mirror),
                          'left_shear': functools.partial(left_shear, border=border, s=np.random.uniform(0,shear)),
                          'right_shear': functools.partial(right_shear, border=border, s=np.random.uniform(0,shear)),
                          'rotate': functools.partial(rotate, deg=np.random.uniform(-deg,deg), border=border),
                          'isotropic_scale': functools.partial(isotropic_scale, s=np.random.uniform(1.0-z, 1.0+z), border=border),
                          'ghost': functools.partial(ghost,
                                                     txr=random.uniform(-0.01, 0.01),
                                                     tyr=random.uniform(-0.01, 0.01),
                                                     txg=random.uniform(-0.01, 0.01),
                                                     tyg=random.uniform(-0.01, 0.01),
                                                     txb=random.uniform(-0.01, 0.01),
                                                     tyb=random.uniform(-0.01, 0.01)),
                          'bit_depth': functools.partial(bit_depth, d=random.randint(1,7)),
                          'permute_color_channels': functools.partial(permute_color_channels, order=random.sample([[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]], 1)[0]),
                          'fliplr': functools.partial(fliplr),
                          'flipud': functools.partial(flipud),
                          'rot90cw': functools.partial(rot90cw),
                          'rot90ccw': functools.partial(rot90ccw),
                          'solarize': functools.partial(solarize, t=random.randint(128, 255)),
                          'colorjitter': functools.partial(colorjitter),
                          'autocontrast': functools.partial(autocontrast),
                          'sharpness': functools.partial(sharpness, f=10*random.random()),
                          'gamma': functools.partial(gamma, f=random.random()+0.5),
                          'crop': functools.partial(crop, s=1-0.2*random.random()),
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

