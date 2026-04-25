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


class Noise():
    def __init__(self, magnitude=0.25, provenance=False, register=None, deregister=('hot', 'rainbow', 'vertical_mirror', 'flipud', 'rot90cw', 'rot90ccw')):

        self._registry = (
            'translate', 'horizontal_motion_blur', 'vertical_motion_blur', 'barrel',
            'left_swirl', 'right_swirl', 'horizontal_mirror', 'vertical_mirror',
            'left_shear', 'right_shear', 'rotate', 'isotropic_scale',
            'fliplr', 'flipud', 'rot90cw', 'rot90ccw', 'crop',
            'blur', 'salt_and_pepper', 'jpeg_compression', 'greyscale', 'bgr',
            'hot', 'rainbow', 'saturate', 'edge', 'emboss', 'darken', 'negative',
            'scan_lines', 'additive_gaussian_noise', 'ghost', 'bit_depth',
            'permute_color_channels', 'solarize', 'colorjitter', 'autocontrast',
            'sharpness', 'gamma',
        )

        self._n = magnitude
        self._provenance = provenance
        self._registry = register if register is not None else self._registry
        self._registry = tuple(set(self._registry).difference(set(deregister)))

    # ============================================================
    # GEOMETRIC transforms
    # ============================================================

    @staticmethod
    def left_shear(im, s, border='zero'):
        assert isinstance(im, vipy.image.Image)
        assert s >= 0
        T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
        A = np.dot(vipy.geometry.affine_transform(ky=s), T)
        A = np.dot(np.linalg.inv(T), A)
        return im.load().affine_transform(A, border=border).mat2gray(0, 255)

    @staticmethod
    def right_shear(im, s, border='zero'):
        assert isinstance(im, vipy.image.Image)
        assert s >= 0
        T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
        A = np.dot(vipy.geometry.affine_transform(ky=-s), T)
        A = np.dot(np.linalg.inv(T), A)
        return im.load().affine_transform(A, border=border).mat2gray(0, 255)

    @staticmethod
    def rotate(im, rad=None, deg=None, border='zero'):
        assert isinstance(im, vipy.image.Image)
        assert rad is not None or deg is not None
        r = rad if rad is not None else (deg * np.pi / 180.0)
        T = vipy.geometry.affine_transform(txy=(-im.width()//2, -im.height()/2))
        A = np.dot(vipy.geometry.affine_transform(r=r), T)
        A = np.dot(np.linalg.inv(T), A)
        return im.load().affine_transform(A, border=border).mat2gray(0, 255)

    @staticmethod
    def barrel(im):
        assert isinstance(im, vipy.image.Image)
        (width, height) = (im.width(), im.height())
        distCoeff = np.zeros((4, 1), np.float64)
        k1 = 0.1 + 0.05 * np.random.randn()
        k2 = 0.1 + 0.05 * np.random.randn()
        p1 = 0.1 + 0.05 * np.random.randn()
        p2 = 0.1 + 0.05 * np.random.randn()
        distCoeff[0, 0] = k1
        distCoeff[1, 0] = k2
        distCoeff[2, 0] = p1
        distCoeff[3, 0] = p2
        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = width / 2.0   # define center x
        cam[1, 2] = height / 2.0  # define center y
        cam[0, 0] = 600.          # define focal length x
        cam[1, 1] = 600.          # define focal length y
        dst = cv2.undistort(im.load().array(), cam, distCoeff)
        return im.array(dst)

    @staticmethod
    def left_swirl(im, center=None, strength=2):
        assert isinstance(im, vipy.image.Image)
        center = center if center is not None else (im.width() * center[0], im.height() * center[1])
        return im.array(np.array(skimage.transform.swirl(im.load().array(), rotation=0, strength=strength, center=center, radius=im.maxdim())).astype(np.float32)).mat2gray(0, 1)

    @staticmethod
    def right_swirl(im, strength=2, center=None):
        assert isinstance(im, vipy.image.Image)
        center = center if center is not None else (im.mindim() * center[0], im.mindim() * center[1])
        return im.array(np.array(skimage.transform.swirl(im.load().fliplr().array(), rotation=0, center=center, strength=strength, radius=im.maxdim())).astype(np.float32)).mat2gray(0, 1).fliplr()

    @staticmethod
    def horizontal_mirror(im):
        assert isinstance(im, vipy.image.Image)
        img = im.load().array()
        (i, j) = (im.width()//2 if vipy.math.iseven(im.width()) else (im.width()//2) + 1, im.width()//2)
        img[:, i:] = np.fliplr(img[:, 0:j])
        return im.array(img)

    @staticmethod
    def vertical_mirror(im):
        assert isinstance(im, vipy.image.Image)
        img = im.load().array()
        (i, j) = (im.height()//2 if vipy.math.iseven(im.height()) else (im.height()//2) + 1, im.height()//2)
        img[i:, :] = np.flipud(img[0:j, :])
        return im.array(img)

    @staticmethod
    def vertical_motion_blur(im, kernel_size):
        assert isinstance(im, vipy.image.Image)
        cs = im.colorspace()
        kernel_size = max(3, int(im.mindim() * kernel_size))
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_v /= kernel_size
        return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_v)).mat2gray(0, 255).colorspace('float').to_colorspace(cs)

    @staticmethod
    def horizontal_motion_blur(im, kernel_size):
        assert isinstance(im, vipy.image.Image)
        cs = im.colorspace()
        kernel_size = max(3, int(im.mindim() * kernel_size))
        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size
        return im.load().map(lambda img: cv2.filter2D(img, -1, kernel_h)).mat2gray(0, 255).colorspace('float').to_colorspace(cs)

    @staticmethod
    def ghost(im, txr=0.01, tyr=0.01, txg=-0.01, tyg=-0.01, txb=0.01, tyb=-0.01):
        assert isinstance(im, vipy.image.Image)
        (imc, imz) = (im.clone().rgb().load(), im.clone().rgb().load().zeros())
        img = imz.numpy()
        img[:, :, 0] = Noise.translate(imc.red(), txr, tyr).load().array()
        img[:, :, 1] = Noise.translate(imc.green(), txg, tyg).load().array()
        img[:, :, 2] = Noise.translate(imc.blue(), txb, tyb).load().array()
        return imz

    @staticmethod
    def crop(im, s=0.8):
        (H, W) = (im.height(), im.width())
        return im.crop(vipy.geometry.BoundingBox(xmin=random.randint(0, int((1-s) * im.width())),
                                                 ymin=random.randint(0, int((1-s) * im.height())),
                                                 width=math.ceil(s * im.width()),
                                                 height=math.ceil(s * im.height()))).resize(height=H, width=W)

    @staticmethod
    def fliplr(im):
        return im.load().fliplr()

    @staticmethod
    def flipud(im):
        return im.load().flipud()

    @staticmethod
    def rot90cw(im):
        return im.load().rot90cw()

    @staticmethod
    def rot90ccw(im):
        return im.load().rot90ccw()

    @staticmethod
    def translate(im, dx, dy):
        """Translate by (dx,dy) normalized pixels, with zero border handling"""
        assert isinstance(im, vipy.image.Image)
        return im.load().padcrop(im.imagebox().translate(int(dx * im.mindim()), int(dy * im.mindim())))

    @staticmethod
    def isotropic_scale(im, s, border='zero'):
        assert isinstance(im, vipy.image.Image)
        assert s > 0
        (W, H) = (im.width(), im.height())
        return im.load().rescale(s).padcrop(vipy.geometry.BoundingBox(centroid=(im.width()//2, im.height()//2), width=W, height=H))

    @staticmethod
    def zoom(im, s, zx=None, zy=None, border='zero'):
        assert isinstance(im, vipy.image.Image)
        assert s > 0
        (zx, zy) = ((-im.width()/2) if zx is None else -zx, (-im.height()/2) if zy is None else -zy)
        return Noise.isotropic_scale(Noise.translate(im, zx, zy), s)

    # ============================================================
    # PHOTOMETRIC transforms
    # ============================================================

    @staticmethod
    def mask(im, num_masks=1, xywh_range=((0.1, 0.9), (0, 1, 0.9), (0.3, 0.5), (0.3, 0.5)), fill='zeros', radius=7):
        """Introduce one or more rectangular masks filled by mask_type.

        The position and size of masks are uniformly sampled from the provided range of xywh = (xmin, ymin, width, height)

        xywh_range = ((xmin_lowerbound, xmin_upperbound), (ymin_lb,ymin_ub), (width_lb,width_ub), (height_lb,height_ub))) relative to the normalized height and width

        Allowable fills = ['zeros', 'inverse_zeros', 'mean', 'blur', 'pixelize', 'inverse_mean', 'inverse_blur']

        - zeros: all masks are replaced with zeros
        - inverse_zeros: all pixels outside masks are replaced with zeros
        - mean: all pixels inside masks are replaced with the mean pixel
        - inverse_mean: all pixels outside masks are replaced with the mean pixel
        - blur: all pixels inside masks are replaced with blurred pixels with a given gaussian radius
        - inverse_blur: all pixels outside masks are replaced with blurred pixels with a given gaussian radius
        - pixelize: all pixels inside masks are replaced with low resolution pixels with a given downscale pixel radius

        Radius is specific to blur and pixelize masks only and ignored for others
        """
        (H, W) = (im.height(), im.width())
        masks = [vipy.object.Detection(category='mask%d' % k,
                                       xmin=np.random.randint(xywh_range[0][0]*W, xywh_range[0][1]*W),
                                       ymin=np.random.randint(xywh_range[1][0]*H, xywh_range[1][1]*H),
                                       width=np.random.randint(xywh_range[2][0]*W, xywh_range[2][1]*W),
                                       height=np.random.randint(xywh_range[3][0]*H, xywh_range[3][1]*H))
                 for k in range(num_masks)]
        im = vipy.image.Scene.cast(im.clone()).objects(masks)

        if fill == 'zeros':
            im = im.fgmask()
        elif fill == 'inverse_zeros':
            im = im.bgmask()
        elif fill == 'mean':
            im = im.mean_mask()
        elif fill == 'rand':
            im = im.random_mask()
        elif fill == 'inverse_mean':
            im = im.inverse_mean_mask()
        elif fill == 'blur':
            im = im.blur_mask(radius=radius)
        elif fill == 'inverse_blur':
            im = im.inverse_blur_mask(radius=radius)
        elif fill in ['pixel', 'pixelize', 'pixelate']:
            im = im.pixel_mask(radius=radius)
        elif fill == 'alpha':
            return im.alpha_mask()
        else:
            raise ValueError("unknown mask type '%s'" % fill)
        return im

    @staticmethod
    def blur(im, sigma):
        assert isinstance(im, vipy.image.Image)
        imblur = im.rgb().blur(sigma=sigma * im.mindim())
        return imblur.greyscale().colorspace_like(im) if im.channels() == 1 else imblur.colorspace_like(im)

    @staticmethod
    def salt_and_pepper(im, p):
        assert isinstance(im, vipy.image.Image)
        assert p >= 0 and p <= 1
        W = np.array(np.random.rand(im.height(), im.width(), 1) > p).astype(np.float32)
        B = np.array(np.random.rand(im.height(), im.width(), 1) > p).astype(np.float32)
        return im.rgb().load().float().mat2gray().map(lambda x: np.multiply(x, W) + 1 - W).map(lambda x: np.multiply(x, B)).colorspace_like(im)

    @staticmethod
    def jpeg_compression(im, quality):
        assert isinstance(im, vipy.image.Image)
        assert quality >= 0 and quality <= 95
        out = BytesIO()
        im.load().rgb().pil().save(out, format='jpeg', quality=quality)
        out.seek(0)
        return im.load().array(np.array(PIL.Image.open(out))).mat2gray(0, 255).colorspace_like(im)

    @staticmethod
    def bit_depth(im, d):
        assert d >= 1 and d <= 8
        cs = im.colorspace()
        return im.array(np.array(PIL.ImageOps.posterize(im.load().clone().rgb().pil(), d))).colorspace('rgb').to_colorspace(cs)

    @staticmethod
    def solarize(im, t):
        """All pixels above threshold are inverted"""
        assert t >= 0 and t <= 255
        cs = im.colorspace()
        return im.array(np.array(PIL.ImageOps.solarize(im.load().clone().rgb().pil(), t))).colorspace('rgb').to_colorspace(cs)

    @staticmethod
    def permute_color_channels(im, order):
        assert sorted(order) == [0, 1, 2]
        return im.array(np.array(im.load().rgb())[:, :, order].copy())

    @staticmethod
    def greyscale(im):
        assert isinstance(im, vipy.image.Image)
        return im.clone().load().greyscale().colorspace_like(im)

    @staticmethod
    def bgr(im):
        assert isinstance(im, vipy.image.Image)
        return im.array(im.bgr().load().array().copy()).colorspace('rgb')

    @staticmethod
    def hot(im):
        assert isinstance(im, vipy.image.Image)
        return im.hot().load().rgb()

    @staticmethod
    def rainbow(im):
        assert isinstance(im, vipy.image.Image)
        return im.rainbow().load().rgb()

    @staticmethod
    def saturate(im, low, high):
        assert isinstance(im, vipy.image.Image)
        return im.saturate(low, high)

    @staticmethod
    def colorjitter(im):
        transform = torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))
        cs = im.colorspace()
        return im.array(np.array(transform(im.load().clone().rgb().pil()))).colorspace('rgb').to_colorspace(cs)

    @staticmethod
    def sharpness(im, f):
        cs = im.colorspace()
        return im.array(np.array(torchvision.transforms.v2.functional.adjust_sharpness(im.load().clone().rgb().torch('CHW'), f).permute(1, 2, 0))).colorspace('rgb').to_colorspace(cs)

    @staticmethod
    def gamma(im, f):
        cs = im.colorspace()
        return im.array(np.array(torchvision.transforms.v2.functional.adjust_gamma(im.load().clone().rgb().torch('CHW'), f).permute(1, 2, 0))).colorspace('rgb').to_colorspace(cs)

    @staticmethod
    def autocontrast(im):
        return im.mat2gray().rgb()

    @staticmethod
    def edge(im):
        assert isinstance(im, vipy.image.Image)
        return im.array(np.array(im.luminance().pil().filter(PIL.ImageFilter.FIND_EDGES))).colorspace('lum').colorspace_like(im)

    @staticmethod
    def emboss(im):
        assert isinstance(im, vipy.image.Image)
        return im.array(np.array(im.luminance().pil().filter(PIL.ImageFilter.EMBOSS))).colorspace('lum').colorspace_like(im)

    @staticmethod
    def darken(im, g):
        assert isinstance(im, vipy.image.Image)
        cs = im.colorspace()
        return im.gain(g).mat2gray(0, 255).colorspace('float').to_colorspace(cs)

    @staticmethod
    def negative(im):
        assert isinstance(im, vipy.image.Image)
        cs = im.colorspace()
        return im.load().array(1 - im.load().mat2gray(0, 255).array()).colorspace('float').to_colorspace(cs)

    @staticmethod
    def scan_lines(im):
        assert isinstance(im, vipy.image.Image)
        imc = im.load()
        imc._array[::2, :] = 0
        return imc

    @staticmethod
    def additive_gaussian_noise(im, sigma=0.1):
        assert isinstance(im, vipy.image.Image)
        cs = im.colorspace()
        return im.rgb().float().map(lambda img: img + max(sigma * im.maxpixel(), sigma) * np.random.randn(img.shape[0], img.shape[1], im.channels()).astype(np.float32)).saturate(0, 255).mat2gray(0, 255).colorspace('float').to_colorspace(cs)

    def transformations(self):
        return list(self._registry)

    def _dispatch(self, im, name):
        """Dispatch a registered transform with freshly-sampled parameters."""
        n = self._n

        # no-arg transforms
        if name == 'greyscale':         return self.greyscale(im)
        if name == 'bgr':               return self.bgr(im)
        if name == 'hot':               return self.hot(im)
        if name == 'rainbow':           return self.rainbow(im)
        if name == 'edge':              return self.edge(im)
        if name == 'emboss':            return self.emboss(im)
        if name == 'negative':          return self.negative(im)
        if name == 'scan_lines':        return self.scan_lines(im)
        if name == 'barrel':            return self.barrel(im)
        if name == 'horizontal_mirror': return self.horizontal_mirror(im)
        if name == 'vertical_mirror':   return self.vertical_mirror(im)
        if name == 'fliplr':            return self.fliplr(im)
        if name == 'flipud':            return self.flipud(im)
        if name == 'rot90cw':           return self.rot90cw(im)
        if name == 'rot90ccw':          return self.rot90ccw(im)
        if name == 'colorjitter':       return self.colorjitter(im)
        if name == 'autocontrast':      return self.autocontrast(im)

        # randomized per call
        if name == 'blur':              return self.blur(im, sigma=n*random.uniform(0.01, 0.05))
        if name == 'salt_and_pepper':   return self.salt_and_pepper(im, p=random.uniform(0, n*0.1))
        if name == 'jpeg_compression':  return self.jpeg_compression(im, quality=random.randint(3, max(4, 100 - int(100*n))))
        if name == 'saturate':          return self.saturate(im, low=random.randint(0, 64), high=random.randint(255-64, 255))
        if name == 'darken':            return self.darken(im, g=random.uniform(1-n, 1))
        if name == 'additive_gaussian_noise':
            return self.additive_gaussian_noise(im, sigma=float(np.random.uniform(n*0.01, n*0.5)))
        if name == 'translate':
            d = n * 0.15
            return self.translate(im, dx=float(np.random.uniform(-d, d)), dy=float(np.random.uniform(-d, d)))
        if name == 'horizontal_motion_blur':
            return self.horizontal_motion_blur(im, kernel_size=random.uniform(0.05*n, 0.15*n))
        if name == 'vertical_motion_blur':
            return self.vertical_motion_blur(im, kernel_size=random.uniform(0.05*n, 0.15*n))
        if name == 'left_swirl':
            return self.left_swirl(im, center=(random.random(), random.random()), strength=n)
        if name == 'right_swirl':
            return self.right_swirl(im, center=(random.random(), random.random()), strength=n)
        if name == 'left_shear':
            return self.left_shear(im, border='replicate', s=float(np.random.uniform(0, n*0.25)))
        if name == 'right_shear':
            return self.right_shear(im, border='replicate', s=float(np.random.uniform(0, n*0.25)))
        if name == 'rotate':
            return self.rotate(im, deg=float(np.random.uniform(-n*25, n*25)), border='replicate')
        if name == 'isotropic_scale':
            z = n * 0.2
            return self.isotropic_scale(im, s=float(np.random.uniform(1.0-z, 1.0+z)), border='replicate')
        if name == 'ghost':
            r = lambda: random.uniform(-0.01, 0.01)  # local helper, not stored, not pickled
            return self.ghost(im, txr=r(), tyr=r(), txg=r(), tyg=r(), txb=r(), tyb=r())
        if name == 'bit_depth':
            return self.bit_depth(im, d=random.randint(1, 7))
        if name == 'permute_color_channels':
            return self.permute_color_channels(im, order=random.choice([[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]))
        if name == 'solarize':          return self.solarize(im, t=random.randint(128, 255))
        if name == 'sharpness':         return self.sharpness(im, f=random.uniform(0, 10))
        if name == 'gamma':             return self.gamma(im, f=random.uniform(0.5, 1.5))
        if name == 'crop':              return self.crop(im, s=random.uniform(0.8, 1.0))

        raise ValueError("unknown transform '%s'" % name)

    def transform(self, im, transform):
        assert isinstance(im, vipy.image.Image)
        assert transform in self._registry, "unknown transform '%s'" % transform
        assert im.array().dtype == np.uint8 or (im.colorspace() == 'float' and im.channels() in (1, 3)), \
            "noise transforms require uint8 input or float input with 1 or 3 channels, got dtype=%s colorspace=%s channels=%d" % (im.array().dtype, im.colorspace(), im.channels())

        print(transform)
        imt = im.clone()
        if self._provenance:
            imt = vipy.image.Scene.cast(imt).append_object(vipy.object.Detection.cast(imt.imagebox()).new_category('provenance'))
        imt = self._dispatch(imt.load(), transform)
        if self._provenance:
            imt.setattribute('vipy.noise', {'transform': transform, 'bbox': imt.last_object()})  # bounding box of original image in this geometrically perturbed image
        return imt

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
                                   'left_swirl', 'right_swirl', 'horizontal_mirror', 'left_shear', 'right_shear',
                                   'rotate', 'isotropic_scale', 'fliplr', 'crop'])

class Photometric(Noise):
    def __init__(self, magnitude=0.25, provenance=False):
        super().__init__(magnitude=magnitude, provenance=provenance,
                         register=['blur', 'salt_and_pepper', 'jpeg_compression', 'greyscale', 'bgr', 'hot', 'saturate', 'edge', 'emboss',
                                   'darken', 'negative', 'scan_lines', 'additive_gaussian_noise', 'bit_depth', 'permute_color_channels', 'solarize',
                                   'colorjitter', 'sharpness', 'gamma', 'ghost'])


geometric = Geometric(provenance=True)
photometric = Photometric(provenance=True)
randomcrop = RandomCrop(provenance=True)
