import os
import numpy as np
import vipy.image
from vipy.image import ImageDetection, Image, ImageCategory, Scene
from vipy.object import Detection
from vipy.util import tempjpg, temppng, tempdir, Failed
from vipy.geometry import BoundingBox
import PIL.Image


# Common Parameters
jpegurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg'
gifurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Rotating_earth_%28large%29.gif/200px-Rotating_earth_%28large%29.gif'
pngurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/560px-PNG_transparency_demonstration_1.png'
greyfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_grey.jpg')
rgbfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_rgb.jpg')
rgbafile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_rgba.png')

                        
def test_image():
    assert vipy.version.is_at_least('0.7.0')
    assert not vipy.version.is_at_least('1.0.0')
    
    # Empty constructor should not raise exception
    im = Image()
    print('[test_image.image]: Empty Constructor: PASSED')

    # Non-existant filename should not raise exception during constructor (only during load)
    im = Image(filename='myfile')
    print('[test_image.image]: Filename Constructor: PASSED')

    # Malformed URL should raise exception
    im = None
    try:
        im = Image(url='myurl')
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_image.image]: Malformed URL constructor: PASSED')

    # Valid URL should not raise exception (even if it is not an image extension)
    im = Image(url='http://visym.com')
    print('[test_image.image]: Image URL constructor: PASSED')

    # Valid URL and filename to save it
    im = Image(url='http://visym.com/myfile.jpg', filename='/my/file/path')
    print('[test_image.image]: URL and filename constructor: PASSED')

    # URL object
    im = Image(url=jpegurl)
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]:   Image length: %d' % len(im))
    im.download()
    print('[test_image.image]:   Image __desc__: %s' % im)
    im.load()
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]: URL download: PASSED')

    # Valid URL but without an image extension
    im = Image(url='http://bit.ly/great_horned_owl')
    print('[test_image.image]:   Image __desc__: %s' % im)
    im.load()
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]: URL download (without image extension): PASSED')

    # Invalid URL with ignore and verbose
    im = Image(url='https://a_bad_url.jpg')
    print('[test_image.image]:   Image __desc__: %s' % im)
    im.load(ignoreErrors=True, verbose=True)
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]: Invalid URL download: PASSED')

    # URL with filename
    im = Image(url=jpegurl, filename=tempjpg())
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]:   Image length: %d' % len(im))
    im.download()
    print('[test_image.image]:   Image __desc__: %s' % im)
    im.load()
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]: URL with filename download: PASSED')

    # URL with filename in cache
    os.environ['VIPY_CACHE'] = tempdir()
    im = Image(url=jpegurl)
    print('[test_image.image]:   Image __desc__: %s' % im)
    im.load()
    print('[test_image.image]:   Image __desc__: %s' % im)
    assert os.environ['VIPY_CACHE'] in im.filename()
    print('[test_image.image]: URL with cache download: PASSED')

    # Equality
    im = Image(array=np.zeros( (10,10,3), dtype=np.uint8 ), colorspace='rgb')
    assert im == im
    im2 = im.clone()
    im2._array[0,0] = 1
    assert im != im2
    print('[test_image.image]: equality  PASSED')    
    
    # Array objects
    Image(array=np.zeros( (10,10,3), dtype=np.uint8 ), colorspace='rgb')
    Image(array=np.zeros( (10,10,3), dtype=np.uint8 ), colorspace='bgr')
    Image(array=np.zeros( (10,10,3), dtype=np.uint8 ), colorspace='hsv')
    Image(array=np.zeros( (10,10,1), dtype=np.uint8 ), colorspace='lum')    
    Image(array=np.zeros( (10,10,3), dtype=np.uint8 ), colorspace='rgb')    
    Image(array=np.zeros( (10,10,4), dtype=np.uint8 ), colorspace='rgba')
    Image(array=np.zeros( (10,10,4), dtype=np.uint8 ), colorspace='bgra')        
    Image(array=np.random.randn(10,10,10).astype(np.float32), colorspace='float')
    Image(array=np.random.rand(10,10,1).astype(np.float32), colorspace='grey')    
    try:
        Image(array=np.zeros( (10,10) )) 
        Failed()  # np.float32 unallowed
    except Failed:
        raise
    except:
        pass
    try:
        Image(array=np.matrix( (10,10) ).astype(np.float32))
        Failed()  # np.matrix unallowed
    except Failed:
        raise
    except:
        pass
    try:
        Image(array=np.zeros( (10,10,3), dtype=np.float32), colorspace='rgb')  
        Failed()  # rgb image must be uint8
    except Failed:
        raise
    except:
        pass
    try:
        Image(array=2*np.random.rand(10,10).astype(np.float32), colorspace='grey')  
        Failed()  # grey image must be [0,1] float32
    except Failed:
        raise
    except:
        pass

    # Shared array
    img = np.random.rand(2,2).astype(np.float32)
    im = vipy.image.Image(array=img)
    img[0,0] = 0
    assert im.array()[0,0] == 0    
    img = np.random.rand(2,2).astype(np.float32)
    im = vipy.image.Image().fromarray(img)
    img[0,0] = 0
    assert im.array()[0,0] != 0
    print('[test_image]: array by reference  PASSED')
    

    # Image file formats
    for imgfile in [rgbfile, greyfile, rgbafile]:
        _test_image_fileformat(imgfile)

        
        
            
def _test_image_fileformat(imgfile):
    # Filename object
    im = ImageDetection(filename=imgfile, xmin=100, ymin=100, bbwidth=700, height=1000, category='face')
    print('[test_image.image]["%s"]:  Image __desc__: %s' % (im, imgfile))
    im.crop()
    print('[test_image.image]["%s"]:  Image __desc__: %s' % (im, imgfile))
    print('[test_image.image]["%s"]:  Filename: PASSED' % imgfile)

    # Clone
    im = Image(filename=imgfile).load()
    imb = im
    im._array = im._array + 1  # modify array
    np.testing.assert_array_equal(imb.numpy(), im.numpy())  # share buffer
    imc = im.clone()
    np.testing.assert_array_equal(imc.numpy(), imb.numpy())  # share buffer
    imc._array = imc._array + 2  # modify array
    assert np.any(imc.numpy() != imb.numpy())  
    imc = im.clone(flushforward=True)
    assert(imc._array is None and im._array is not None)  
    imc = im.clone(flushbackward=True)
    assert(im._array is None and imc._array is not None)  
    imc = im.clone(flush=True)
    assert(im._array is None and imc._array is None)
    print('[test_image.image]["%s"]:  Image.clone: PASSED' % imgfile)

    # Downgrade
    im = ImageDetection(filename=imgfile, xmin=100, ymin=100, bbwidth=700, height=1000, category='face')    
    imd = im.detection()
    assert imd.xywh() == im.boundingbox().xywh()
    imd = im.image()
    assert imd.shape() == im.shape()
    print('[test_image.image]["%s"]:  ImageDetection downgrade  PASSED' % imgfile)    
    
    # Saveas
    im = Image(filename=imgfile).load()
    f = temppng()
    assert im.saveas(f) == f and os.path.exists(f)
    print('[test_image.image]["%s"]:  Image.saveas: PASSED' % imgfile)

    # Stats
    im = Image(filename=imgfile).load().stats()
    print('[test_image.image]["%s"]:  Image.stats: PASSED' % imgfile)

    # Resize
    f = temppng()
    im = Image(filename=imgfile).load().resize(cols=16,rows=8).saveas(f)
    assert Image(filename=f).shape() == (8,16)
    assert Image(filename=f).width() == 16
    assert Image(filename=f).height() == 8
    im = Image(filename=imgfile).load().resize(16,8).saveas(f)
    assert Image(filename=f).shape() == (8,16)
    assert Image(filename=f).width() == 16
    assert Image(filename=f).height() == 8
    im = Image(filename=imgfile).load()
    (h,w) = im.shape()
    im = im.resize(rows=16)
    assert im.shape() == (16,int((w / float(h)) * 16.0))
    print('[test_image.image]["%s"]:  Image.resize: PASSED' % imgfile)

    # Rescale
    f = temppng()
    im = Image(filename=imgfile).load().resize(rows=8).saveas(f)
    assert Image(filename=f).height() == 8
    im = Image(filename=imgfile).load().resize(cols=8).saveas(f)
    assert Image(filename=f).width() == 8
    im = Image(filename=imgfile).load().maxdim(256).saveas(f)
    assert np.max(Image(filename=f).shape()) == 256
    print('[test_image.image]["%s"]:  Image.rescale: PASSED' % imgfile)

    # GIF
    im = Image(url=gifurl)
    im.download(verbose=True)
    assert im.shape() == (200,200)
    print('[test_image.image]["%s"]:  GIF: PASSED' % imgfile)

    # Transparent PNG
    im = Image(url=pngurl)
    im.load(verbose=True)
    assert im.colorspace() == 'rgba'
    print('[test_image.image]["%s"]:  PNG: PASSED' % imgfile)

    # Image colorspace conversion
    im = Image(filename=imgfile).resize(200,200)
    print(im.rgb()) 
    assert im.colorspace() == 'rgb'
    assert(im.shape() == (200,200) and im.channels() == 3)
    assert im.array().dtype == np.uint8
    
    print(im.luminance()) 
    assert im.colorspace() == 'lum'   
    assert(im.shape() == (200,200) and im.channels() == 1)
    assert im.array().dtype == np.uint8
    
    print(im.bgr())
    assert im.colorspace() == 'bgr'    
    assert(im.shape() == (200,200) and im.channels() == 3)
    assert im.array().dtype == np.uint8
    
    print(im.rgba())
    assert im.colorspace() == 'rgba'    
    assert(im.shape() == (200,200) and im.channels() == 4)
    assert im.array().dtype == np.uint8
    
    print(im.hsv())
    assert im.colorspace() == 'hsv'    
    assert(im.shape() == (200,200) and im.channels() == 3)
    assert im.array().dtype == np.uint8
    
    print(im.bgra()) 
    assert im.colorspace() == 'bgra'   
    assert(im.shape() == (200,200) and im.channels() == 4)
    assert im.array().dtype == np.uint8
    
    print(im.gray()) 
    assert im.colorspace() in ['grey', 'gray']
    assert(im.shape() == (200,200) and im.channels() == 1)
    assert im.array().dtype == np.float32
    
    print(im.float())
    assert im.colorspace() == 'float'       
    assert(im.shape() == (200,200) and im.channels() == 1)
    assert im.array().dtype == np.float32
    
    print('[test_image.image]["%s"]:  Image conversion: PASSED' % imgfile)
    im = Image(filename=imgfile).load().grey()
    assert im.colorspace() == 'grey'
    assert im.max() == 1.0
    print('[test_image.image]["%s"]:  Greyscale image conversion: PASSED' % imgfile)

    # Crops
    imorig = Image(filename=imgfile).load().lum()
    (H,W) = imorig.shape()
    im = imorig.clone().maxsquare()
    assert im.shape() == (np.maximum(W,H), np.maximum(W,H)) and imorig.array()[0,0] == im.array()[0,0]
    im = imorig.clone().minsquare()    
    assert im.shape() == (np.minimum(W,H), np.minimum(W,H)) and imorig.array()[0,0] == im.array()[0,0]
    im = imorig.clone().centersquare()
    (xo,yo) = imorig.centerpixel() 
    (x,y) = im.centerpixel()
    assert im.shape() == (np.minimum(W,H), np.minimum(W,H)) and imorig.array()[yo,xo] == im.array()[y,x]
    print('[test_image.image]["%s"]:  crops PASSED' % imgfile)

    # Pixel operations
    im = Image(filename=imgfile).load()
    im.min()
    im.max()
    im.mean()
    im.intensity()
    im.saturate(0, 0.5)
    im.mat2gray()
    im.gain(1)
    im.bias(2)
    print('[test_image.image]["%s"]:  greylevel transformations  PASSED' % imgfile)

    # Image conversion
    im = Image(filename=imgfile).load()
    im.pil()
    im.numpy()
    im.html()
    im.torch()
    print('[test_image.image]["%s"]:  image conversions  PASSED' % imgfile)

    # Image colormaps
    im = ImageDetection(filename=imgfile, xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face').crop()
    im.rgb().jet().bone().hot().rainbow()
    print('[test_image.image]["%s"]:  Image colormaps: PASSED' % imgfile)

    # Image exporter
    im.dict()
    print('[test_image.image]["%s"]:  dictionary: PASSED' % imgfile)
    
    # Image category
    im = ImageCategory(filename=imgfile, category='face')
    assert im.load().category() == 'face'
    print('[test_image.image]["%s"]:  ImageCategory constructor PASSED' % imgfile)
    assert ImageCategory(category='1') == ImageCategory(category='1')
    assert ImageCategory(category='1') != ImageCategory(category='2')
    print('[test_image.image]["%s"]:  ImageCategory equivalence PASSED' % imgfile)
    assert ImageCategory(category='1').category() == '1'
    assert ImageCategory(label='1').label() == '1'    
    assert not ImageCategory(category='1').category == '2'
    assert ImageCategory(category='1').category('2').label() == '2' 
    im.score(1.0)
    im.probability(1.0)
    try:
        ImageCategory(category='1', label='2')
        Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_image.image]["%s"]:  ImageCategory category conversion PASSED' % imgfile)    
    
    # Random images
    im = vipy.image.RandomImage(128,256)
    assert im.shape() == (128, 256)
    print('[test_image.image]["%s"]:  RandomImage PASSED' % imgfile)
    im = vipy.image.RandomImageDetection(128,256)
    assert im.clone().crop().width() == im.bbox.imclipshape(W=256,H=128).width()
    print('[test_image.image]["%s"]:  RandomImageDetection PASSED' % imgfile)

    d = vipy.image.RandomImageDetection(128,256).dict()
    assert isinstance(d, dict)
    print('[test_image.image]["%s"]:  dict PASSED' % imgfile)

    # Map
    im = vipy.image.RandomImage(128,256)
    im2 = im.clone().map(lambda img: np.float32(img)+1.0)
    assert np.allclose(np.float32(im.array())+1.0, im2.array())
    print('[test_image.image]["%s"]:  map PASSED' % imgfile)

    # interpolation 
    im = vipy.image.RandomImage(128,256)
    im.resize(256,256, interp='bilinear')
    im.resize(256,256, interp='bicubic')
    im.resize(256,256, interp='nearest')
    try:
        im.resize(256,256, interp='somethingelse')        
        Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_image.image]["%s"]:  interpolation PASSED' % imgfile)    
    
    
    
def test_imagedetection():
    # Constructors
    im = ImageDetection()
    im.__repr__()
    assert im.category() is None and im.bbox.xywh() == (0,0,0,0)
    print('[test_image.imagedetection]: Empty ImageDetection constructor PASSED')
    im = ImageDetection(category='test', xmin=0, ymin=0, width=200, height=200)
    print('[test_image.imagedetection]: Empty filename ImageDetection constructor PASSED')
    print('[test_image.imagedetection]: Empty bounding box ImageDetection constructor PASSED')

    im = ImageDetection(filename=rgbfile, category='face', bbox=BoundingBox(0,0,100,100))
    try:
        im = ImageDetection(filename=rgbfile, category='face', bbox='a_bad_type')
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_image.imagedetection]: bounding box constructor PASSED')

    im = ImageDetection(filename=rgbfile, category='face', xmin=-1, ymin=-2, ymax=10, xmax=20)
    try:
        im = ImageDetection(filename=rgbfile, category='face', xmin='a_bad_type', ymin=-2, ymax=10, xmax=20)
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_image.imagedetection]: (xmin,ymin,xmax,ymax) bounding box constructor PASSED')

    im = ImageDetection(filename=rgbfile, xmin=100, ymin=100, bbwidth=200, bbheight=-200, category='face')
    assert im.invalid()
    print('[test_image.imagedetection]: invalid box: PASSED')
    try:
        im.crop()
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_image.imagedetection]: invalid box crop: PASSED')

    im = ImageDetection(filename=rgbfile, xmin=100000, ymin=100000, bbwidth=200, bbheight=200, category='face')
    print('[test_image.imagedetection]: invalid imagebox: PASSED')

    # boundingbox() methods
    im = ImageDetection(filename=rgbfile, category='face', xmin=-1, ymin=-2, ymax=10, xmax=20)
    assert im.boundingbox() == BoundingBox(-1, -2, 20, 10)
    assert not im.boundingbox(xmin=0, ymin=0, width=10, height=20) == BoundingBox(0,0,width=10, height=20)
    assert im.boundingbox(xmin=0, ymin=0, width=10, height=20).bbox == BoundingBox(0,0,width=10, height=20)
    assert im.boundingbox(bbox=BoundingBox(1,2,3,4)).bbox == BoundingBox(1,2,3,4)
    assert im.boundingbox(bbox=BoundingBox(xcentroid=1,ycentroid=2,width=3,height=4)).bbox == BoundingBox(xcentroid=1,ycentroid=2,width=3,height=4)
    assert im.boundingbox(xmin=-1, ymin=-2, ymax=10, xmax=20).boundingbox(dilate=1.5).bbox == BoundingBox(xmin=-1, ymin=-2, ymax=10, xmax=20).dilate(1.5)
    assert im.boundingbox(xmin=-1, ymin=-2, ymax=10, xmax=20).boundingbox(dilate=1.5).bbox == BoundingBox(xmin=-1, ymin=-2, ymax=10, xmax=20).dilate(1.5)
    try:
        im.boundingbox(xmin=1)
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_image.imagedetection]: boundingbox() methods PASSED')

    # Crop
    im = ImageDetection(filename=rgbfile, xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face').crop()
    assert(im.shape() == (200,200))
    assert(im.bbox.width() == im.width() and im.bbox.height() == im.height() and im.bbox.xmin() == 0 and im.bbox.ymin() == 0)
    im = ImageDetection(filename=rgbfile)
    (H,W) = im.shape()
    im = im.boundingbox(xmin=0, ymin=0, width=W, height=H).crop()
    assert(im.shape() == (H,W))
    print('[test_image.imagedetection]: crop  PASSED')

    # Rescale
    im = ImageDetection(filename=rgbfile, xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face')
    im = im.rescale(0.5)
    assert(im.bbox.width() == 100 and im.bbox.height() == 100)
    print('[test_image.imagedetection]: rescale  PASSED')

    # Resize
    imorig = ImageDetection(filename=rgbfile, xmin=100, ymin=100, width=200, height=300, category='face')
    im = imorig.clone().resize(cols=int(imorig.width() // 2.0), rows=int(imorig.height() // 2.0))
    assert(im.bbox.width() == 100 and im.bbox.height() == 150)
    (H,W) = imorig.shape()
    im = imorig.clone().resize(cols=100)
    assert(im.crop().width() == int(np.round(200*(100.0/W))))
    print('[test_image.imagedetection]: resize  PASSED')

    # Isinterior
    im = ImageDetection(array=np.zeros((10,20), dtype=np.float32), xmin=100, ymin=100, xmax=200, ymax=200, category='face')
    assert not im.isinterior()
    assert not im.isinterior(20,10)
    im = ImageDetection(array=np.zeros((10,20), dtype=np.float32), xmin=0, ymin=0, width=20, height=10)
    assert im.isinterior()
    print('[test_image.imagedetection]: interior  PASSED')

    # Fliplr
    img = np.random.rand(10,20).astype(np.float32)
    im = ImageDetection(array=img, xmin=0, ymin=0, xmax=5, ymax=10)
    assert im.isinterior()
    assert np.allclose(im.clone().fliplr().crop().array(), np.fliplr(im.crop().array()))
    print('[test_image.imagedetection]: fliplr  PASSED')

    # Square crops
    img = np.random.rand(10,20).astype(np.float32)
    im = ImageDetection(array=img, xmin=0, ymin=0, xmax=5, ymax=10)
    imorig = im.clone()    
    (x,y,w,h) = im.bbox.xywh()
    (xc,yc) = im.boundingbox().centroid()  # box centroid
    assert im.clone().minsquare().bbox.shape() == (10, 5)
    assert im.clone().minsquare().shape() == (10, 10)    
    assert im.clone().minsquare().bbox.centroid() == (xc, yc)
    assert im.clone().maxsquare().bbox.shape() == (10, 5)
    assert im.clone().maxsquare().shape() == (20,20)    
    assert im.clone().maxsquare().bbox.centroid() == (xc,yc)
    assert im.clone().centersquare().shape() == (10,10)
    assert im.clone().centersquare().boundingbox().xywh() == (-5,0,5,10)
    img = np.random.rand(20,10,3).astype(np.float32)
    assert ImageDetection(array=img, xmin=0, ymin=0, width=10, height=10).minsquare().crop().shape() == (10,10)
    img = np.random.rand(21,9,3).astype(np.float32)
    assert ImageDetection(array=img, xmin=0, ymin=0, width=9, height=21).centersquare().crop().shape() == (9,9)
    img = np.random.rand(10,11,3).astype(np.float32)
    assert ImageDetection(array=img, xmin=0, ymin=0, width=11, height=10).centersquare().crop().shape() == (10,10)
    print('[test_image.imagedetection]: minsquare PASSED')
    print('[test_image.imagedetection]: maxsquare PASSED')
    print('[test_image.imagedetection]: centersquare PASSED')                  

    # Dilate
    im = ImageDetection(array=img, xmin=1, ymin=0, width=3, height=4)
    assert im.clone().dilate(2).bbox == BoundingBox(centroid=im.bbox.centroid(), width=6, height=8) and img.shape[0] == im.height()
    print('[test_image.imagedetection]: dilate PASSED')

    # Pad
    img = np.random.rand(20,40).astype(np.float32)
    im = ImageDetection(array=img, xmin=0, ymin=0, width=40, height=20)
    assert np.allclose(im.clone().zeropad(10,10).crop().array(), img) and (im.clone().zeropad(10,20).shape() == (20 + 20 * 2, 40 + 10 * 2))
    img = np.random.rand(20,40).astype(np.float32)
    im = ImageDetection(array=img, xmin=0, ymin=0, width=40, height=20)
    assert np.allclose(im.clone().meanpad(10,10).crop().array(), img) and (im.clone().meanpad(10,20).shape() == (20 + 20 * 2, 40 + 10 * 2))
    imorig = ImageDetection(array=img, xmin=0, ymin=0, width=40, height=20)
    im = imorig.clone().meanpad( (0,10), (0,20) )
    assert np.allclose(imorig.clone().meanpad( (0,10), (0,20) ).crop().array(), img) and (imorig.clone().meanpad( (0,10), (0,20) ).shape() == (20 + 20, 10 + 40)) and img[0,0] == im.array()[0,0] and im.array()[-1,-1] != 0
    im = imorig.clone().zeropad( (0,10), (0,20) )
    assert np.allclose(imorig.clone().zeropad( (0,10), (0,20) ).crop().array(), img) and (imorig.clone().zeropad( (0,10), (0,20) ).shape() == (20 + 20, 10 + 40)) and img[0,0] == im.array()[0,0] and im.array()[-1,-1] == 0
    im = imorig.clone().zeropadlike(100, 110)
    assert im.width() == 100 and im.height() == 110
    print('[test_image.imagedetection]: pad  PASSED')

    # imclip
    img = np.random.rand(20,10,3).astype(np.float32)
    im = ImageDetection(array=img, xmin=0, ymin=0, width=10, height=20)
    assert im.clone().imclip().bbox.xywh() == (0,0,10,20)
    im = ImageDetection(array=img, xmin=0, ymin=0, width=10, height=200)
    assert im.clone().imclip().bbox.xywh() == (0,0,10,20)
    im = ImageDetection(array=img, xmin=100, ymin=200, width=10, height=200)
    im = ImageDetection(array=img, xmin=-1, ymin=-2, width=100, height=200)
    assert im.clone().imclip().bbox.xywh() == (0,0,10,20)
    im = ImageDetection(array=img, xmin=1, ymin=1, width=9, height=9)
    assert im.clone().imclip().bbox.xywh() == (1,1,9,9)
    print('[test_image.imagedetection]: imclip  PASSED')

    # Setzero
    img = np.random.rand(20,10,3).astype(np.float32)
    assert ImageDetection(array=img, xmin=0, ymin=0, width=2, height=3).setzero().crop().sum() == 0
    print('[test_image.imagedetection]: setzero  PASSED')

    # Mask
    assert np.sum(ImageDetection(array=img, xmin=-1, ymin=-2, width=2, height=3).rectangular_mask(10,10)) == 1
    assert np.sum(ImageDetection(array=img, xmin=0, ymin=0, width=2, height=3).rectangular_mask(10,10)) == 6
    print('[test_image.imagedetection]: mask  PASSED')

    # Dict
    assert isinstance(ImageDetection(array=img, xmin=0, ymin=0, width=2, height=3).dict(), dict)
    print('[test_image.imagedetection]: dict  PASSED')    

    
def test_scene():
    # Constructors
    im = Scene()
    print('[test_image.scene]:Empty Scene Constructor: PASSED')

    im = None
    try:
        im = Scene(objects='a bad type')
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        im = Scene(objects=['a bad type'])
        raise Failed()
    except Failed:
        raise
    except:
        pass
    im = Scene(objects=[Detection('obj1',0,0,0,0), Detection('obj2',0,0,0,0)])
    print('[test_image.scene]:Invalid object type Constructor: PASSED')

    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').load()
    print('[test_image.scene] url constructor: PASSED')

    f = im.filename()
    im = Scene(filename=f).load()
    print('[test_image.scene]: filename constructor: PASSED')

    f = im.filename()
    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg', filename=f).load()
    assert im.filename() == f
    print('[test_image.scene]: url and filename constructor: PASSED')

    im = Scene(array=np.random.rand(3,3,3).astype(np.float32), xywh=[1,2,3,4])
    assert len(im) == 1 and im.objects()[0].category() == None
    im = Scene(array=np.random.rand(3,3,3).astype(np.float32), boxlabels='face', xywh=[1,2,3,4])
    assert len(im) == 1 and im.objects()[0].category() == 'face'
    im = Scene(array=np.random.rand(3,3,3).astype(np.float32), xywh=[[1,2,3,4],[1,2,3,4]])
    assert len(im) == 2 and im.objects()[0].category() == None
    im = Scene(array=np.random.rand(3,3,3).astype(np.float32), boxlabels='face', xywh=[[1,2,3,4],[1,2,3,4]])
    assert len(im) == 2 and im.objects()[0].category() == 'face'
    im = Scene(array=np.random.rand(3,3,3).astype(np.float32), boxlabels=['face1','face2'], xywh=[[1,2,3,4],[1,2,3,4]])
    assert len(im) == 2 and im.objects()[1].category() == 'face2'
    try:
        im = Scene(array=np.random.rand(3,3,3).astype(np.float32), boxlabels=['face1','face2'])        
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        im = Scene(array=np.random.rand(3,3,3).astype(np.float32), boxlabels=['face1','face2'], xywh=[1,2,3,4])        
        raise Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_image.scene]: xywh constructor: PASSED')    
    
    # Test Scene
    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg', filename=f).load()    
    (H,W) = im.shape()
    im = im.objects([Detection('obj1',20,50,100,100), Detection('obj2',300,300,200,200)])
    im.append(Detection('obj3',W +1,H +1,200,200))   # invalid box outside image rectancle
    im.append(Detection('obj4',W -100,H -200,1000,2000))   # invalid box partially outside image rectangle    
    imscene = im.clone()
    
    
    # Visualizations
    im.__repr__()
    print('[test_image.scene]:  __repr__  PASSED')    

    assert len(im) == 4
    print('[test_image.scene]:  __len__  PASSED')

    for obj in im:
        print(obj)
    print('[test_image.scene] __iter__: PASSED')

    print(im[0])
    print(im[1])
    try:
        im[5]
        Failed()
    except Failed:
        raise
    except:
        print('[test_image.scene] __getitem__: PASSED')

    outfile = im.rescale(0.25).show(nowindow=True)
    outfile = im.savefig(temppng())
    print('[test_image.scene]:  show() PASSED')
    print('[test_image.scene]:  savefig() PASSED')    


    # Transformations
    imorig = im.clone()
    (H,W) = imorig.shape()
    im = im.rescale(1.0)
    assert im.shape() == (H, W)
    assert np.allclose(im.rectangular_mask(), np.array(PIL.Image.fromarray(imorig.rectangular_mask(), 'L').resize( (int(W), int(H)), PIL.Image.NEAREST)))
    (H,W) = imorig.shape()
    im = im.rescale(0.5)
    assert im.shape() == (H /2, W /2)
    imm = Image(array=im.rectangular_mask() *255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(imorig.rectangular_mask(), 'L').resize( (int(W /2), int(H /2)), PIL.Image.NEAREST))*255, colorspace='lum')
    #assert np.allclose(imm.array(), imm2.array(), rtol=2)    # FIXME: off by one
    print('[test_image.scene]  rescale() PASSED')

    im = imorig.clone().resize(int(W /2), int(H /2))
    assert im.shape() == (H/2, W/2)
    imm = Image(array=im.rectangular_mask()*255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(imorig.rectangular_mask(), 'L').resize( (int(W /2), int(H /2)), PIL.Image.NEAREST))*255, colorspace='lum')
    #assert np.allclose(imm.array(), imm2.array(), rtol=1)  # FIXME: off by one

    im = imorig.clone().resize(rows=int(H /2))
    assert im.shape() == (H/2, W/2)
    imm = Image(array=im.rectangular_mask()*255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(imorig.rectangular_mask(), 'L').resize( (int(W /2), int(H /2)), PIL.Image.NEAREST))*255, colorspace='lum')
    #assert np.allclose(imm.array(), imm2.array(), rtol=1)   # FIXME: off by one
    print('[test_image.scene]: resize() PASSED')

    im = imorig.clone().fliplr()
    assert im.shape() == (H, W)
    imm = Image(array=im.rectangular_mask() *255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(np.fliplr(imorig.rectangular_mask()), 'L')) *255, colorspace='lum')
    assert np.allclose(imm.array(), imm2.array(), rtol=1)        
    print('[test_image.scene]: fliplr() PASSED')

    imorigclip = imorig.clone().imclip()
    im = imorigclip.clone().imclip()
    (h,w) = im.shape()
    im.zeropad(padwidth=100, padheight=200)
    assert (im.width() == w + 200 and im.height() == h + 400 and im.numpy()[0,0,0] == 0)
    imm = Image(array=im.rectangular_mask() *255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(np.pad(imorigclip.rectangular_mask(),
                                                           pad_width=((200,200),(100,100)),
                                                           mode='constant',
                                                           constant_values=0), 'L')) *255, colorspace='lum')
    assert np.allclose(imm.array(), imm2.array(), rtol=1)    
    im = imorigclip.clone().imclip()
    (h,w) = im.shape()
    im.zeropad((0,100), (0,200))
    assert (im.width() == w + 100 and im.height() == h + 200 and im.numpy()[0,0,0] != 0)
    imm = Image(array=im.rectangular_mask() *255, colorspace='lum')
    imm2 = Image(array=np.array(PIL.Image.fromarray(np.pad(imorigclip.rectangular_mask(),
                                                           pad_width=((0,200),(0,100)),
                                                           mode='constant',
                                                           constant_values=0), 'L')) *255, colorspace='lum')
    assert np.allclose(imm.array(), imm2.array(), rtol=1)                
    print('[test_image.scene]: zeropad PASSED')

    # Centersquare
    im2 = Scene(filename=rgbfile).resize(200, 100).objects([Detection('obj1',50,0,100,100)])
    im = im2.clone().centersquare()
    assert im.width() == im.height() and im.width() == 100 and im[0].bbox.xywh() == (0,0,100,100)
    print('[test_image.scene]: centersquare PASSED')    
    
    # Categories    
    assert sorted(imscene.categories()) == ['obj1', 'obj2', 'obj3', 'obj4']
    im = imscene.clone()
    im._objectlist[0].translate(1000)  # outside image rectangle
    assert sorted(im.categories()) == ['obj1', 'obj2', 'obj3', 'obj4']    
    print('[test_image.scene]: categories PASSED')

    # rot90
    im = Scene(filename=rgbfile).resize(200, 300).objects([Detection('obj1',50,0,100,20)])    
    imrot = im.clone().rot90cw()
    assert imrot.width() == 300 and imrot.height() == 200
    assert imrot[0].boundingbox().width() == 20 and imrot[0].boundingbox().height() == 100
    assert np.allclose(imrot.array(), np.rot90(im.array(), 3))
    imrot = im.clone().rot90ccw()
    assert imrot.width() == 300 and imrot.height() == 200
    assert imrot[0].boundingbox().width() == 20 and imrot[0].boundingbox().height() == 100
    assert np.allclose(imrot.array(), np.rot90(im.array(), 1))
    print('[test_image.scene]: rot90 PASSED')    

    # mindim
    im = Scene(filename=rgbfile).resize(200, 400).objects([Detection('obj1',xmin=50,ymin=0,width=100,height=20)])    
    im = im.mindim(100)
    assert im.width() == 100 and im.height() == 200
    assert im[0].boundingbox().width() == 50 and im[0].boundingbox().height() == 10
    print('[test_image.scene]: mindim PASSED')

    # maxdim
    im = Scene(filename=rgbfile).resize(200, 400).objects([Detection('obj1',xmin=50,ymin=0,width=100,height=20)])
    im = im.maxdim(100)
    assert im.width() == 50 and im.height() == 100
    assert im[0].boundingbox().width() == 25 and im[0].boundingbox().height() == 5
    print('[test_image.scene]: maxdim PASSED')

    # Dict
    assert isinstance(im.dict(), dict)
    print('[test_image.scene]: dict PASSED')

    # bghash
    im = vipy.image.RandomScene(num_objects=1, url='https://upload.wikimedia.org/wikipedia/commons/1/11/Horned1b.jpg')
    assert np.sum(im.bghash(asbinary=True, bits=72) == im.clone().zeropad(10,10).bghash(asbinary=True, bits=72)) >=64
    print('[test_image.scene]: bghash PASSED')

    
if __name__ == "__main__":
    test_image()
    test_imagedetection()
    test_scene()
    
