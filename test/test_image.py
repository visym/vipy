import os
import numpy as np
import vipy.image
from vipy.image import ImageDetection, Image, ImageCategory, Scene
from vipy.object import Detection
from vipy.util import tempjpg, tempdir
from test_vipy import TestFailed
from vipy.geometry import BoundingBox

def image():
    
    # Common Parameters
    jpegurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg'
    gifurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Rotating_earth_%28large%29.gif/200px-Rotating_earth_%28large%29.gif'
    pngurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/560px-PNG_transparency_demonstration_1.png'
    greyfile = 'jebyrne_grey.jpg'
    rgbfile = 'jebyrne.jpg'    
    
    # Empty constructor should not raise exception
    im = Image()
    print('[test_image.image]: Empty Constructor: PASSED')

    # Non-existant filename should not raise exception during constructor (only during load)
    im = Image(filename='myfile')
    print('[test_image.image]: Filename Constructor: PASSED')

    # Malformed URL should raise exception
    im = None
    try:
        im = Image(url='myurl');
        raise TestFailed()
    except TestFailed:
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
    print('[test_image.image]:   Image length: %d' %  len(im))
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
    im = Image(url=jpegurl, filename='/tmp/myfile.jpg')
    print('[test_image.image]:   Image __desc__: %s' % im)
    print('[test_image.image]:   Image length: %d' %  len(im))
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
               
    # Filename object
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, bbwidth=700, bbheight=1000, category='face')
    print('[test_image.image]: Image __desc__: %s' % im)
    im.crop()
    print('[test_image.image]: Image __desc__: %s' % im)
    print('[test_image.image]: Filename: PASSED')

    # Clone
    im = Image(filename='jebyrne.jpg').load()
    imb = im
    im._array = im._array + 1  # modify array
    np.testing.assert_array_equal(imb.numpy(), im.numpy())  # share buffer
    imc = im.clone().flush().load()
    assert(np.any(im.numpy() != imc.numpy()))  # does not share buffer
    print('[test_image.image]: Image.clone: PASSED')    

    # Saveas
    im = Image(filename='jebyrne.jpg').load()
    f = tempjpg()    
    assert im.saveas(f) == f and os.path.exists(f)
    print('[test_image.image]: Image.saveas: PASSED')        

    # Stats
    im = Image(filename='jebyrne.jpg').load().stats()
    print('[test_image.image]: Image.stats: PASSED')        

    # Resize
    f = tempjpg()
    im = Image(filename='jebyrne.jpg').load().resize(cols=16,rows=8).saveas(f)
    assert Image(filename=f).shape() == (8,16)
    assert Image(filename=f).width() == 16
    assert Image(filename=f).height() == 8
    im = Image(filename='jebyrne.jpg').load().resize(16,8).saveas(f)
    assert Image(filename=f).shape() == (8,16)
    assert Image(filename=f).width() == 16
    assert Image(filename=f).height() == 8
    im = Image(filename='jebyrne.jpg').load()
    (h,w) = im.shape()
    im = im.resize(rows=16)
    assert im.shape() == (16,int((w/float(h))*16.0))
    print('[test_image.image]: Image.resize: PASSED')        

    # Rescale
    f = tempjpg()
    im = Image(filename='jebyrne.jpg').load().resize(rows=8).saveas(f)
    assert Image(filename=f).height() == 8
    im = Image(filename='jebyrne.jpg').load().resize(cols=8).saveas(f)
    assert Image(filename=f).width() == 8
    im = Image(filename='jebyrne.jpg').load().maxdim(256).saveas(f)
    assert np.max(Image(filename=f).shape()) == 256
    print('[test_image.image]: Image.rescale: PASSED')        

    # GIF
    im = Image(url=gifurl)
    im.download(verbose=True)
    assert im.shape() == (200,200)
    print('[test_image.image]: GIF: PASSED')

    # Transparent PNG
    im = Image(url=pngurl)
    im.load(verbose=True)
    print('[test_image.image]: PNG: PASSED')

    # Image colorspace conversion
    im = Image(filename='jebyrne.jpg').resize(200,200)
    print(im.rgb())
    assert(im.shape() == (200,200) and im.channels() == 3)
    print(im.bgr())
    assert(im.shape() == (200,200) and im.channels() == 3)    
    print(im.rgba())
    assert(im.shape() == (200,200) and im.channels() == 4)        
    print(im.hsv())
    assert(im.shape() == (200,200) and im.channels() == 3)            
    print(im.bgra())
    assert(im.shape() == (200,200) and im.channels() == 4)                
    print(im.gray())
    assert(im.shape() == (200,200) and im.channels() == 1)                    
    print(im.float())
    assert(im.shape() == (200,200) and im.channels() == 1)                    
    print('[test_image.image]: Image conversion: PASSED')

    # Image colorspace conversion
    im = Image(filename='jebyrne_grey.jpg').load()
    assert im.attributes['colorspace'] == 'grey'
    assert im.max() == 255
    print('[test_image.image]: Greyscale image conversion: PASSED')
    
    # Image colormaps
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face').crop()
    im.rgb().jet().bone().hot().rainbow()
    print('[test_image.image]: Image colormaps: PASSED')

    # ImageDetection
    im = ImageDetection()
    im.__repr__()
    assert im.category() is None and im.bbox.xywh() == (0,0,0,0)
    print('[test_image.image] Empty ImageDetection constructor PASSED')
    im = ImageDetection(category='test', xmin=0, ymin=0, width=200, height=200)
    print('[test_image.image] Empty filename ImageDetection constructor PASSED')
    print('[test_image.image] Empty bounding box ImageDetection constructor PASSED')

    im = ImageDetection(filename='jebyrne.jpg', category='face', bbox=BoundingBox(0,0,100,100))
    try:
        im = ImageDetection(filename='jebyrne.jpg', category='face', bbox='a_bad_type')
        raise TestFailed()
    except TestFailed:
        raise
    except:
        print('[test_image.image] ImageDetection bounding box constructor PASSED')

    im = ImageDetection(filename='jebyrne.jpg', category='face', xmin=-1, ymin=-2, ymax=10, xmax=20)
    try:
        im = ImageDetection(filename='jebyrne.jpg', category='face', xmin='a_bad_type', ymin=-2, ymax=10, xmax=20)        
        raise TestFailed()
    except TestFailed:
        raise
    except:
        print('[test_image.image] ImageDetection (xmin,ymin,xmax,ymax) bounding box constructor PASSED')


    # Image detections - invalid box
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, bbwidth=200, bbheight=-200, category='face')
    assert im.invalid()
    print('[test_image.image]: ImageDetection invalid box: PASSED')
    try:
        im.crop();
        raise TestFailed()
    except TestFailed:
        raise
    except:
        print('[test_image.image]: ImageDetection invalid box crop: PASSED')        
    im = ImageDetection(filename='jebyrne.jpg', xmin=100000, ymin=100000, bbwidth=200, bbheight=200, category='face')
    assert im.boxclip().bbox is None
    print('[test_image.image]: ImageDetection invalid imagebox: PASSED')            

    
    im = ImageDetection(filename='jebyrne.jpg', category='face', xmin=-1, ymin=-2, ymax=10, xmax=20)
    assert im.boundingbox() == BoundingBox(-1, -2, 20, 10)
    assert not im.boundingbox(xmin=0, ymin=0, width=10, height=20) == BoundingBox(0,0,width=10, height=20)
    assert im.boundingbox(xmin=0, ymin=0, width=10, height=20).bbox == BoundingBox(0,0,width=10, height=20)    
    assert im.boundingbox(bbox=BoundingBox(1,2,3,4)).bbox == BoundingBox(1,2,3,4)
    assert im.boundingbox(bbox=BoundingBox(xcentroid=1,ycentroid=2,width=3,height=4)).bbox == BoundingBox(xcentroid=1,ycentroid=2,width=3,height=4)
    assert im.boundingbox(xmin=-1, ymin=-2, ymax=10, xmax=20).boundingbox(dilate=1.5).bbox == BoundingBox(xmin=-1, ymin=-2, ymax=10, xmax=20).dilate(1.5)
    assert im.boundingbox(xmin=-1, ymin=-2, ymax=10, xmax=20).boundingbox(dilate=1.5).bbox == BoundingBox(xmin=-1, ymin=-2, ymax=10, xmax=20).dilate(1.5)    
    try:
        im.boundingbox(xmin=1)
        raise TestFailed()
    except TestFailed:
        raise
    except:
        print('[test_image.image] ImageDetection.boundingbox()  PASSED')
    
    
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, xmax=200, ymax=200, category='face').crop()    
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face').crop()
    assert(im.shape() == (200,200))
    assert(im.bbox.width() == im.width() and im.bbox.height() == im.height() and im.bbox.xmin()==0 and im.bbox.ymin()==0)
    print('[test_image.image] ImageDetection crop  PASSED')
    
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, bbwidth=200, bbheight=200, category='face')
    im = im.rescale(0.5)
    assert(im.bbox.width() == 100 and im.bbox.height() == 100)
    print('[test_image.image] ImageDetection rescale  PASSED')

    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, width=200, height=300, category='face')
    im = im.resize(cols=int(im.width()//2.0), rows=int(im.height()//2.0))
    assert(im.bbox.width() == 100 and im.bbox.height() == 150)
    print('[test_image.image] ImageDetection resize  PASSED')
    
    im = ImageDetection(array=np.zeros( (10,10) ), xmin=100, ymin=100, xmax=200, ymax=200, category='face')
    assert not im.isinterior()
    assert not im.isinterior(10,10)    
    print('[test_image.image] ImageDetection interior  PASSED')
    

    
    
    # Image category
    im = ImageCategory(filename='jebyrne.jpg', category='face')
    assert im.load().category() == 'face'
    print('[test_image.image]: ImageCategory constructor PASSED')                
    assert ImageCategory(category='1') == ImageCategory(category='1')
    assert ImageCategory(category='1') != ImageCategory(category='2')
    print('[test_image.image]: ImageCategory equivalence PASSED')
    assert ImageCategory(category='1').iscategory('1')
    assert not ImageCategory(category='1').iscategory('2')
    assert ImageCategory(category='1').ascategory('2').iscategory('2')    
    print('[test_image.image]: ImageCategory category conversion PASSED')    
    im.score(1.0)
    im.probability(1.0)
    
    
    # Random images
    im = vipy.image.RandomImage(128,256)
    assert im.shape() == (128, 256)
    print('[test_image.image]: RandomImage PASSED')    
    im = vipy.image.RandomImageDetection(128,256)
    assert im.clone().crop().width() == im.bbox.imclipshape(W=256,H=128).width()
    print('[test_image.image]: RandomImageDetection PASSED')                        

    
def scene():
    im = Scene()
    print('[test_image.scene]:Empty Scene Constructor: PASSED')

    im = None
    try:
        im = Scene(objects='a bad type');
        raise TestFailed()
    except TestFailed:
        raise
    except:
        pass
    try:
        im = Scene(objects=['a bad type']);
        raise TestFailed()
    except TestFailed:
        raise
    except:
        pass
    im = Scene(objects=[Detection('obj1',0,0,0,0), Detection('obj2',0,0,0,0)])        
    print('[test_image.scene]:Invalid object type Constructor: PASSED')

    assert sorted(im.categories()) == ['obj1', 'obj2']
    print('[test_image.scene]:Scene.categories: PASSED')    
    
    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').load()
    print('[test_image.scene]:Scene.url: PASSED')
    im = im.rescale(0.5).objects([Detection('obj1',20,50,100,100), Detection('obj2',300,300,200,200)])
    print('[test_image.scene]:Scene.rescale: PASSED')

    outfile = im.show(nowindow=True)
    outfile = im.savefig(tempjpg())
    print('[test_image.scene]:Scene.show().savefig() ("%s"): PASSED' % outfile)
    
    im = im.resize(1000,100)
    outfile = im.show().savefig(tempjpg())
    print('[test_image.scene]:Scene.resize() ("%s"): PASSED' % outfile)    

    (h,w) = im.shape()
    im.zeropad(padwidth=100,padheight=200)    
    assert (im.width() == w+200 and im.height()==h+400 and im.numpy()[0,0,0] == 0)
    print('[test_image.scene]:Scene.zeropad: PASSED')
    
    
if __name__ == "__main__":
    image()
    scene()



