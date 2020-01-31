import sys
sys.path.append('..')  # FIXME: relative import
from vipy.image import ImageDetection, Image, ImageCategory, Scene
from vipy.object import Detection

def run():
    # Empty constructor should not raise exception
    im = Image()
    print('Empty Constructor: PASSED')

    # Non-existant filename should not raise exception during constructor (only during load)
    im = Image(filename='myfile')
    print('Filename Constructor: PASSED')

    # Malformed URL should raise exception
    try:
        im = Image(url='myurl')
        raise
    except:
        print('Malformed URL constructor: PASSED')

    # Valid URL should not raise exception (even if it is not an image extension)
    im = Image(url='http://visym.com')
    print('Image URL constructor: PASSED')

    # Valid URL and filename to save it
    im = Image(url='http://visym.com/myfile.jpg', filename='/my/file/path')
    print('URL and filename constructor: PASSED')

    # URL object
    im = Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg')
    print('  Image __desc__: %s' % im)
    print('  Image length: %d' %  len(im))
    im.download()
    print('  Image __desc__: %s' % im)
    im.load()
    print('  Image __desc__: %s' % im)
    print('URL download: PASSED')

    # Valid URL but without an image extension 
    im = Image(url='http://bit.ly/great_horned_owl')
    print('  Image __desc__: %s' % im)
    im.load()
    print('  Image __desc__: %s' % im)
    print('URL download (without image extension): PASSED')
    
    # Invalid URL with ignore
    im = Image(url='https://a_bad_url.jpg')
    print('  Image __desc__: %s' % im)
    im.load(ignoreErrors=True)
    print('  Image __desc__: %s' % im)
    print('Invalid URL download: PASSED')

    # URL with filename 
    im = Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg', filename='/tmp/myfile.jpg')
    print('  Image __desc__: %s' % im)
    print('  Image length: %d' %  len(im))
    im.download()
    print('  Image __desc__: %s' % im)
    im.load()
    print('  Image __desc__: %s' % im)
    print('URL with filename download: PASSED')

    # Filename object
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, width=700, height=1000, category='face')
    print('Image __desc__: %s' % im)
    #im.show(figure=1)
    im.crop()
    print('Image __desc__: %s' % im)
    print('Filename: PASSED')

    # Array constructor

    # Scene
    im = Scene()
    print('Empty Constructor: PASSED')

    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').load()
    print('URL: PASSED')

    im = im.objects([Detection('obj1',50,100,300,300), Detection('obj2',600,600,400,400)])
    im.show(outfile='test_scene.jpg')

    # Image conversion
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, width=700, height=1000, category='face')
    im.rgb()
    im.bgr()
    im.hsv()    
    im.gray()
    im.float()
    print('Image conversion: PASSED')
    
if __name__ == "__main__":
    run()



