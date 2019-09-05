import sys
sys.path.append('..')  # FIXME: relative import
from vipy.image import ImageDetection, Image, ImageCategory

def run():
    # Empty constructor should not raise exception
    im = Image()
    print('Empty Constructor: PASSED')

    # Non-existant filename should not raise exception
    im = Image(filename='myfile')
    print('Filename Constructor: PASSED')

    # Malformed URL should raise exception
    try:
        im = Image(url='myurl')
        raise
    except:
        print('Malformed URL constructor: PASSED')

    # Valid URL but not image should raise exception
    try:
        im = Image(url='http://visym.com')
        raise
    except:
        print('Image URL constructor: PASSED')

    # Valid URL and file to save it
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

    # Invalid URL with ignore
    im = Image(url='https://a_bad_url.jpg')
    print('  Image __desc__: %s' % im)
    im.load(ignoreErrors=False)
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
    

if __name__ == "__main__":
    run()



