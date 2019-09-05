import sys
sys.path.append('..')  # FIXME: relative import
from vipy.image import ImageDetection, Image, ImageCategory

def run():
    # Empty constructor should raise exception
    try:
        im = Image()
        raise
    except:
        print('Empty Constructor: PASSED')

    # Ambiguous constructor should raise exception
    try:
        im = Image(filename='myfile', url='myurl')
        raise
    except:
        print('Ambiguous Constructor: PASSED')

    # URL constructor
    im = Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg')
    print(im)
    print(len(im))
    im.load()
    print('URL: PASSED')

    # Filename constructor
    im = ImageDetection(filename='jebyrne.jpg', xmin=100, ymin=100, width=700, height=1000, category='face')
    im.show(figure=1)
    im.crop().show()
    print('Filename: PASSED')


    # Array constructor
    

if __name__ == "__main__":
    run()



