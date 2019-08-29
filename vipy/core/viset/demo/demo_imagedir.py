from viset.library.imagedir import ImageDir
from viset.dataset import Viset
from viset.show import imshow
import sys

if __name__ == '__main__':
    """Create imageviset from directory of images.  First argument is database name, second argument is image directory"""
    dbname = sys.argv[1]
    imdir = sys.argv[2]

    print '[viset.demo_imagedir]: Creating viset "%s" ' % dbname    
    print '[viset.demo_imagedir]: Reading images from directory = %s' % imdir
    dbfile = ImageDir().export(dbname, imdir, verbose=True)
    db = Viset(dbfile, verbose=True)
    for im in db.image:
        imshow(im)

        
