import os
import csv
from bobo.cache import Cache
from bobo.viset.stream import ImageCategoryStream

URL = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'
SHA1 = 'b8ca4fe15bcd0921dfda882bd6052807e63b4c96'
VISET = 'caltech101'
SUBDIR = os.path.join('caltech101', '101_ObjectCategories')

cache = Cache()

def stream():
    csvfile = os.path.join(cache.root(), '%s.csv' % VISET)            
    if not os.path.isfile(csvfile):
        csvfile = export()    
    return ImageCategoryStream(csvfile)

def export():
    key = cache.get(URL, key='%s.tgz' % VISET)
    pkgdir = cache.unpack(key, unpackto=os.path.join(cache.root(), VISET), sha1=SHA1, cleanup=True)

    # Output file
    outfile = cache.abspath('%s.csv' % VISET)    
    
    # Return json or CSV file containing dataset description    
    categorydir = os.path.join(cache.root(), SUBDIR)          

    # Write to annotation stream
    with open(outfile, 'wb') as csvfile:        
        f = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (idx_category, category) in enumerate(os.listdir(categorydir)):
            imdir = os.path.join(categorydir, category)        
            for im in os.listdir(imdir):
                f.writerow([cache.key(os.path.join(categorydir, category, im)), category]);

    # Return CSV file
    return outfile




