raise ValueError('FIXME')




import os
import csv
from bobo.cache import Cache
from bobo.util import remkdir, isstring
from bobo.image import ImageCategoryStream

URL = 'http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/VOCtrainval_11-May-2012.tar'
SHA1 = None
IMDIR = 'VOCdevkit/VOC2012/JPEGImages'
IMSETDIR = 'VOCdevkit/VOC2012/ImageSets/Main'
VISET = 'pascalvoc_2012'

cache = Cache(subdir=VISET)



def stream(csvfile=None, outdir=None):
    if csvfile is None:
        csvfile = os.path.join(cache.root(), '%s.csv' % VISET)            
    if outdir is not None:
        cache.setroot(outdir)
    if not os.path.isfile(csvfile):
        csvfile = export()        
    return ImageCategoryStream(csvfile, cache=cache)


def export(outdir=None):
    # Update cache
    if outdir is not None:
        cache.setroot(outdir)

    # Fetch data necessary to initial construction
    pkgdir = cache.unpack(cache.get(URL), sha1=SHA1)
    imsetdir = os.path.join(pkgdir, IMSETDIR)                              
    outfile = os.path.join(cache.root(), '%s.csv' % VISET)
    
    # Write images to database
    with open(outfile, 'wb') as csvfile:            
        f = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for (idx_category, imset) in enumerate(os.listdir(imsetdir)):
            (filebase,ext) = os.path.splitext(os.path.basename(imset))
            try:
                (category, set) = filebase.split('_')            
            except:
                continue
            if set == 'trainval':
                for (idx_image, line) in enumerate(open(os.path.join(imsetdir, imset), 'r')):
                    (im, label) = line.split()
                    im = im + '.jpg'
                    if int(label) == 1:
                        f.writerow([os.path.join(IMDIR, im), category])                                    

    return outfile
