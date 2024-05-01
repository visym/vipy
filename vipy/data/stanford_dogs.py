import os
import vipy
from vipy.util import filebase


URLS = ['http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar']


class StanfordDogs(vipy.dataset.Dataset):
    def __init__(self, datadir=vipy.util.tocache('stanford_dogs'), redownload=False):
        self._datadir = vipy.util.remkdir(datadir)

        for url in URLS:
            if redownload or not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir)
                
        # Read cached XML
        xmlfiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self._datadir, 'Annotation')) for f in filenames]

        d_imgname_to_annotation = {}
        for xmlfile in xmlfiles:
            with open(xmlfile, 'r') as f:
                 data = f.read()

            imgname = vipy.util.filetail(xmlfile)
            d_imgname_to_annotation[imgname] = {'xmin':int(data.split('<xmin>',1)[1].split('</xmin>',1)[0]),
                                                'ymin':int(data.split('<ymin>',1)[1].split('</ymin>',1)[0]),
                                                'xmax':int(data.split('<xmax>',1)[1].split('</xmax>',1)[0]),
                                                'ymax':int(data.split('<ymax>',1)[1].split('</ymax>',1)[0])}

        # Read images
        imgfiles = vipy.util.findimages(os.path.join(self._datadir, 'Images'))
        
        imlist = [(f,
                   d_imgname_to_annotation[filebase(f)]['xmin'] if filebase(f) in d_imgname_to_annotation else None,
                   d_imgname_to_annotation[filebase(f)]['ymin'] if filebase(f) in d_imgname_to_annotation else None,
                   d_imgname_to_annotation[filebase(f)]['xmax'] if filebase(f) in d_imgname_to_annotation else None,
                   d_imgname_to_annotation[filebase(f)]['ymax'] if filebase(f) in d_imgname_to_annotation else None,
                   os.path.dirname(f).rsplit('-',1)[1])
                   for f in imgfiles]

        loader = lambda x: vipy.image.ImageDetection(filename=x[0], xmin=x[1], ymin=x[2], xmax=x[3], ymax=x[4], category=x[5])         
        super().__init__(imlist, id='stanford_dogs', loader=loader)

