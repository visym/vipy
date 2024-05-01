import os
import vipy


URLS = ['http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar']


class StanfordDogs(vipy.dataset.Dataset):
    def __init__(self, datadir=vipy.util.tocache('stanford_dogs')):
        self._datadir = vipy.util.remkdir(datadir)

        for url in URLS:
            if not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir)

                
        # Read cached JSON
        jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stanford_dogs.json')
        if not os.path.exists(jsonfile):
            self._cache_annotations(jsonfile)
        self._json = vipy.util.readjson(jsonfile)

        imlist = [(os.path.join(self._datadir, 'dogs_train', f), 
                   d['xmin'], d['ymin'], d['xmax'], d['ymax'],
                   self._json['classidx_to_classname'][str(d['classidx'])])
                  for (f,d) in self._json['filename_to_annotation'].items()]

        loader = lambda x: vipy.image.ImageDetection(filename=x[0], xmin=x[1], ymin=x[2], xmax=x[3], ymax=x[4], category=x[5]) 
        
        super().__init__(imlist, id='stanford_dogs', loader=loader)

