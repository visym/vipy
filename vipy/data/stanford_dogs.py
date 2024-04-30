import os
import vipy


URLS = ['http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar']


class StanfordCars(vipy.dataset.Dataset):
    """Project: https://ai.stanford.edu/~jkrause/cars/car_dataset.html"""
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

        imlist = [vipy.image.ImageDetection(filename=os.path.join(self._datadir, 'cars_train', f), 
                                            xmin=d['xmin'], ymin=d['ymin'], xmax=d['xmax'], ymax=d['ymax'],
                                            category=self._json['classidx_to_classname'][str(d['classidx'])])
                  for (f,d) in self._json['filename_to_annotation'].items()]
        
        super().__init__(imlist, 'stanford_cars')

