import os
import vipy


# URLs are broken
URLS = ['http://ai.stanford.edu/~jkrause/car196/cars_train.tgz',
        'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz',
        'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz']


class StanfordCars(vipy.dataset.Dataset):
    """Project: https://ai.stanford.edu/~jkrause/cars/car_dataset.html"""
    def __init__(self, datadir=vipy.util.tocache('stanford_cars'), redownload=False):
        self._datadir = vipy.util.remkdir(datadir)

        for url in URLS:
            if redownload or not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir)

        # Read cached JSON
        jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stanford_cars.json')
        if not os.path.exists(jsonfile):
            self._cache_annotations(jsonfile)
        self._json = vipy.util.readjson(jsonfile)
        
        imlist = [(os.path.join(self._datadir, 'cars_train', f), 
                   d['xmin'], d['ymin'], d['xmax'], d['ymax'],
                   self._json['classidx_to_classname'][str(d['classidx'])])
                  for (f,d) in self._json['filename_to_annotation'].items()]

        loader = lambda x: vipy.image.ImageDetection(filename=x[0], xmin=x[1], ymin=x[2], xmax=x[3], ymax=x[4], category=x[5]) 
        super().__init__(imlist, id='stanford_cars', loader=loader)
                          
    def _cache_annotations(self, outjson='stanford_cars.json'):        
        assert os.path.exists(os.path.join(self._datadir, 'cars_annos.mat'))

        vipy.util.try_import('scipy.io', 'scipy')
        import scipy.io
        mat = scipy.io.loadmat(os.path.join(self._datadir, 'devkit', 'cars_meta.mat'))
        d_classidx_to_classname = {str(k):str(x[0]) for (k,x) in enumerate(mat['class_names'][0], start=1)}
        
        mat_train = scipy.io.loadmat(os.path.join(self._datadir, 'devkit', 'cars_train_annos.mat'))

        return vipy.util.writejson({'classidx_to_classname':d_classidx_to_classname,
                                    'filename_to_annotation':{str(x[5][0]):{'xmin':int(x[0][0][0]), 'ymin':int(x[1][0][0]), 'xmax':int(x[2][0][0]), 'ymax':int(x[3][0][0]), 'classidx':int(x[4][0][0])} for x in mat_train['annotations'][0]}}, outjson)

    def testset(self):
        loader = lambda x: vipy.image.Image(filename=x)
        return vipy.dataset.Dataset(vipy.util.findimages(os.path.join(self._datadir, 'cars_test')), id='stanford_cars_test', loader=loader)
    
