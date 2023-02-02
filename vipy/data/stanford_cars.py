import os
import vipy


URLS = ['http://ai.stanford.edu/~jkrause/car196/cars_train.tgz',
        'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz',
        'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz']


class StanfordCars(vipy.dataset.Dataset):
    """Project: https://ai.stanford.edu/~jkrause/cars/car_dataset.html"""
    def __init__(self, datadir):
        self._datadir = vipy.util.remkdir(datadir)

        for url in URLS:
            if not os.path.exists(os.path.join(datadir, vipy.util.filetail(url))):
                vipy.downloader.download_and_unpack(url, self._datadir)

        # Read cached JSON
        jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stanford_cars.json')
        if not os.path.exists(jsonfile):
            self._cache_annotations(jsonfile)
        self._json = vipy.util.readjson(jsonfile)

        imlist = [vipy.image.ImageDetection(filename=os.path.join(self._datadir, 'cars_train', f), 
                                            xmin=d['xmin'], ymin=d['ymin'], xmax=d['xmax'], ymax=d['ymax'],
                                            category=self._json['classidx_to_classname'][str(d['classidx'])])
                  for (f,d) in self._json['filename_to_annotation'].items()]
        
        super().__init__(imlist, 'stanford_cars')
                          
    def _cache_annotations(self, outjson='stanford_cars.json'):        
        assert os.path.exists(os.path.join(self._datadir, 'cars_annos.mat'))

        import scipy.io
        mat = scipy.io.loadmat(os.path.join(self._datadir, 'devkit', 'cars_meta.mat'))
        d_classidx_to_classname = {str(k):str(x[0]) for (k,x) in enumerate(mat['class_names'][0], start=1)}
        
        mat_train = scipy.io.loadmat(os.path.join(self._datadir, 'devkit', 'cars_train_annos.mat'))

        return vipy.util.writejson({'classidx_to_classname':d_classidx_to_classname,
                                    'filename_to_annotation':{str(x[5][0]):{'xmin':int(x[0][0][0]), 'ymin':int(x[1][0][0]), 'xmax':int(x[2][0][0]), 'ymax':int(x[3][0][0]), 'classidx':int(x[4][0][0])} for x in mat_train['annotations'][0]}}, outjson)

    def testset(self):        
        return vipy.dataset.Dataset([vipy.image.Image(filename=f) for f in vipy.util.findimages(os.path.join(self._datadir, 'cars_test'))], 'stanford_cars_test')
    
