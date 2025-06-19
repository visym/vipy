import os
import vipy.downloader
import vipy.dataset
from vipy.util import remkdir, tocache, try_import
from vipy.image import ImageCategory


IMAGE_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
ANNO_URL = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
SHA1 = 'f5b09dbcd82e3eb09ef97f265ff6d0ae95b75a80'


class Flowers102(vipy.dataset.Dataset):
    """Project: https://www.robots.ox.ac.uk/~vgg/data/flowers/102"""
    def __init__(self, datadir=None, redownload=False):
        # Download (if not cached)
        datadir = tocache('flowers102') if datadir is None else datadir
        
        self._datadir = remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(IMAGE_URL, self._datadir, sha1=None)            

        # Read cached JSON
        jsonfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'oxford_flowers_102.json')
        if not os.path.exists(jsonfile):
            self._cache_annotations(jsonfile)
        self._json = vipy.util.readjson(jsonfile)

        # Create dataset
        imlist = tuple((f, self._json['labelindex_to_category'][self._json['imageindex_to_labelindex'][k]]) for (k,f) in enumerate(sorted(vipy.util.findimages(self._datadir))))
        loader = lambda x: vipy.image.ImageCategory(filename=x[0], category=x[1])
        super().__init__(imlist, id='flowers102', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()
        
    def _cache_annotations(self, outjson='oxford_flowers_102.json'):        
        if not os.path.exists(os.path.join(self._datadir, 'imagelabels.mat')):
            vipy.downloader.download(ANNO_URL, os.path.join(self._datadir, 'imagelabels.mat'), sha1=None)            
        
        # Thanks to: https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt
        category = ['pink primrose',
                    'hard-leaved pocket orchid',
                    'canterbury bells',
                    'sweet pea',
                    'english marigold',
                    'tiger lily',
                    'moon orchid',
                    'bird of paradise',
                    'monkshood',
                    'globe thistle',
                    'snapdragon',
                    "colt's foot",
                    'king protea',
                    'spear thistle',
                    'yellow iris',
                    'globe-flower',
                    'purple coneflower',
                    'peruvian lily',
                    'balloon flower',
                    'giant white arum lily',
                    'fire lily',
                    'pincushion flower',
                    'fritillary',
                    'red ginger',
                    'grape hyacinth',
                    'corn poppy',
                    'prince of wales feathers',
                    'stemless gentian',
                    'artichoke',
                    'sweet william',
                    'carnation',
                    'garden phlox',
                    'love in the mist',
                    'mexican aster',
                    'alpine sea holly',
                    'ruby-lipped cattleya',
                    'cape flower',
                    'great masterwort',
                    'siam tulip',
                    'lenten rose',
                    'barbeton daisy',
                    'daffodil',
                    'sword lily',
                    'poinsettia',
                    'bolero deep blue',
                    'wallflower',
                    'marigold',
                    'buttercup',
                    'oxeye daisy',
                    'common dandelion',
                    'petunia',
                    'wild pansy',
                    'primula',
                    'sunflower',
                    'pelargonium',
                    'bishop of llandaff',
                    'gaura',
                    'geranium',
                    'orange dahlia',
                    'pink-yellow dahlia?',
                    'cautleya spicata',
                    'japanese anemone',
                    'black-eyed susan',
                    'silverbush',
                    'californian poppy',
                    'osteospermum',
                    'spring crocus',
                    'bearded iris',
                    'windflower',
                    'tree poppy',
                    'gazania',
                    'azalea',
                    'water lily',
                    'rose',
                    'thorn apple',
                    'morning glory',
                    'passion flower',
                    'lotus',
                    'toad lily',
                    'anthurium',
                    'frangipani',
                    'clematis',
                    'hibiscus',
                    'columbine',
                    'desert-rose',
                    'tree mallow',
                    'magnolia',
                    'cyclamen ',
                    'watercress',
                    'canna lily',
                    'hippeastrum ',
                    'bee balm',
                    'ball moss',
                    'foxglove',
                    'bougainvillea',
                    'camellia',
                    'mallow',
                    'mexican petunia',
                    'bromelia',
                    'blanket flower',
                    'trumpet creeper',
                    'blackberry lily', '102']
        
        labelindex_to_category = {str(k):c for (k,c) in enumerate(category, start=1)}  # one-indexed

        # Import, cache and reuse JSON
        try_import('scipy.io', 'scipy')
        import scipy.io
        mat = scipy.io.loadmat(os.path.join(self._datadir, 'imagelabels.mat'))
        imageindex_to_labelindex = [str(c) for c in mat['labels'][0]]

        return vipy.util.writejson({'imageindex_to_labelindex': imageindex_to_labelindex, 'labelindex_to_category':labelindex_to_category}, outjson)
