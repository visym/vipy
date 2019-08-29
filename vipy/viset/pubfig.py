import os
from bobo.util import remkdir, quietprint, filetail
from bobo.image import ImageCategory
import bobo.app

DEV_IMAGES_URL = 'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt'
#DEV_PEOPLE_URL = 'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_people.txt'
#EVAL_PEOPLE_URL = 'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_people.txt'
EVAL_IMAGES_URL = 'http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt'

DEV_IMAGES_SHA1 = '9eb10c01d46c5d06a8f70b9b8a9ff6b8fe4b0e41';
EVAL_IMAGES_SHA1 = '0fd4cfc464993909c45f9bce1322747c9a9baef9';


class PubFig(object):
    def __init__(self, datadir=None):
        self.datadir = bobo.app.datadir() if datadir is None else datadir
        self.imdir = os.path.join(self.datadir, 'pubfig')
        
        if not os.path.isfile(os.path.join(self.imdir, 'dev_urls.txt')):
            raise ValueError('Download PubFig dev_urls.txt manually and save to "%s" ' % self.imdir)
        
        
    def __repr__(self):
        return str('<viset.pubfig: %s>' % self.imdir)

    
    def rdd(self, sparkContext=None, ignore=True, fetch=True):
        """Create a resilient distributed dataset"""
        sparkContext = bobo.app.init('viset_pubfig') if sparkContext is None else sparkContext
        csvfile = os.path.join(self.imdir, 'dev_urls.txt')
        return (sparkContext.textFile(csvfile)  # RDD of textfile lines
                            .map(lambda row: row.decode('utf-8').rstrip().split())  # CSV to list
                            .filter(lambda r: r[0] not in ['#', '\u#'])
                            .map(lambda x: ImageCategory(url=x[3], filename=os.path.join(self.imdir, filetail(x[3])),  category='%s_%s' % (x[0], x[1]))))
        
    def devlist(self):
        """Return interface objects for this dataset"""
        return self.rdd().collect()

    def download(self):
        return self.rdd().map(lambda im: im.download(ignoreErrors=True, timeout=10)).collect()

