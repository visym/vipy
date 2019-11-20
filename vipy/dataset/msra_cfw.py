import os
import csv
import bobo.util
from bobo.util import remkdir, quietprint, filetail, fileext
from bobo.image import ImageCategory
import bobo.app

class CelebrityFacesInTheWild(object):
    def __init__(self, datadir=None):
        self.datadir = bobo.app.datadir() if datadir is None else datadir
        self.imdir = os.path.join(self.datadir, 'msra_cfw')

        raise ValueError("FIXME")
        if not os.path.isfile(os.path.join(self.imdir, 'dev_urls.txt')):
            raise ValueError('Download PubFig dev_urls.txt manually and save to "%s" ' % self.imdir)
        
        
    def __repr__(self):
        return str('<viset.cfw: %s>' % self.imdir)



    def rdd(sparkContext, ignore=True, fetch=True):
        """Create a resilient distributed dataset"""
        quietprint("[bobo.viset.msra_cfw]: Creating spark RDD for viset '%s'" % VISET)        
        csvfile = os.path.join(CACHE.root(), VISET, '%s.csv' % VISET)            
        if not os.path.isfile(csvfile):
            csvfile = export()
    
        # Parse CSV into RDD stream of ImageCategory objects
        return (sparkContext.textFile(csvfile)  # RDD of textfile lines
                .map(lambda row: row.encode('utf-8').split())  # CSV to list
                .map(lambda x: ImageCategory(url=x[0], key=os.path.join(VISET, x[2], x[1]),  category=x[2], ignore=ignore, fetch=fetch)))  # List to image category 
        
    def export():
        # Output file
        outfile = os.path.join(CACHE.root(), VISET, '%s.csv' % VISET)
    
        # Write images and annotations
        categorydir = os.path.join(CACHE.root(), VISET, VISET)
        k_img = 0
        with open(outfile,'w') as csvfile:
            for (idx_category, category) in enumerate(os.listdir(categorydir)):
                label = bobo.util.tofilename(category)
                if not bobo.util.is_hiddenfile(category):
                    print '[viset.library.msra_cfw]: exporting "%s"' % label
            
                    imdir = os.path.join(categorydir, category)        
                    txtfile = os.path.join(imdir, 'info.txt')
                
                    with open(txtfile,'r') as f:
                        for line in f:
                            row = line.decode('utf-8').strip().split()
                            if len(row) == 3:
                                ext = fileext(row[2])
                                csvfile.write('%s %s %s\n' % (row[2].strip(), '%s_%07d%s' % (label, k_img, ext if ext is not None and len(ext)==4 else ''), label))
                            elif len(row) == 1:
                                ext = fileext(row[0])
                                csvfile.write('%s %s %s\n' % (row[0].strip(), '%s_%07d%s' % (label, k_img, ext if ext is not None and len(ext)==4 else ''), label))
                            else:
                                print 'skipping "%s"' % line 
                            k_img += 1

        return outfile
                
    
    
