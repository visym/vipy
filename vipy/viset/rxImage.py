import os
import bobo.app
from bobo.util import remkdir, isstring, filebase, quietprint, tolist, islist, is_hiddenfile
from bobo.image import Image, ImageDetection
from bobo.viset.stream import ImageStream
from janus.template import GalleryTemplate
from bobo.geometry import BoundingBox

class rxImage(object):            
    def __init__(self, datadir=None, sparkContext=None):
        self.datadir = os.path.join(bobo.app.datadir(), 'rxImage') if datadir is None else datadir        
        if not os.path.isdir(os.path.join(self.datadir)):
            raise ValueError('Download rxImages dataset manually to "%s" ' % self.datadir)
        self.sparkContext = bobo.app.init('pir-rxImage') if sparkContext is None else sparkContext                    

    def __repr__(self):
        return str('<pir.dataset.rxImage: %s>' % self.datadir)
                                
    def csvfile(self):
        visetdir = self.datadir
        return os.path.join(visetdir, 'rxImages.csv')

    def rdd_from_text(self):
        """Create a resilient distributed dataset from csv file"""
        visetdir = os.path.join(self.datadir)
        csvfile = self.csvfile()
    
        # Parse CSV file based on protocol into RDD of ImageDetection objects, all non-detection properties go in attributes dictionary
        # imgName|ndc11|color|shape|size|txt|txtType|txtColor|name|labeler
        # 00185-0615-01.jpg|00185-0615-01|GREEN, WHITE|CAPSULE|15|E615|PRINTED|BLACK|Hydroxyzine Hydrochloride 50 MG Oral Capsule|Eon Labs, Inc.
        schema = ['FILE', 'NDC11', 'COLOR', 'SHAPE', 'SIZE', 'TEXT', 'TEXT_TYPE', 'TEXT_COLOR', 'NAME', 'LABELER']
        sc = self.sparkContext
        rdd = (sc.textFile(csvfile)
                   .map(lambda row: row.encode('utf-8').split('|'))  # Pipe-delimited columns
                   .filter(lambda x: x[0] != 'imgName')  # hack to ignore first row
                   .map(lambda x: ImageDetection(filename=os.path.join(visetdir, x[0]), category=x[1],
                                                 attributes={k:v for (k,v) in zip(schema,x)})))  # Parse row into ImageDetection objects

        return rdd

    def rdd(self):
        """Create a resilient distributed dataset from csv file"""
        csvfile = self.csvfile()
        visetdir = self.datadir
    
        # Parse CSV file based on protocol into RDD of ImageDetection objects, all non-detection properties go in attributes dictionary
        schema = ['NDC11', 'FILE', 'SIGHTING_ID', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'COLOR', 'SHAPE', 'SIZE', 'TEXT', 'TEXT_TYPE', 'TEXT_COLOR', 'NAME', 'LABELER']
        sc = self.sparkContext
        rdd = (sc.textFile(csvfile)
                   .map(lambda row: row.encode('utf-8').split(','))  # Pipe-delimited columns
                   .filter(lambda x: x[0] != 'NDC11')  # hack to ignore first row
                   .map(lambda x: ImageDetection(filename=os.path.join(visetdir, x[1]), category=x[0],
                                                 xmin=float(x[2]), ymin=float(x[3]), xmax=float(x[4]), ymax=float(x[5]),
                                                 attributes={k:v for (k,v) in zip(schema,x)})))  # Parse row into ImageDetection objects

        return rdd

    def rdd_as_templates(self):
        """Create a resilient distributed dataset of Gallery Templates from a file"""
        return (self.rdd()  # RDD of Image(im)
                .keyBy(lambda im: '%s' % (str(im.attributes['NDC11'])))  # (TEMPLATEID, im)
                .reduceByKey(lambda a,b: tolist(a)+tolist(b))  # list of images for each template
                .map(lambda (k,medialist): GalleryTemplate(media=tolist(medialist))))  # construct template as medialist of images and videos assigned to template id

    def as_templates(self):
        return self.rdd_as_templates().collect()

    def as_detections(self):
        return self.rdd().collect()
                                            
    def stats(self):
        n_detections = self.rdd().count()
        n_templates = self.rdd_as_templates().count()        
        return {'numDetections':n_detections, 'numTemplates':n_templates}

    def stream(self):
        csvfile = self.csvfile()
        visetdir = self.datadir
        def parser(row):
            x = row.encode('utf-8').split(',') if not islist(row) else row  # CSV to list
            schema = ['NDC11', 'FILE', 'SIGHTING_ID', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'COLOR', 'SHAPE', 'SIZE', 'TEXT', 'TEXT_TYPE', 'TEXT_COLOR', 'NAME', 'LABELER']
            return(ImageDetection(filename=os.path.join(visetdir, x[1]), category=x[0],
                                  xmin=float(x[2]), ymin=float(x[3]), xmax=float(x[4]), ymax=float(x[5]),
                                  attributes={k:v for (k,v) in zip(schema,x)}))  # Parse row into ImageDetection objects

        return ImageStream(csvfile, parser=parser, delimiter=',', rowstart=2)

