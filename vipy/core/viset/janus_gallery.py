import os
import csv
from glob import glob
from bobo.cache import Cache
from bobo.util import remkdir, isstring, filebase, quietprint, tolist, islist, is_hiddenfile
from bobo.viset.stream import ImageStream
from bobo.image import ImageDetection
from bobo.video import VideoDetection
from janus.template import GalleryTemplate
import bobo.app

VISET = 'Janus_Gallery'

cache = Cache()

def rdd(sparkContext=None, galleryname=None, split=1):
    """Create a resilient distributed dataset from a split file"""
    if galleryname is None:
        raise ValueError('galleryname is required')
    schema = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID', 'FRAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'RIGHT_EYE_X', 'RIGHT_EYE_Y', 'LEFT_EYE_X', 'LEFT_EYE_Y', 'NOSE_BASE_X', 'NOSE_BASE_Y']
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    csvfile = os.path.join(visetdir, '%s.csv' % (galleryname))

    # Parse CSV file based on protocol into RDD of ImageDetection objects, all non-detection properties go in attributes dictionary
    schema = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID', 'FRAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'RIGHT_EYE_X', 'RIGHT_EYE_Y', 'LEFT_EYE_X', 'LEFT_EYE_Y', 'NOSE_BASE_X', 'NOSE_BASE_Y']
    sc = bobo.app.init('janus_gallery') if sparkContext is None else sparkContext
    return (sc.textFile(csvfile)
              .map(lambda row: row.encode('utf-8').split(','))  # CSV to list of row elements
              .filter(lambda x: x[0] != 'TEMPLATE_ID')  # hack to ignore first row
              .map(lambda x: ImageDetection(filename=os.path.join(visetdir, x[2]), category=x[1],
                                            xmin=float(x[6]) if len(x[6]) > 0 else float('nan'),
                                            ymin=float(x[7]) if len(x[7]) > 0 else float('nan'),
                                            xmax = float(x[6])+float(x[8]) if ((len(x[6])>0) and (len(x[8])>0)) else float('nan'),
                                            ymax = float(x[7])+float(x[9]) if ((len(x[7])>0) and (len(x[9])>0)) else float('nan'),
                                            attributes={k:v for (k,v) in zip(schema,x)})))  # Parse row into ImageDetection objects


def rdd_as_templates(sparkContext=None, galleryname=None):
    """Create a resilient distributed dataset of Gallery Templates from a split file"""
    return (rdd(sparkContext=sparkContext, galleryname=galleryname)  # RDD of ImageDetections (im)
            .keyBy(lambda m: str(m.attributes['TEMPLATE_ID'])) # keyby template id only
            .reduceByKey(lambda a,b: tolist(a)+tolist(b))  # list of images/videos for each template
            .map(lambda (k,medialist): GalleryTemplate(media=tolist(medialist))))  # construct template as medialist of images and videos assigned to template id


def rdd_templates_as_video(sparkContext=None, galleryname=None, split=1):
    """Create a resilient distributed dataset from a split file"""
    if galleryname is None:
        raise ValueError('galleryname is required')
    sparkContext = bobo.app.init('janus_gallery') if sparkContext is None else sparkContext
    schema = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID', 'FRAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'RIGHT_EYE_X', 'RIGHT_EYE_Y', 'LEFT_EYE_X', 'LEFT_EYE_Y', 'NOSE_BASE_X', 'NOSE_BASE_Y']
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    csvfile = os.path.join(visetdir, '%s.csv' % (galleryname))

    # Parse CSV file based on protocol, all non-detection properties go in attributes dictionary
    return (sparkContext.textFile(csvfile)
            .map(lambda row: row.encode('utf-8').split(','))  # CSV to list of row elements
            .filter(lambda x: x[0] != 'TEMPLATE_ID')  # hack to ignore first row
            .map(lambda x: ImageDetection(url=cache.key(os.path.join(visetdir, x[2])), category=x[1],
                                          xmin=float(x[6]) if len(x[6]) > 0 else float('nan'),
                                          ymin=float(x[7]) if len(x[7]) > 0 else float('nan'),
                                          xmax = float(x[6])+float(x[8]) if ((len(x[6])>0) and (len(x[8])>0)) else float('nan'),
                                          ymax = float(x[7])+float(x[9]) if ((len(x[7])>0) and (len(x[9])>0)) else float('nan'),
                                          attributes={k:v for (k,v) in zip(schema,x)}))  # Parse row into
            .keyBy(lambda im: str(im.attributes['TEMPLATE_ID']))  # Construct (TemplateID, x) tuples
            .reduceByKey(lambda a,b: tolist(a)+tolist(b))  # Construct (TemplateID, [x1,x2,...,xk]) tuples for videos
            .map(lambda (k,fr): VideoDetection(frames=tolist(fr), attributes=tolist(fr)[0].attributes)))

def stream(galleryname, split=None):
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    csvfile = os.path.join(visetdir, '%s.csv' % (galleryname))
    def parser(row):
        x = row.encode('utf-8').split(',') if not islist(row) else row  # CSV to list
        schema = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID', 'FRAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'RIGHT_EYE_X', 'RIGHT_EYE_Y', 'LEFT_EYE_X', 'LEFT_EYE_Y', 'NOSE_BASE_X', 'NOSE_BASE_Y']
        return (ImageDetection(url=cache.key(os.path.join(visetdir, x[2])), category=x[1],
                               xmin=float(x[6]) if len(x[6]) > 0 else float('nan'),
                               ymin=float(x[7]) if len(x[7]) > 0 else float('nan'),
                               xmax = float(x[6])+float(x[8]) if ((len(x[6])>0) and (len(x[8])>0)) else float('nan'),
                               ymax = float(x[7])+float(x[9]) if ((len(x[7])>0) and (len(x[9])>0)) else float('nan'),
                               attributes={k:v for (k,v) in zip(schema,x)}))

    return ImageStream(csvfile, parser=parser, delimiter=',', rowstart=2)

def export(galleryname):
    # Dataset downloaded?
    if not os.path.isdir(os.path.join(cache.root(), VISET)):
        raise ValueError('Download Janus Gallery dataset manually to "%s" ' % os.path.join(cache.root(), VISET, galleryname))

def visetdir(galleryname):
    """Returns location of this viset"""
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    remkdir(visetdir)
    return visetdir

def nextSightingId(galleryname):
    """Return the next value to be used for a new sighting"""
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    csvfile = os.path.join(visetdir, '%s.csv' % (galleryname))

    nextid = 0
    if os.path.exists(csvfile):
        with open(csvfile, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sightingid = int(row['SIGHTING_ID'])
                    nextid = max(nextid, (sightingid))
                except ValueError:
                    pass

    return nextid+1


def enroll(galleryname, imset):
    """Create or update gallery metadata and image cache with a list of images"""
    visetdir = os.path.join(cache.root(), VISET, galleryname)
    csvfile = os.path.join(visetdir, '%s.csv' % (galleryname))
    imgdir = os.path.join(visetdir, 'img')
    remkdir(imgdir)
    schema = ['TEMPLATE_ID', 'SUBJECT_ID', 'FILE', 'MEDIA_ID', 'SIGHTING_ID', 'FRAME', 'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT', 'RIGHT_EYE_X', 'RIGHT_EYE_Y', 'LEFT_EYE_X', 'LEFT_EYE_Y', 'NOSE_BASE_X', 'NOSE_BASE_Y']

    sighting_id = nextSightingId(galleryname)

    if not os.path.exists(csvfile):
        with open(csvfile, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=schema)
            writer.writeheader()

    with open(csvfile, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=schema)
        for im in imset:
            if im.attributes is None:
                im.attributes = { 'TEMPLATE_ID': im.category(), 'SUBJECT_ID': im.category() }
            filename = os.path.join(imgdir, '%08d.png' % sighting_id)
            im.attributes['FILE'] = filename
            im.attributes['SIGHTING_ID'] = str(sighting_id)
            sighting_id += 1
            im.saveas(filename)

            if filename.startswith(visetdir):
                filename = filename[len(visetdir):].lstrip('/')
            r = {}
            for (k, v) in im.attributes.iteritems():
                if k in schema:
                    r[k] = v
            r['FILE'] = filename # Write out path relative to csv file
            writer.writerow(r)

