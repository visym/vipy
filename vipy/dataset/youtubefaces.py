import os
from glob import glob
from bobo.util import remkdir, isstring, quietprint, islist, tolist, loadmat
from bobo.image import ImageDetection, ImageCategory
from bobo.video import VideoDetection
import bobo.app


    
class YouTubeFaces(object):
    def __init__(self, datadir=None, parsed=None):
        self.datadir = bobo.app.datadir() if datadir is None else datadir
        self.ytfdir = os.path.join(self.datadir, 'YouTubeFaces')
        
        if not os.path.isdir(os.path.join(self.ytfdir, 'frame_images_DB')):
            raise ValueError('Download YouTubeFaces dataset manually and unpack to to "%s", pass in datadir, or set $JANUS_DATA ' % self.ytfdir)        

        self._parsed = parsed  # user saved output from self.parse for optional caching

        
    def __repr__(self):
        return str('<viset.youtubefaces: %s>' % self.ytfdir)
    

    def parse(self, max_num_subjects=None):
        """Parse youtubefaces into a list of ImageDetections"""
        
        if self._parsed is not None:
            return self._parsed
                
        # Write images and annotations
        # The data in this file is in the following format:
        # filename,[ignore],x,y,width,height,[ignore],[ignore]
        # where:
        # x,y are the center of the face and the width and height are of the rectangle that the face is in.
        # For example:
        # $ head -3 Richard_Gere.labeled_faces.txt
        # Richard_Gere\3\3.618.jpg,0,262,165,132,132,0.0,1
        # Richard_Gere\3\3.619.jpg,0,260,164,131,131,0.0,1
        # Richard_Gere\3\3.620.jpg,0,261,165,129,129,0.0,1

        imlist = []
        categorydir = os.path.join(self.ytfdir, 'frame_images_DB')
        for i,infilename in enumerate(glob('%s/*labeled_faces.txt'%categorydir)):
            if max_num_subjects and max_num_subjects == i: break
            print '[viset.youtubefaces:] parsing "%s" ' % infilename
            with open(infilename, 'r') as infile:
                for line in infile:
                    ( imname, ignore, x_ctr, y_ctr, w, h, ignore, ignore) = line.split(',')
                    imname = imname.replace('\\', '/')
                    category = imname.split('/')[0]
                    xmin = int(x_ctr) - int(float(w) / 2.0)
                    ymin = int(y_ctr) - int(float(h) / 2.0)
                    xmax = int(xmin) + int(w)
                    ymax = int(ymin) + int(h)
                    p = os.path.join(categorydir, imname) 
                
                    (category,videoid,filename) = imname.split('/')
                    mediaID = '%s_%s' % (category, videoid)
                    imlist.append(ImageDetection(filename=p, category=category, xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax)).setattribute('MEDIA_ID', str(mediaID)));

        # Done!
        self._parsed = imlist
        return imlist
    
    def subjects(self, sparkContext=None, max_num_subjects=None):
        return self.rdd(sparkContext=sparkContext, minPartitions=16, max_num_subjects=max_num_subjects).map(lambda v: v.category()).distinct().collect()
    
    def images(self, max_num_subjects=None):
        return self.parse(max_num_subjects=max_num_subjects)
            
    def videos(self, sparkContext=None, max_num_subjects=None):
        return self.rdd(sparkContext=sparkContext, max_num_subjects=max_num_subjects).collect()
                
    def rdd(self, sparkContext=None, appname='viset_youtubefaces', minPartitions=None, max_num_subjects=None):
        """Create a resilient distributed dataset"""
        sparkContext = bobo.app.init(appname) if sparkContext is None else sparkContext                                            
        return (sparkContext.parallelize(self.parse(max_num_subjects=max_num_subjects), minPartitions)
                            .map(lambda im: (im.attributes['MEDIA_ID'], im))  # keyby media ID
                            .reduceByKey(lambda a,b: tolist(a)+tolist(b))  # group all frames of video together by videoID key into giant concatenated list of frames
                            .map(lambda (k,fr): VideoDetection(frames = fr, attributes=fr[0].attributes, category=fr[0].category())))  # construct video object from grouped frames

    def splits(self):
        mat = loadmat(os.path.join(self.ytfdir, 'meta_data', 'meta_and_splits.mat'))
        return mat['Splits']  # splits are indexed into mat['mat_names'] dictionary, parse me!

        
class YouTubeFacesAligned(YouTubeFaces):
    def parse(self, max_num_subjects=None):
        """Parse youtubefaces into a list of ImageDetections"""
        
        if self._parsed is not None:
            return self._parsed
                
        imlist = []
        categorydir = os.path.join(self.ytfdir, 'frame_images_DB')
        for i,nfilename in enumerate(glob('%s/*labeled_faces.txt'%categorydir)):
            if max_num_subjects and max_num_subjects == i: break
            print '[viset.youtubefaces:] parsing "%s" ' % infilename
            with open(infilename, 'r') as infile:
                for line in infile:
                    ( imname, ignore, x_ctr, y_ctr, w, h, ignore, ignore) = line.split(',')
                    imname = imname.replace('\\', '/')
                    category = imname.split('/')[0]
                    (category,videoid,filename) = imname.split('/')                    
                    p = os.path.join(self.ytfdir, 'aligned_images_DB', category, videoid, 'aligned_detect_%s' % filename)                
                    mediaID = '%s_%s' % (category, videoid)
                    imlist.append(ImageDetection(filename=p, category=category).setattribute('MEDIA_ID', str(mediaID)))  # FIXME: bounding boxes?

        # Done!
        self._parsed = imlist
        return imlist
    
