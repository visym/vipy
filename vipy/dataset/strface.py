import numpy as np
import bobo.app
from bobo.util import load, save, csvlist, dirlist, imlist, readcsv, filebase, imwrite, remkdir, temppng, writecsv, filepath, fileext, filetail
from bobo.image import ImageDetection, Image
from bobo.video import VideoDetection
import os
import copy


class STRFace(object):
    def __init__(self, datadir='/proj/janus3/strface'):
        self.datadir = datadir 
        self.subdatadir = [os.path.join(self.datadir, d) for d in ['detections_02OCT15', 'detections_25SEP15', 'detections_30SEP15', 'detections_12OCT15', 'detections_16OCT15', 'detections_05NOV15', 'detections_10NOV15', 'detections_13NOV15']]
        self.csvlabelfiles = [os.path.join(self.datadir, 'turklabel', f) for f in ['Batch_2121390_batch_results.csv', 'Batch_2183593_batch_results.csv']]
        self._parsed = None
        
    def __repr__(self):
        return str('<viset.strface: %s>' % self.datadir)
    
    def _parse(self):
        """Return dictionary of subjectid to list of csvfiles for mated subjects"""
        if self._parsed is not None:
            return self._parsed
        
        csv1 = readcsv(self.csvlabelfiles[0])  # turk labeling
        csv2 = readcsv(self.csvlabelfiles[1])
        csv = csv1 + csv2

        mate = [y[2].replace('"', '') for y in [x for x in csv if 'Same' in x[5]]]
        nonmate = [y[2].replace('"', '') for y in [x for x in csv if 'Different' in x[5]]]
        uniquemate = set([y[2].replace('"', '') for y in [x for x in csv if 'Same' in x[5]]])  # 
        unique = set([y[2].replace('"', '') for y in csv])

        subjects = {}    
        for (j,indir) in enumerate(self.subdatadir):
            #print '[strface.parse][%d/%d]: importing %s' % (j+1, len(self.subdatadir), indir)
        
            files = csvlist(indir)
            for (k,f) in enumerate(files):
                subjectid = filebase(f).split('_')[1]  #            
                if subjectid not in uniquemate:
                    continue  # JUST LABELED MATED SUBJECTS
                if subjectid not in subjects.keys():
                    subjects[subjectid] = [f]
                else:
                    subjects[subjectid].append(f)

        self._parsed = subjects
        return subjects

    def takesubject(self, n, subjectid=None):
        """Randomly select n frames from dataset for a single subject"""
        takelist = []
        subjectdict = self._parse()
        SCHEMA = self.schema()
        subjectid = np.random.choice(self.subjects(), 1)[0] if subjectid is None else subjectid
        for k in range(0,n):
            csvfile = np.random.choice(subjectdict[subjectid], 1)[0]
            csv = readcsv(csvfile)
            r = csv[np.random.randint(1,len(csv))] # not including header
            im = ImageDetection(filename=os.path.join(filepath(csvfile), r[0]), category=subjectid, attributes={k:v for (k,v) in zip(SCHEMA,r)}).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6]))
            takelist.append(im)
        return takelist

        
    def take(self, n):
        """Randomly select n frames from dataset"""
        takelist = []
        subjectdict = self._parse()
        SCHEMA = self.schema()        
        for subjectid in np.random.choice(self.subjects(), n):
            csvfile = np.random.choice(subjectdict[subjectid], 1)[0]
            csv = readcsv(csvfile)
            r = csv[np.random.randint(1,len(csv))] # not including header
            im = ImageDetection(filename=os.path.join(filepath(csvfile), r[0]), category=subjectid, attributes={k:v for (k,v) in zip(SCHEMA,r)}).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6]))
            takelist.append(im)
        return takelist
    
    def subjects(self):
        return self._parse().keys()


    def schema(self):
        """Schema for CSV files"""
        return ['FILENAME', 'MEDIA_ID', 'TRACK_ID' ,'FACE_X', 'FACE_Y', 'FACE_WIDTH', 'FACE_HEIGHT',
                'LeftBrowLeftCorner_X', 'LeftBrowLeftCorner_Y', 'LeftBrowCenter_X', 'LeftBrowCenter_Y', 'LeftBrowRightCorner_X', 'LeftBrowRightCorner_Y', 'RightBrowLeftCorner_X',
                'RightBrowLeftCorner_Y', 'RightBrowCenter_X', 'RightBrowCenter_Y','RightBrowRightCorner_X', 'RightBrowRightCorner_Y', 'LeftEyeLeftCorner_X', 'LeftEyeLeftCorner_Y',
                'LeftEyeCenter_X', 'LeftEyeCenter_Y', 'LeftEyeRightCorner_X', 'LeftEyeRightCorner_Y', 'RightEyeLeftCorner_X', 'RightEyeLeftCorner_Y', 'RightEyeCenter_X', 'RightEyeCenter_Y',
                'RightEyeRightCorner_X', 'RightEyeRightCorner_Y', 'LeftEar_X', 'LeftEar_Y', 'NoseLeft_X', 'NoseLeft_Y', 'NoseCenter_X', 'NoseCenter_Y', 'NoseRight_X', 'NoseRight_Y', 'RightEar_X',
                'RightEar_Y', 'MouthLeftCorner_X', 'MouthLeftCorner_Y', 'MouthCenter_X', 'MouthCenter_Y', 'MouthRightCorner_X', 'MouthRightCorner_Y', 'ChinCenter_X', 'ChinCenter_Y']

    
    def dataset(self):
        """Return a generator to iterate over dataset"""
        V = []
        subjects = self._parse()
        SCHEMA = self.schema()
        for (k, (subjectid, csvfiles)) in enumerate(subjects.iteritems()):
            #print '[strface.dataset][%d/%d]: importing "%s"' % (k+1, len(subjects), subjectid)
            for (j,f) in enumerate(csvfiles):
                csv = readcsv(f)                  
                if len(csv) > 1:                    
                    frames = [ImageDetection(filename=os.path.join(filepath(f), r[0]), category=subjectid, attributes={k:v for (k,v) in zip(SCHEMA,r)}).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6])) for r in csv[1:]]
                    yield VideoDetection(frames=frames)

        
    def rdd(self, sparkContext):
        return (sparkContext.parallelize(self._parse().items())
                .flatMap(lambda (k,v): [(k,f,readcsv(f)[1:]) for f in v])
                .map(lambda (k,f,v): VideoDetection(frames=[ImageDetection(filename=os.path.join(filepath(f), r[0]), category=k).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6])).setattribute('MEDIA_ID', r[1]).setattribute('TRACK_ID', r[2]) for r in v] if len(v)>0 else [])))
    
    def broadset(self):
        """Four images and two small videos [2,30] for 2500 subjects to match distribution of CS2 media, total of 15000 encodings"""
        V = []
        subjects = {k:v for (k,v) in self._parse().iteritems() if len(v)>=6}  # 3263 -> 2878, minimum 6 videos per subject
        subjects = {k:v for (j,(k,v)) in enumerate(subjects.iteritems()) if j<2500}  # 2971 --> 2500        
        for (k, (subjectid, csvfiles)) in enumerate(subjects.iteritems()):
            print '[strface.dataset][%d/%d]: importing "%s"' % (k+1, len(subjects), subjectid)
            np.random.permutation(csvfiles)  # randomly permute videos 
            for (j,f) in enumerate(csvfiles):
                csv = readcsv(f)                  
                if len(csv) > 1:                    
                    frames = [ImageDetection(filename=os.path.join(filepath(f), r[0]), category=subjectid).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6])).setattribute('MEDIA_ID', r[1]).setattribute('TRACK_ID', r[2]) for r in csv[1:]]
                    if j < 4:
                        V.append(VideoDetection(frames=list(np.random.choice(frames,1)))) # three single images, randomly selected from video
                    elif j < 6:
                        V.append(VideoDetection(frames=list(np.random.choice(frames, np.random.randint(min(2,len(frames)), min(30,len(frames))), replace=False))))  # single video with [2,30] frames randomly selected
                    else:
                        break
        return V

    def deepset(self):
        """Ten images and five videos [2,30] for 1000 subjects to match distribution of CS2 media, total of 15000 encodings"""
        V = []
        subjects = {k:v for (k,v) in self._parse().iteritems() if len(v)>=20}  # 3263 -> 1523, minimum 20 videos per subject
        subjects = {k:v for (j,(k,v)) in enumerate(subjects.iteritems()) if j<1000}  # 1523 --> 1000        
        for (k, (subjectid, csvfiles)) in enumerate(subjects.iteritems()):
            print '[strface.dataset][%d/%d]: importing "%s"' % (k+1, len(subjects), subjectid)
            np.random.permutation(csvfiles)  # randomly permute videos 
            for (j,f) in enumerate(csvfiles):
                csv = readcsv(f)                  
                if len(csv) > 1:                    
                    frames = [ImageDetection(filename=os.path.join(filepath(f), r[0]), category=subjectid).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6])).setattribute('MEDIA_ID', r[1]).setattribute('TRACK_ID', r[2]) for r in csv[1:]]
                    if j < 10:
                        V.append(VideoDetection(frames=list(np.random.choice(frames,1, replace=False)))) # 10 single images, each randomly selected from a video
                    elif j < 15:
                        V.append(VideoDetection(frames=list(np.random.choice(frames,  np.random.randint(min(2,len(frames)), min(30,len(frames))), replace=False))))  # 5 single video, each with [2,30] frames randomly selected from video
                    else:
                        break
        return V
            

    def deeperset(self):
        """All videos, [2,30] frames, All subjects"""
        V = []
        subjects = self._parse()
        for (k, (subjectid, csvfiles)) in enumerate(subjects.iteritems()):
            print '[strface.dataset][%d/%d]: importing "%s"' % (k+1, len(subjects), subjectid)
            for (j,f) in enumerate(csvfiles):
                csv = readcsv(f)                  
                if len(csv) > 1:                    
                    frames = [ImageDetection(filename=os.path.join(filepath(f), r[0]), category=subjectid).boundingbox(xmin=float(r[3]), ymin=float(r[4]), width=float(r[5]), height=float(r[6])).setattribute('MEDIA_ID', r[1]).setattribute('TRACK_ID', r[2]) for r in csv[1:]]
                    V.append(VideoDetection(frames=list(np.random.choice(frames,  np.random.randint(min(2,len(frames)), min(30,len(frames))), replace=False))))  # 5 single video, each with [2,30] frames randomly selected from video
        return V
    
    
def exporturls(outfile, strface=STRFace()):
    """Annotated URLs file"""
    S = {k:str(j) for (j,k) in enumerate(strface.subjects())}  # subject id -> unique index
    SCHEMA = ['URL', 'SUBJECT_ID', 'VIDEO_ID', 'FRAME_ID', 'MILLISECOND'] + strface.schema()[3:]
    with open(outfile, 'w') as f:
        f.write(','.join(SCHEMA) + '\n')        
        for (i,v) in enumerate(strface.dataset()):  # video iterator
            try:
                urlpath = '/data/vision/janus/data/videosearch/' + 'downloads%s' % (filepath(v[0].filename()).split('detections')[1]) + '.url'  # FIXME: RSYNC
                url = readcsv(urlpath)[0][0]  # video URL
            except:
                urlpath = strface.datadir + '/downloads%s' % (filepath(v[0].filename()).split('detections')[1]) + '.url'  
                url = readcsv(urlpath)[0][0]  # video URL

            if ',' in url:
                raise ValueError('url contains comma!  Escape me')
            
            print '[strface.export]: exporting "%s" from "%s"' % (v.category(), url)
            for (j,im) in enumerate(v):  # frame
                d = im.attributes
                d.update({'URL':url, 'SUBJECT_ID':S[v.category()], 'MILLISECOND':filebase(im.filename()), 'VIDEO_ID':i, 'FRAME_ID':j})
                attr = ','.join([str(d[k]) for k in SCHEMA])
                f.write(attr + '\n')                

    return outfile
        

def exportcrops(strface, sparkContext, outdir):
    """Crops using strface object and sparkcontext, write to outdir"""
    """sparkContext = pyspark.SparkContext(appName='strface', environment=os.environ) """
    remkdir(outdir)
    bobo.app.setverbosity(1)
    subject_to_index = {s:k for (k,s) in enumerate(strface.subjects())}
    rdd = strface.rdd(sparkContext)            
    print '[strface.exportcrops]: Writing crops for %d subjects' % len(subject_to_index)

    def f_saveas(im, savedir):
        try:
            bw = im.boundingbox().width()
            bh = im.boundingbox().height()
            cx = im.boundingbox().x_centroid()
            cy = im.boundingbox().y_centroid()
            im = im.boundingbox(xmin=cx-0.75*bw, ymin=cy-1.1*bh, xmax=cx+0.75*bw, ymax=cy+0.75*bh)  # from Omkar
            im = im.crop()  # crop here to image rectangle
            if im.boundingbox().width() < im.boundingbox().height():
                im = im.resize(cols=256)  # smallest dimension is 256
            else:
                im = im.resize(rows=256)
            im.saveas(os.path.join(savedir, '%s' % filetail(im.filename())))
        except:
            print 'skipping'
        
    (rdd.flatMap(lambda v: v.frames())
        .map(lambda im: (remkdir(os.path.join(outdir, '%04d' % subject_to_index[im.category()], '%02d' % int(im.attributes['MEDIA_ID'][-4:]))), im))
        .foreach(lambda (savedir, im):  f_saveas(im, savedir))) 
                


class STRFaceCrops(object):
    def __init__(self, datadir='/proj/janus3/strfacecrops'):
        self.datadir = datadir 
        self._subjects = None
        
    def __repr__(self):
        return str('<viset.strfacecrops: "%s">' % self.datadir)

    def subjects(self):
        if self._subjects is None:
            self._subjects = [filebase(d) for d in dirlist(os.path.join(self.datadir, 'images'))]
        return self._subjects # cached

    def dataset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir, 'images')):
            for sd in dirlist(d):  # video index
                for f in imlist(sd):
                    im = ImageDetection(filename=f, category=filebase(d))
                    im = im.boundingbox(xmin=float(im.width()-256)/2.0, ymin=float(im.height()-256.0)/2.0, xmax=256.0+((im.width()-256.0)/2.0),ymax=256.0+((im.height()-256.0)/2.0))
                    im = im.boundingbox(dilate=0.875)  # central 224x224
                    yield im

    def fastset(self):
        """Return a generator to iterate over dataset"""
        for d in dirlist(os.path.join(self.datadir, 'images')):  # subject
            for sd in dirlist(d):  # video index
                for f in imlist(sd):  # frames
                    im = ImageDetection(filename=f, category=filebase(d))
                    #im = im.boundingbox(xmin=float(im.width()-256)/2.0, ymin=float(im.height()-256.0)/2.0, xmax=256.0+((im.width()-256.0)/2.0),ymax=256.0+((im.height()-256.0)/2.0))
                    #im = im.boundingbox(dilate=0.875)  # central 224x224
                    yield im

    def videos(self, s):
        return [os.path.join(self.datadir, s, d) for d in dirlist(os.path.join(self.datadir, s))]


    def take(self, n):
        """Take one image from each video of n subjects"""
        S = np.random.choice(self.subjects(), n)
        takelist = []
        for d in dirlist(os.path.join(self.datadir, 'images')):
            for sd in dirlist(d):  # video index
                if filebase(d) in S:
                    f = np.random.choice(imlist(sd),1)[0]
                    im = ImageDetection(filename=f, category=filebase(d))                
                    #im = im.boundingbox(xmin=float(im.width()-256)/2.0, ymin=float(im.height()-256.0)/2.0, xmax=256.0+((im.width()-256.0)/2.0),ymax=256.0+((im.height()-256.0)/2.0))
                    #im = im.boundingbox(dilate=0.875)  # central 224x224
                    takelist.append(im)
        return takelist

    def takesubject(self, n, subjectid=None):
        """Randomly select n frames from dataset for a single subject"""
        S = np.random.choice(self.subjects(), 1)
        takelist = []
        for d in dirlist(os.path.join(self.datadir, 'images')):
            if filebase(d) in S:
                for sd in np.random.choice(dirlist(d), n):  # video index
                    f = np.random.choice(imlist(sd),1)[0]
                    im = ImageDetection(filename=f, category=filebase(d))                
                    #im = im.boundingbox(xmin=float(im.width()-256)/2.0, ymin=float(im.height()-256.0)/2.0, xmax=256.0+((im.width()-256.0)/2.0),ymax=256.0+((im.height()-256.0)/2.0))
                    #im = im.boundingbox(dilate=0.875)  # central 224x224
                    takelist.append(im)
        return takelist

    def takevideo(self, n, subjectid=None):
        """Randomly select n frames from each video for a single subject"""
        S = np.random.choice(self.subjects(), 1) if subjectid is None else [subjectid]
        takelist = []
        for d in dirlist(os.path.join(self.datadir, 'images')):
            if filebase(d) in S:
                for sd in dirlist(d):  # video index
                    imset = imlist(sd)
                    F = np.random.choice(imset,n)
                    im = [ImageDetection(filename=f, category=filebase(d)) for f in F]
                    #im = im.boundingbox(xmin=float(im.width()-256)/2.0, ymin=float(im.height()-256.0)/2.0, xmax=256.0+((im.width()-256.0)/2.0),ymax=256.0+((im.height()-256.0)/2.0))
                    #im = im.boundingbox(dilate=0.875)  # central 224x224
                    takelist = takelist + im
        return takelist

    
