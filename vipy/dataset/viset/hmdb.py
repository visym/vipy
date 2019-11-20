raise ValueError('FIXME')

import os
import csv
from bobo.cache import Cache
from bobo.video import VideoCategoryStream
from bobo.util import remkdir, isexe

URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
SHA1 = None
VISET = 'hmdb'


def split(csvfile=None):
    pass


def unpack(rarfile, outdir):
    os.system('unrar e %s %s' % (rarfile, outdir))
    for (idx_category, rarfile) in enumerate(os.listdir(outdir)):
        (category, ext) = os.path.splitext(rarfile)    
        if not os.path.isdir(os.path.join(outdir,category)):
            os.mkdir(os.path.join(outdir, category))
            os.mkdir(os.path.join(outdir, category, 'export'))        
            cmd = 'unrar e %s %s' % (os.path.join(outdir,rarfile), os.path.join(outdir,category))
            os.system(cmd)
            #for (idx_video, avifile) in enumerate(os.listdir(os.path.join(outdir, category))):
            #    os.system('mv \'%s\' %s' % (avifile, '%s_%04d.avi' % (category, idx_video)))

            
def export_videolist(outdir=None):
    cache = Cache(cacheroot=outdir, subdir=VISET)    
    vidlist = []
    for (idx_category, category) in enumerate(os.listdir(cache.root())):
        if os.path.isdir(os.path.join(cache.root(), category)):
            for (idx_video, filename) in enumerate(os.listdir(os.path.join(cache.root(), category))):    
                [avibase,ext] = os.path.splitext(filename)
                if ext == '.avi':
                    vidlist.append( (os.path.join(category, filename), category) )
    return vidlist


def export(outdir=None, clean=False):
    # Unpack dataset
    cache = Cache(cacheroot=outdir, subdir=VISET)        
    if clean:
        cache.clean()
                        
    rarfile = unpack(cache.get(URL, SHA1), cache.root())

    vidlist = []
    for (idx_category, category) in enumerate(os.listdir(cache.root())):
        if os.path.isdir(os.path.join(cache.root(), category)):
            for (idx_video, filename) in enumerate(os.listdir(os.path.join(cache.root(), category))):    
                [avibase,ext] = os.path.splitext(filename)
                if ext == '.avi':
                    vidlist.append( (os.path.join(category, filename), category) )
    return vidlist
    
    
    # Check for frame export utility
    #if not isexe('ffmpeg'):
    #    raise IOError('[bobo.viset.hmdb]: ffmpeg not found on path')
                    

def stream(outdir=None):
    cache = Cache(cacheroot=outdir, subdir=VISET)            
    csvfile = os.path.join(cache.root(), '%s.csv' % VISET)            
    if not os.path.isfile(csvfile):
        csvfile = export()        
    return VideoCategoryStream(csvfile, cache=cache)
        



def export_frames(indir):
    vidlist = videolist_small(indir)     
    imdir = os.path.join(indir, 'export')   
    if not os.path.isdir(imdir):
        os.mkdir(imdir)
    
    for (k_video, (avifile, category)) in enumerate(vidlist):
        [filebase,ext] = os.path.splitext(avifile)
        outdir = os.path.join(imdir, 'video_%04d' % k_video)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        cmd = "ffmpeg -i \'%s\' -r 25 %s/img_%%06d.png" % (os.path.join(indir, avifile), outdir)
        print cmd
        os.system(cmd)
    
def export_category(outdir, mycategory):
    vidlist = export_videolist(outdir)    
    for (idx_video, (vidfile, category)) in enumerate(vidlist):
        if category == mycategory:
            (viddir, ext) = os.path.splitext(vidfile)            
            imdir = os.path.join(outdir, category, 'export', 'video_%03d' % idx_video)
            if not os.path.isdir(imdir):
                os.mkdir(imdir)
            cmd = "ffmpeg -i \'%s\' -r 30 %s/img_%%08d.jpg" % (os.path.join(outdir,vidfile), imdir)
            print cmd
            os.system(cmd)
        
