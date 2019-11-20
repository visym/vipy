import os
import csv
from bobo.cache import Cache
from bobo.video import VideoCategoryStream
from bobo.util import remkdir, isexe, isvideo, isimg, imsavelist, imlist
import shutil

def videolist(viddir=None):
    return [os.path.join(viddir, filename) for filename in os.listdir(viddir) if (isvideo(filename) and (filename[0] != '.'))]

def frames(viddir):
    for v in videolist(viddir):
        (outdir, ext) = os.path.splitext(v)
        cmd = 'ffmpeg  -r 25 -i \'%s\' -qscale:v 2 -vf "scale=-1:240" %s/%%08d.jpg &> /dev/null' % (v, remkdir(outdir))        
        print '[bobo.viset.videodir]: exporting frames from "%s" to "%s"' % (v, outdir)
        os.system(cmd)

def clean(viddir):
    for v in videolist(viddir):
        (outdir, ext) = os.path.splitext(v)
        if os.path.isdir(outdir):
            print '[bobo.viset.videodir]: removing frame directory "%s"' % (outdir)            
            shutil.rmtree(outdir)

def framelist(viddir):
    fulllist = []
    for v in videolist(viddir):
        (framedir, ext) = os.path.splitext(v)
        if os.path.isdir(framedir):
            print '[bobo.viset.videodir]: generating image list for frame directory "%s"' % (framedir)            
            fulllist.append(imlist(framedir))
    return fulllist

def framedirlist(viddir):
    return [os.path.join(viddir, filename) for filename in os.listdir(viddir) if (os.path.isdir(os.path.join(viddir, filename)) and (filename[0] != '.'))]

        
                    
