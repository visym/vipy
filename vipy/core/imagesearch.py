import urllib
import urllib2
import sys
import os
import datetime
import json
from strpy.bobo.util import remkdir, imlist, dirlist
import numpy as np
import strpy.bobo.annotation


def count(outdir):
    num_files = 0
    for d in dirlist(outdir):
        num_files += len(imlist(d))
    return num_files

    
def facebookprofilerange(fbid, numid, outdir='./imgs', cleanup=True, hierarchical=False, redownload=False):
    for x in range(fbid, fbid+numid):
        facebookprofile(x, outdir, cleanup, hierarchical, redownload)

def facebookprofile(fbid, outdir='./imgs', cleanup=True, hierarchical=False, redownload=False):
    if hierarchical:
        subdir = remkdir(os.path.join(outdir, str(int(float(fbid)/1E4)))) # 10000 images per directory
        outfile = os.path.join(subdir, '%d.jpg' % int(fbid))  # outdir/1000/10000001.jpg
    else:
        outfile = os.path.join(outdir, '%d.jpg' % int(fbid))
        
    url = "http://graph.facebook.com/picture?id=" + str(fbid) + "&width=800"
    if not os.path.exists(outfile) or redownload:
        try:
            print '[facebookprofile.download]: Downloading "%s" to "%s"' % (url, outfile) 
            
            user_agent = np.random.choice(bobo.annotation.common_user_agents)
            headers = {'User-Agent':user_agent}                     
            req = urllib2.Request(url, None, headers)
            imgfile = urllib2.urlopen(req)
            total_size = int(imgfile.info().getheader('Content-Length').strip())
            downloaded = 0
            CHUNK = 256 * 10240
            with open(outfile, 'wb') as fp:
                while True:
                    chunk = imgfile.read(CHUNK)
                    downloaded += len(chunk)
                    #print math.floor( (downloaded / total_size) * 100 )
                    if not chunk: break
                    fp.write(chunk)
                        
            #urllib.urlretrieve(url, outfile)
            
            s = os.path.getsize(outfile)
            if cleanup and (s < 11000 or s == 10626 or s == 10491):
                print '[facebookprofile.download]: deleting invalid file "%s"' % outfile
                os.remove(outfile)

        except urllib2.HTTPError, e:
            print '[fb_image.download]: Skipping "%s"' % (url)            
            print "HTTP Error:",e.code , url
            #return False
        except urllib2.URLError, e:
            print '[fb_image.download]: Skipping "%s"' % (url)            
            print "URL Error:",e.reason , url
            #return False
        except KeyboardInterrupt:
            raise
        except:
            print "UNKNOWN ERROR"
            #raise
