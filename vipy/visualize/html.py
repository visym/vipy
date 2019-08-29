import os
import numpy as np
import strpy.bobo.util
from strpy.bobo.util import remkdir, imlist, filetail
import time
import shutil

def imagelist(list_of_image_files, outdir, title='Image Visualization', imagewidth=64):
    """Given a list of image filenames wth absolute paths, copy to outdir, and create an index.html file that visualizes each"""
    k_divid = 0;
    
    # Create summary page to show precomputed images
    outdir = remkdir(outdir)
    filename = os.path.join(remkdir(outdir), 'index.html');
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">Title: %s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of Images: %d<br>\n' % len(list_of_image_files))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid);  k_divid = k_divid + 1;
    
    # Generate images and html
    for (k, imsrc) in enumerate(list_of_image_files):
        shutil.copyfile(imsrc, os.path.join(outdir, filetail(imsrc)))
        imdst = filetail(imsrc)
        f.write('<p>\n</p>\n')        
        f.write('<b>Filename: %s</b><br>\n' % imdst) 
        f.write('<br>\n')                
        f.write('<img src="%s" alt="image" width=%d/>\n' % (imdst, imagewidth))
        f.write('<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')        
                
    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')    
    f.close()
    return filename


def imagetuplelist(list_of_tuples_of_image_files, outdir, title='Image Visualization', imagewidth=64):
    """Imageset but put tuples on same row"""
    k_divid = 0;
    
    # Create summary page to show precomputed images
    outdir = remkdir(outdir)
    filename = os.path.join(remkdir(outdir), 'index.html');
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">Title: %s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of Tuples: %d<br>\n' % len(list_of_tuples_of_image_files))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid);  k_divid = k_divid + 1;
    
    # Generate images and html
    for (k, imsrclist) in enumerate(list_of_tuples_of_image_files):
        f.write('<p>\n</p>\n') 
        for imsrc in imsrclist:       
            shutil.copyfile(imsrc, os.path.join(outdir, filetail(imsrc)))
            imdst = filetail(imsrc)
            f.write('<b>Filename: %s</b><br>\n' % imdst) 
        f.write('<p>\n</p>\n')     
        f.write('<br>\n')                   
        for imsrc in imsrclist:
            imdst = filetail(imsrc)
            f.write('<img src="%s" alt="image" width=%d/>' % (imdst, imagewidth))
        f.write('\n<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')        
                
    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')    
    f.close()
    return filename
