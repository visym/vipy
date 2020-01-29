import os
import numpy as np
from vipy.util import remkdir, imlist, filetail
import shutil
from vipy.util import istuple, islist, isnumpy, quietprint, remkdir, filebase, filetail
from vipy.image import Image
from vipy.video import VideoDetection
from vipy.show import savefig
from collections import defaultdict
import time
import PIL


def montage(imset, m, n, rows=None, cols=None, aspectratio=1, crop=False, skip=True, grayscale=True, do_plot=False, figure=None, border=0, border_bgr=(128,128,128), do_flush=False, verbose=False):
    """Montage image of images of size (m,n), such that montage has given aspect ratio  or is exactly (rows x cols).  Pass in iterable of imagedetection objects which is used to montage rowwise"""

    n_imgs = len(imset)
    M = int(np.ceil(np.sqrt(n_imgs)))
    N = M
    if aspectratio != 1:
        x = int(round((aspectratio*N-M)/(1+aspectratio)))
        N = N - x
        M = M + x
    elif rows is not None and cols is not None:
        N = rows
        M = cols
    padding = (M+1) * border
    size = (M * m + padding, N * n + padding)
    bc = border_bgr
    if grayscale:
        if islist(bc) or istuple(bc) or isnumpy(bc):
            bc = np.mean(bc)
        I = np.array(PIL.Image.new(mode='L', size=size, color=bc))
    else:
        I = np.array(PIL.Image.new(mode='RGB', size=size, color=bc))
    k = 0
    for j in range(N):
        for i in range(M):
            if k >= n_imgs: 
                break
            sliceM, sliceN = i * (m+border) + border, j * (n+border) + border
            try:
                if crop:
                    if not imset[k].bbox.valid():
                        print('[janus.visualize.montage] invalid bounding box "%s" ' % str(imset[k].bbox))
                        if skip == False:
                            print('[janus.visualize.montage] using original image')
                            if grayscale:
                                im = imset[k].grayscale().resize(n,m)._array                                                
                            else:
                                im = imset[k].resize(n,m)._array
                        else:
                            raise
                    else:
                        if grayscale:
                            im = imset[k].grayscale().crop(imset[k].bbox).resize(n,m)._array
                        else:
                            im = imset[k].crop(imset[k].bbox).resize(n,m)._array
                else:
                    if grayscale:
                        im = imset[k].grayscale().resize(n,m)._array  # m=width, n=height
                    else:
                        im = imset[k].resize(n,m)._array
       
                if im.dtype == np.float32:
                    if im.max() <= 1.0:
                        im *= 255.0
                    im = im.astype(np.uint8)

                I[sliceN:sliceN + n, sliceM:sliceM + m] = im
                
            except KeyboardInterrupt:
                raise
            except:
                print('[janus.visualize.montage] skipping...')
                if skip:
                    pass
                else:
                    raise

            if do_flush:
                imset[k].flush()  # clear memory
            if verbose and ((k % 100) == 0):
                print('[janus.visualize.montage][%d/%d] processing...' % (k, n_imgs))
            
            k += 1

    if k == 0:
        print('[janus.visualize.montage] Warning: No images were processed')

    if do_plot is True:
        im = Image('')
        im._array = I
        # HACK: float(0-255) graycale images display incorrectly
        if grayscale:
            im.preprocess().rgb().show(figure=figure)
        else:
            im.show(figure=figure)

    return I       


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
