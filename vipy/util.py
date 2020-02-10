import urllib.request
import urllib.parse
import urllib.error
from urllib.parse import urlparse
from os import chmod
import os.path
import numpy as np
import tempfile
import time
from time import gmtime, strftime, localtime
import sys
import csv
import hashlib
import shutil
import json
import scipy.io
import shelve
import re
import uuid
import dill
import builtins
import pickle as cPickle
import PIL
import matplotlib.pyplot as plt
from itertools import groupby as itertools_groupby
import importlib


def try_import(package, pipname=None):
    """Show a helpful error message for missing optional packages"""
    try:
        importlib.import_module(package)
    except:
        raise ImportError('Optional package "%s" not installed -  Run "pip install %s" ' % (package, package if pipname is None else pipname))


def count_images_in_subdirectories(indir):
    """Count the total number of images in indir/subdir1, indir/subdir2, ..."""
    num_files = 0
    for d in dirlist(outdir):
        num_files += len(imlist(d))
    return num_files


def quietprint(x, verbosity=None):
    print(x)


def rowvectorize(X):
    """Convert a 1D numpy array to a 2D row vector of size (1,N)"""
    return X.reshape((1, X. size)) if X.ndim == 1 else X


def columnvectorize(X):
    """Convert a 1D numpy array to a 2D column vector of size (N,1)"""
    return X.reshape((X. size, 1)) if X.ndim == 1 else X


def isodd(x):
    return x % 2 == 0


def keymax(d):
    """Return key in dictionary containing maximum value"""
    vmax = max(d.values())
    for (k, v) in d.items():
        if v == vmax:
            return k


def keymin(d):
    """Return key in dictionary containing minimum value"""
    vmin = min(d.values())
    for (k, v) in d.items():
        if v == vmin:
            return k


def isjsonfile(filename):
    return hasextension(filename) and fileext(filename).lower() == '.json'


def writejson(d, outfile):
    with open(outfile, 'w') as f:
        json.dump(d, f)
    return outfile


def readjson(jsonfile):
    with open(jsonfile) as f:
        data = json.loads(f.read())
    return data


def groupby(inset, keyfunc):
    """groupby on unsorted inset"""
    return itertools_groupby(sorted(inset, key=keyfunc), keyfunc)


def vipy_groupby(inset, keyfunc):
    """groupby on unsorted inset"""
    return groupby(inset, keyfunc)


def groupbyasdict(inset, keyfunc):
    """Return dictionary of keys and lists from groupby on unsorted inset"""
    return {k: list(v) for (k, v) in groupby(inset, keyfunc)}


def softmax(x, temperature=1.0):
    """Row-wise softmax"""
    assert x.ndim == 2
    z = np.exp((x - np.max(x, axis=1).reshape(x.shape[0], 1)) / temperature)
    return z / np.sum(z, axis=1).reshape(x.shape[0], 1)


def permutelist(inlist):
    """randomly permute list order"""
    return [inlist[k] for k in
            np.random.permutation(list(range(0, len(inlist))))]


def flatlist(inlist):
    """Convert list of tuples into a list expanded by concatenating tuples"""
    outlist = []
    for r in inlist:
        for x in r:
            outlist.append(x)
    return outlist


def rmdir(indir):
    """Recursively remove directory and all contents (if the directory
    exists)"""
    if os.path.exists(indir) and os.path.isdir(indir):
        shutil.rmtree(indir)


def chunklist(inlist, num_chunks):
    """Convert list into a list of lists of length num_chunks each element
    is a list containing a sequential chunk of the original list"""
    (m, n) = (num_chunks, int(np.ceil(float(len(inlist)) / float(num_chunks))))
    return [inlist[i * n:min(i * n + n, len(inlist))] for i in range(0, m)]


def chunklistbysize(inlist, size_per_chunk):
    """Convert list into a list of lists such that each element is a list
    containing a sequential chunk of the original list of length
    size_per_chunk"""
    assert size_per_chunk >= 1
    num_chunks = np.maximum(int(np.ceil(len(inlist) / float(size_per_chunk))),
                            1)
    return chunklist(inlist, num_chunks)


def chunklistWithOverlap(inlist, size_per_chunk, overlap_per_chunk):
    """Convert list into a list of lists such that each element is a list
    containing a sequential chunk of the original list of length
    size_per_chunk"""
    assert size_per_chunk >= 1 and overlap_per_chunk >= 0 and \
        size_per_chunk > overlap_per_chunk
    return [inlist[i:i + size_per_chunk] for i in range(
        0, len(inlist), size_per_chunk - overlap_per_chunk)]


def imwritejet(img, imfile=None):
    """Write a grayscale numpy image as a jet colormapped image to the
    given file"""
    if imfile is None:
        imfile = temppng()

    if isnumpy(img):
        if img.ndim == 2:
            cm = plt.get_cmap('gist_rainbow')
            PIL.Image.fromarray(np.uint8(255 * cm(img)[:,:,:3])).save(imfile)
        else:
            raise ValueError('Input must be a 2D numpy array')
    else:
        raise ValueError('Input must be numpy array')
    return imfile


def isuint8(img):
    return isnumpy(img) and img.dtype == np.dtype('uint8')


def isnumber(x):
    """Is the input a python type of a number or a string containing a number?"""
    return isinstance(x, (int, float)) or (isnumpy(x) and np.isscalar(x)) or (isstring(x) and isfloat(x))


def isfloat(x):
    """Is the input a float or a string that can be converted to float?"""
    try:
        float(x)
        return True
    except ValueError:
        return False


def imwritegray(img, imfile=None):
    """Write a floating point grayscale numpy image in [0,1] as [0,255] grayscale"""
    if imfile is None:
        imfile = temppng()
    if isnumpy(img):
        if img.dtype == np.dtype('uint8'):
            # Assume that uint8 is in the range [0,255]
            PIL.Image.fromarray(img).save(imfile)
        elif img.dtype == np.dtype('float32'):
            # Convert [0, 1.0] to uint8 [0,255]
            PIL.Image.fromarray(np.uint8(img * 255.0)).save(imfile)
        else:
            raise ValueError('Unsupported datatype - '
                             'Numpy array must be uint8 or float32')
    else:
        raise ValueError('Input must be numpy array')
    return imfile


def imwrite(img, imfile=None, writeas=None):
    """Write a floating point 2D numpy image as jet or gray, 3D numpy as
    rgb or bgr"""
    if imfile is None:
        imfile = temppng()
    if not isnumpy(img):
        raise ValueError('image must by numpy object')
    if writeas is None:
        if img.ndim == 2:
            writeas = 'gray'
        else:
            writeas = 'bgr'

    if writeas in ['jet']:
        imwritejet(img, imfile)
    elif writeas in ['gray']:
        imwritegray(img, imfile)
    elif writeas in ['rgb']:
        if img.ndim != 3:
            raise ValueError('numpy array must be 3D')
        if img.dtype == np.dtype('uint8'):
            PIL.Image.fromarray(rgb2bgr(img)).save(imfile)  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            # convert to uint8 then BGR
            PIL.Image.fromarray(rgb2bgr(np.uint8(255.0 * img))).save(imfile)
    elif writeas in ['bgr']:
        if img.ndim != 3:
            raise ValueError('numpy array must be 3D')
        if img.dtype == np.dtype('uint8'):
            PIL.Image.fromarray(img).save(imfile)  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            # convert to uint8 then BGR
            PIL.Image.fromarray(np.uint8(255.0 * img)).save(imfile)
    else:
        raise ValueError('unsupported writeas')

    return imfile


def savetemp(img):
    f = '/tmp/%s.png' % uuid.uuid1().hex
    PIL.Image.fromarray(img.astype(np.uint8)).save(f)
    return f


def gray2jet(img):
    """[0,1] grayscale to [0.255] RGB"""
    import matplotlib.pyplot as plt
    jet = plt.get_cmap('jet')
    return np.uint8(255.0 * jet(img)[:, :, 0:3])


def jet(n, bgr=False):
    """jet colormap"""
    from matplotlib import cm
    cmap = cm.get_cmap('jet', n)
    rgb = np.uint8(255 * cmap(np.arange(n)))
    return rgb if bgr is False else np.fliplr(rgb)


def is_hiddenfile(filename):
    """Does the filename start with a period?"""
    return filename[0] == '.'


def seq(start, stop, step=1):
    """Equivalent to matlab [start:step:stop]"""
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return([start + step * i for i in range(n + 1)])
    else:
        return([])


def loadh5(filename):
    """Load an HDF5 file"""
    if ishdf5(filename):
        try_import('h5py'); import h5py
        f = h5py.File(filename, 'r')
        obj = f[filebase(filename)].value  # FIXME: lazy evaluation?
        return obj
    else:
        raise ValueError('Invalid HDF5 file "%s" ' % filename)


def savemat(outfile, vardict):
    """Write a dictionary to .mat file"""
    scipy.io.savemat(outfile, vardict)
    return outfile


def loadmat(infile):
    """Read a dictionary to .mat file"""
    return scipy.io.loadmat(infile)


def loadmat73(matfile, keys=None):
    """Matlab 7.3 format, keys should be a list of keys to access HDF5
    file as f[key1][key2]...  Returned as numpy array"""
    try_import('h5py'); import h5py
    f = h5py.File(matfile, 'r')
    if keys is None:
        return f
    else:
        for k in keys:
            f = f[k]
        return np.array(f)


def saveas(vars, outfile=None, type='dill'):
    """Save variables as a dill pickled file"""
    outfile = temppickle() if outfile is None else outfile
    if type in ['dill']:
        dill.dump(vars, open(outfile, 'wb'))
        return outfile
    else:
        raise ValueError('unknown serialization type "%s"' % type)

    return outfile


def loadas(infile, type='dill'):
    """Load variables from a dill pickled file"""
    if type in ['dill']:
        try:
            return dill.load(open(infile, 'rb'))
        except Exception:
            import pickle
            with open(infile, 'rb') as pfp:
                fv = pickle.load(pfp, encoding="bytes")
            return fv
    else:
        raise ValueError('unknown serialization type "%s"' % type)


def load(infile):
    """Load variables from a dill pickled file"""
    return loadas(infile, type='dill')


def save(vars, outfile=None, mode=None):
    """Save variables as a dill pickled file"""
    outfile = temppickle() if outfile is None else outfile
    outfile = saveas(vars, outfile, type='dill')
    if mode is not None:
        chmod(outfile, mode)
    return outfile


def fastsave(v, outfile=None):
    """Save variables as a cPickle file with the highest protocol - useful
    for very large custom classes"""
    with open(outfile, 'wb') as fp:
        cPickle.dump(v, fp, cPickle.HIGHEST_PROTOCOL)
    return outfile


def fastload(infile):
    """Load file saved with fastsave"""
    with open(infile, 'rb') as fp:
        v = cPickle.load(fp)
    return v


def loadyaml(yamlfile):
    """Load a numpy array from YAML file exported from OpenCV"""
    return np.squeeze(np.array(cv.Load(yamlfile)))


def matrix2yaml(yamlfile, mtxlist, mtxname=None):
    """Write list of matrices to OpenCV yaml file format with given
    variable names"""
    def _write_matrix(f, M, mtxname):
        f.write('    %s: !!opencv-matrix\n' % mtxname)
        f.write('       rows: %d\n' % M.shape[0])
        f.write('       cols: %d\n' % (M.shape[1] if M.ndim == 2 else 1))
        f.write('       dt: f\n')
        f.write('       data: [ ')
        datastr = ''
        for (k, x) in enumerate(M.flatten()):
            datastr += '%.6e' % x
            if (k + 1 == M.size):
                f.write(datastr)
                break
            datastr += ', '
            if ((k + 1) % 4) == 0:
                f.write(datastr + '\n           ')
                datastr = ''
        f.write(']\n')

    # Write me!
    mtxlist = tolist(mtxlist)
    if mtxname is None:
        mtxname = ['mtx_%02d' % k for k in range(0, len(mtxlist))]
    with open(yamlfile, 'w') as f:
        f.write('%YAML:1.0\n')
        for (m, mname) in zip(mtxlist, mtxname):
            _write_matrix(f, m, mname)

    return yamlfile


def saveyaml(yamlfile, mat):
    """Save a numpy array to YAML file importable by OpenCV"""

    def _write_matrix(f, M):
        f.write('    mtx_01: !!opencv-matrix\n')
        f.write('       rows: %d\n' % M.shape[0])
        f.write('       cols: %d\n' % (M.shape[1] if M.ndim == 2 else 1))
        f.write('       dt: f\n')
        f.write('       data: [ ')
        datastr = ''
        for (k, x) in enumerate(M.flatten()):
            datastr += '%.6e' % x
            if (k + 1 == M.size):
                f.write(datastr)
                break
            datastr += ', '
            if ((k + 1) % 4) == 0:
                f.write(datastr + '\n           ')
                datastr = ''

        f.write(']\n')

    with open(yamlfile, 'w') as f:
        f.write('%YAML:1.0\n')
        _write_matrix(f, mat)

    return yamlfile


def tofilename(s, hyphen=True):
    """Convert arbitrary string to valid filename with underscores
    replacing invalid chars"""
    valid_chars = "-_.%s%s" % (str.ascii_letters, str.digits)
    s = str.replace(s, ' ', '_')
    if hyphen:
        s = str.replace(s, '-', '_')
    return "".join(x for x in s if x in valid_chars)


def isexe(filename):
    """Is the file an executable binary?"""
    return os.path.isfile(filename) and os.access(filename, os.X_OK)


def isinstalled(cmd):
    """Is the command is available on the path"""
    return shutil.which(cmd) is not None


def ispkl(filename):
    """Is the file a pickle archive file"""
    return filename[-4:] == '.pkl' if isstring(filename) and \
        len(filename) >= 4 else False


def ispickle(filename):
    """Is the file a pickle archive file"""
    return isfile(filename) and os.path.exists(filename) and \
        fileext(filename).lower() in ['.pk', '.pkl']


def ndmax(A):
    return np.unravel_index(A.argmax(), A.shape)


def ndmin(A):
    return np.unravel_index(A.argmin(), A.shape)


def ishdf5(path):
    """Is the file an HDF5 file?"""
    # tables.is_hdf5_file(path)
    # tables.is_pytables_file(path)
    (filename, ext) = os.path.splitext(path)
    if (ext is not None) and (len(ext) > 0) and (ext.lower() in ['.h5']):
        return True
    else:
        return False


def filebase(filename):
    """Return c for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    (base, ext) = os.path.splitext(tail)
    return base


def filepath(filename):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return head


def newpath(filename, newdir):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return os.path.join(newdir, tail)


def filefull(f):
    """Return /a/b/c for filename /a/b/c.ext"""
    return splitextension(f)[0]


def filetail(filename):
    """Return c.ext for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return tail


def matread(txtfile, delimiter=' '):
    """Whitespace separated values defining columns, lines define rows.
    Return numpy array"""
    with open(txtfile, 'rb') as csvfile:
        M = [np.float32(row.split(delimiter)) for row in csvfile]
    return np.array(M)


def imlist(imdir):
    """return list of images with absolute path in a directory"""
    return [os.path.abspath(os.path.join(imdir, item))
            for item in os.listdir(imdir)
            if (isimg(item) and not is_hiddenfile(item))]


def videolist(videodir):
    """return list of videos with absolute path in a directory"""
    return [os.path.abspath(os.path.join(videodir, item))
            for item in os.listdir(videodir)
            if (isvideo(item) and not is_hiddenfile(item))]


def dirlist(indir):
    """return list of directories in a directory"""
    return [os.path.abspath(os.path.join(indir, item))
            for item in os.listdir(indir)
            if (os.path.isdir(os.path.join(indir, item)) and
                not is_hiddenfile(item))]


def extlist(indir, ext):
    """return list of files with absolute path in a directory that have
    the provided extension (with the prepended dot)"""
    return [os.path.abspath(os.path.join(indir, item))
            for item in os.listdir(indir)
            if fileext(item) is not None
            and (fileext(item).lower() == ext.lower())]


def writelist(mylist, outfile, mode='w'):
    """Write list of strings to an output file with each row an element of
    the list"""
    with open(outfile, mode) as f:
        for s in mylist:
            f.write(str(s) + '\n')
    return(outfile)


def readlist(infile):
    """Read each row of file as an element of the list"""
    with open(infile, 'r') as f:
        list_of_rows = [r for r in f.readlines()]
    return list_of_rows


def readtxt(infile):
    """Read a text file one string per row"""
    return readlist(infile)


def writecsv(list_of_tuples, outfile, mode='w', separator=','):
    """Write list of tuples to an output csv file with each list element
    on a row and tuple elements separated by comma"""
    list_of_tuples = list_of_tuples if not isnumpy(list_of_tuples) else list_of_tuples.tolist()
    with open(outfile, mode) as f:
        for u in list_of_tuples:
            n = len(u)
            for (k, v) in enumerate(u):
                if (k + 1) < n:
                    f.write(str(v) + separator)
                else:
                    f.write(str(v) + '\n')
    return(outfile)


def readcsv(infile, separator=',', ignoreheader=False):
    """Read a csv file into a list of lists"""
    with open(infile, 'r') as f:
        list_of_rows = [[x.strip() for x in r.split(separator)]
                        for r in f.readlines()]
    return list_of_rows if not ignoreheader else list_of_rows[1:]


def readcsvwithheader(infile, separator=','):
    """Read a csv file into a list of lists"""
    with open(infile, 'r') as f:
        list_of_rows = [[x.strip() for x in r.split(separator)]
                        for r in f.readlines()]
    header_dict = dict()
    for i in range(len(list_of_rows[0])):
        header_dict[list_of_rows[0][i]] = i
    return list_of_rows[1:], header_dict


def imsavelist(imdir, outfile):
    """Write out all images in a directory to a provided file with each
    line containing absolute path to image"""
    return writelist(imlist(imdir), outfile)


def csvlist(imdir):
    """Return a list of absolute paths of *.csv files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir)
            if iscsv(item)]


def pklist(imdir):
    """Return a list of absolute paths of *.pk files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir)
            if ispickle(os.path.join(imdir, item))]


def txtlist(imdir):
    """Return a list of absolute paths of *.txt files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir)
            if istextfile(item) and not is_hiddenfile(item)]


def imlistidx(filelist, idx_in_filename):
    """Return index in list of filename containing index number"""
    return [i for (i, item) in enumerate(filelist)
            if (item.find('%d' % idx_in_filename) > 0)]


def mat2gray(img, min=None, max=None):
    """Convert numpy array to float32 with 1.0=max and 0=min"""
    immin = np.min(img) if min is None else min
    immax = np.max(img) if max is None else max
    if (immax - immin) > 0:
        return (np.float32(img) - immin) / (immax - immin)
    else:
        return img


def mdlist(m, n):
    """Preallocate 2D list of size MxN"""
    return [[None] * n for i in range(m)]


def isurl(path):
    """Is a path a URL?"""
    try:
        return urlparse(path).scheme != "" and \
            '<' not in path and '>' not in path and '"' not in path
    except:
        return False


def isimageurl(path):
    """Is a path a URL with image extension?"""
    return isurl(path) and isimg(path)


def isvideourl(path):
    """Is a path a URL with image extension?"""
    return isurl(path) and isvideo(path)


def isyoutubeurl(path):
    """Is a path a youtube URL?"""
    return isurl(path) and 'youtube.com' in path


def checkerboard(m=8,n=256):
    """m=number of square by column, n=size of final image"""
    return np.array(PIL.Image.fromarray(np.uint8(255 * np.random.rand(m,m,3))).resize((n,n), PIL.Image.NEAREST))


def islist(x):
    """Is an object a python list"""
    return type(x) is list


def isimageobject(x):
    """Is an object a vipy.image class Image, ImageCategory, ImageDetection?"""
    return (str(type(x)) in ["<class 'vipy.image.Image'>",
                             "<class 'vipy.image.ImageCategory'>",
                             "<class 'vipy.image.ImageDetection'>"])


def isvideotype(x):
    """Is an object a vipy.video class Video, VideoCategory, Scene?"""
    return (str(type(x)) in ["<class 'vipy.video.Video'>",
                             "<class 'vipy.video.VideoCategory'>",
                             "<class 'vipy.video.Scene'>"])


def istuple(x):
    """Is an object a python tuple?"""
    return type(x) is tuple


def tolist(x):
    """Convert a python object to a singleton list if not already a list"""
    if type(x) is list:
        return x
    else:
        return [x]


def isimg(path):
    """Is an object an image with a supported image extension
    ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff',
                       '.pgm', '.ppm', '.gif', '.bmp']:
        return True
    else:
        return False


def isvideofile(path):
    """Equivalent to isvideo()"""
    return isvideo(path)


def isimgfile(path):
    """Convenience function for isimg"""
    return isimg(path)


def isimagefile(path):
    """Convenience function for isimg"""
    return isimg(path)


def isgif(path):
    return hasextension(path) and fileext(path).lower() == '.gif'


def isjpeg(path):
    return hasextension(path) and fileext(path).lower() == '.jpg' or fileext(path).lower() == '.jpeg'


def isjpg(path):
    return isjpeg(path)


def iscsv(path):
    """Is a file a CSV file extension?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.csv', '.CSV']:
        return True
    else:
        return False


def isvideo(path):
    """Is a file a video with a known video extension
    ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']:
        return True
    else:
        return False


def isnumpy(obj):
    """Is a python object a numpy array?"""
    return ('numpy' in str(type(obj)))


def istextfile(path):
    """Is the given file a text file?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.txt'] and (filename[0] != '.'):
        return True
    else:
        return False


def isxml(path):
    """Is the given file an xml file?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.xml']:
        return True
    else:
        return False


def bgr2gray(im_bgr):
    """Wrapper for numpy uint8 BGR image to uint8 numpy grayscale"""
    return np.array(PIL.Image.fromarray(im_bgr).convert('L'))


def gray2bgr(im_gray):
    """Wrapper for numpy float32 gray image to uint8 numpy BGR"""
    return np.array(PIL.Image.fromarray(im_gray, mode='F').convert('RGB'))[:,:,::-1]  # Gray -> RGB -> BGR


def gray2rgb(im_gray):
    return bgr2rgb(gray2bgr(im_gray))


def bgr2rgb(im_bgr):
    """Wrapper for numpy BGR uint8 to numpy RGB uint8"""
    return np.array(im_bgr)[:,:,::-1]


def rgb2bgr(im_rgb):
    """same as bgr2rgb"""
    return bgr2rgb(im_rgb)


def bgr2hsv(im_bgr):
    return np.array(PIL.Image.fromarray(bgr2rgb(im_bgr)).convert('HSV'))  # BGR -> RGB -> HSV


def gray2hsv(im_gray):
    return np.array(PIL.Image.fromarray(gray2rgb(im_gray)).convert('HSV'))  # Gray -> RGB -> HSV


def isarchive(filename):
    """Is filename a zip or gzip compressed tar archive?"""
    (filebase, ext) = splitextension(filename)
    if (ext is not None) and (len(ext) > 0) and (ext.lower() in [
            '.egg', '.jar', '.tar', '.tar.bz2', '.tar.gz',
            '.tgz', '.tz2', '.zip', '.gz']):
        return True
    else:
        (filebase, ext) = splitextension(ext[1:])
        if (ext is not None) and (len(ext) > 0) and (ext.lower() in ['.bz2']):
            return True
        else:
            return False


def tempfilename(suffix):
    fd, path = tempfile.mkstemp(suffix)
    os.close(fd)
    return path


def templike(filename):
    suffix = fileext(filename)
    fd, path = tempfile.mkstemp(suffix)
    os.close(fd)
    return path


def tempimage(ext='jpg'):
    """Create a temporary image with the given extension"""
    if ext[0] == '.':
        ext = ext[1:]
    return tempfilename(suffix='.' + ext)


def temppng():
    """Create a temporay PNG file"""
    return tempimage('png')


def temppickle():
    """Create a temporary pickle file"""
    return tempfilename(suffix='.pkl')


def tempjpg():
    """Create a temporary JPG file in system temp directory"""
    return tempimage('jpg')


def tempMP4():
    """Create a temporary MP4 file in system temp directory"""
    return tempimage('mp4')


def tmpjpg():
    """Create a temporary JPG file in /tmp"""
    return '/tmp/%s.jpg' % uuid.uuid4().hex


def tempcsv():
    """Create a temporary CSV file"""
    return tempfilename(suffix='.csv')


def temppkl():
    """Create a temporary pickle file"""
    return temppickle()


def tempyaml():
    """Create a temporary YAML file"""
    return tempfilename(suffix='.yml')


def temppdf():
    """Create a temporary PDF file"""
    return tempfilename(suffix='.pdl')


def mktemp(ext):
    """Create a temporary file with extension .ext"""
    return tempfilename(suffix='.' + ext)


def tempdir():
    """Wrapper around tempfile, because I can never remember the syntax"""
    return tempfile.gettempdir()


def imread(imfile):
    """Wrapper for opencv imread. Note that color images are imported as
    BGR!"""
    return np.array(PIL.Image.open(imfile))[:,:,::-1]


def imrescale(im, scale):
    (height, width) = (im.shape[0], im.shape[1])
    return np.array(PIL.Image.fromarray(im).resize((int(np.round(scale * width)), int(np.round(scale * height))), PIL.Image.BILINEAR))


def imresize(im, rows, cols):
    return np.array(PIL.Image.fromarray(im).resize((rows, cols), PIL.Image.BILINEAR))


def touch(filename, mystr=''):
    """Create an empty file containing mystr"""
    f = open(filename, 'w')
    f.write(str(mystr))
    f.close()


def isboundingbox(obj):
    return isinstance(obj, vipy.geometry.BoundingBox)


class Stopwatch(object):
    """Return elapsed system time in seconds between calls to enter and exit"""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.start = time.time()
        self.last = self.start
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start

    def since(self, start=False):
        """Return seconds since start or last call to this method"""
        now = time.time()
        dur = now - self.start if start is True else now - self.last
        self.last = now
        return dur

    def reset(self):
        self.start = time.time()
        self.last = self.start
        return self


def isfile(path):
    """Wrapper for os.path.isfile"""
    return os.path.isfile(str(path))


def isstring(s):
    """Is an object a python string or unicode string?"""
    return isinstance(s, str)  # python3


def timestamp():
    """Return date and time string in form DDMMMYY_HHMMSS"""
    return str.upper(strftime("%d%b%y_%I%M%S%p", localtime()))


def minutestamp():
    """Return date and time string in form DDMMMYY_HHMM"""
    return str.upper(strftime("%d%b%y_%I%M%p", localtime()))


def datestamp():
    """Return date and time string in form DDMMMYY"""
    return str.upper(strftime("%d%b%y", localtime()))


def print_update(status):
    status = status + chr(8) * (len(status) + 1)
    print((status,))  # space instead of newline
    sys.stdout.flush()


def remkdir(path, flush=False):
    """Create a given directory if not already exists"""
    if os.path.isdir(path) is False:
        os.makedirs(path)
    elif flush is True:
        shutil.rmtree(path)
        os.makedirs(path)
    return os.path.abspath(os.path.expanduser(path))


def toextension(filename, newext):
    """Convert myfile.ext to myfile.new.  Newext does not contain a period."""
    if '.' in newext:
        newext = newext.split('.')[1]
    (filename, oldext) = splitextension(filename)
    return filename + '.' + str(newext)


def splitextension(filename):
    """Given /a/b/c.ext return tuple of strings ('/a/b/c', '.ext')"""
    (head, tail) = os.path.split(filename)
    try:
        (base, ext) = str.split(tail, '.', 1)  # for .tar.gz
        ext = '.' + ext
    except:
        base = tail
        ext = None
    return (os.path.join(head, base), ext)  # for consistency with splitext


def hasextension(filename):
    """Does the provided filename have a file extension (e.g. /path/to/file.ext) or not (e.g. /path/to/file)"""
    return fileext(filename) is not None


def fileext(filename):
    """Given filename /a/b/c.ext return .ext"""
    (head, tail) = os.path.split(filename)
    try:
        parts = str.rsplit(tail, '.', 2)
        if len(parts) == 3:
            ext = '.%s.%s' % (parts[1], parts[2])  # # tar.gz
        else:
            ext = '.' + parts[1]

    except:
        base = tail
        ext = None
    return ext


def ismacosx():
    """Is the current platform MacOSX?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Darwin'


def islinux():
    """is the current platform Linux?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Linux'


def linuxversion():
    """Return linux version"""
    if islinux():
        with open('/etc/redhat-release') as f:
            v = f.readlines()
            m = re.match('[a-zA-Z ]+([0-9]+\.[0-9]+)', v[0])
            return m.groups(1)[0]
    return None


def imcrop(img, bbox):
    """Crop a 2D or 3D numpy image given a vipy.geometry.BoundingBox"""
    return img[bbox.xmin():bbox.xmax(), bbox.ymin():bbox.ymax()]


class Failed(Exception):
    """Raised when unit test fails to throw an exception"""
    pass
