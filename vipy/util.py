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
from datetime import datetime
import sys
import csv
import hashlib
import shutil
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
import pathlib
import socket
import warnings
import copy
import bz2

try:
    import ujson as json  # faster
except ImportError:
    import json


def bz2pkl(filename, obj=None):
    """Read/Write compressed pickle file"""
    assert filename[-8:] == '.pkl.bz2', "Invalid filename - must be '*.pkl.bz2'"
    if obj is not None:
        f = bz2.BZ2File(filename, 'wb')
        cPickle.dump(obj, f)
        f.close()
        return filename
    else:
        f = bz2.BZ2File(filename, 'rb')
        obj = cPickle.load(f)
        f.close()
        return obj
        

def mergedict(d1, d2):
    assert isinstance(d1, dict) and isinstance(d2, dict)
    d = copy.deepcopy(d1)
    d.update(d2)
    return d


def hascache():
    """Is the VIPY_CACHE environment variable set?"""
    return 'VIPY_CACHE' in os.environ


def tocache(filename):
    """If the VIPY_CACHE environment variable is set, then return the filename in the cache"""
    return os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(filename)) if hascache() else filename


def try_import(package, pipname=None, message=None):
    """Show a helpful error message for missing optional packages"""
    try:
        importlib.import_module(package)
    except:
        if message is not None:
            raise ImportError(message)
        else:
            raise ImportError('Optional package "%s" not installed -  Run "pip install %s" or "pip install vipy[all]" ' % (package, package if pipname is None else pipname))


def findyaml(basedir):
    """Return a list of absolute paths to yaml files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.yml')]


def findpkl(basedir):
    """Return a list of absolute paths to pkl files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.pkl')]


def findjson(basedir):
    """Return a list of absolute paths to json files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.json')]

def findimage(basedir):
    """Return a list of absolute paths to image files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*') if isimage(str(path.resolve()))]

def findvideo(basedir):
    """Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*') if isvideo(str(path.resolve()))]
    

def readyaml(yamlfile):
    """Read a yaml file and return a parsed dictionary, this is slow for large yaml files"""
    try_import('yaml', 'pyyaml')
    import yaml
    with open(yamlfile, 'r') as f:
        return yaml.load(f.read(), Loader=yaml.Loader)  # yaml.CLoader is faster, but not installed via pip


def count_images_in_subdirectories(indir):
    """Count the total number of images in indir/subdir1, indir/subdir2, go down only one level and no further..."""
    num_files = 0
    for d in dirlist(outdir):
        num_files += len(imlist(d))
    return num_files


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
    return len(filename) > 5 and filename[-5:] == '.json'


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
    """Return dictionary of keys and lists from groupby on unsorted inset, where keyfunc is a lambda function on elements in inset"""
    return {k: list(v) for (k, v) in groupby(inset, keyfunc)}

def countby(inset, keyfunc):
    """Return dictionary of keys and group sizes for a grouping of the input list by keyfunc lambda function""" 
    return {k:len(v) for (k,v) in groupbyasdict(inset, keyfunc).items()}

def most_frequent(inset):
    """Return the most frequent element as determined by element equality"""
    return sorted([(k,v) for (k,v) in vipy.util.countby(inset, lambda x: x).items()], key=lambda y: y[1])[-1][0]

def countbyasdict(inset, keyfunc):
    """Alias for countby"""
    return countby(inset, keyfunc)

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
    return indir

def dividelist(inlist, fractions):
    """Divide inlist into a list of lists such that the size of each sublist is the requseted fraction of the original list. 
       This operation is deterministic and generates the same division in multiple calls.
       
       Input:
         -inlist=list
         -fractions=(0.1, 0.7, 0.2)   An iterable of fractions that must be non-negative and sum to one
    """
    assert all([f >= 0 and f <=1 for f in fractions])
    assert np.sum(fractions) == 1
    assert len(inlist) >= len(fractions)
    N = np.int32(np.maximum(0, np.ceil(len(inlist)*np.array(fractions))))
    outlist = []
    for n in N:
        outlist.append(inlist[0:n])
        inlist = inlist[n:]
    return outlist

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
    assert size_per_chunk >= 1 and overlap_per_chunk >= 0 and size_per_chunk > overlap_per_chunk
    return [inlist[i-size_per_chunk:i] for i in range(size_per_chunk, len(inlist)+(size_per_chunk-overlap_per_chunk), size_per_chunk - overlap_per_chunk)]


def chunklistwithoverlap(inlist, size_per_chunk, overlap_per_chunk):
    """Alias for chunklistWithOverlap"""
    return chunklistWithOverlap(inlist, size_per_chunk, overlap_per_chunk)

def imwritejet(img, imfile=None):
    """Write a grayscale numpy image as a jet colormapped image to the
    given file"""
    if imfile is None:
        imfile = temppng()

    if isnumpy(img):
        if img.ndim == 2:
            cm = plt.get_cmap('gist_rainbow')
            PIL.Image.fromarray(np.uint8(255 * cm(img)[:,:,:3])).save(os.path.expanduser(imfile))
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
            PIL.Image.fromarray(img).save(os.path.expanduser(imfile))
        elif img.dtype == np.dtype('float32'):
            # Convert [0, 1.0] to uint8 [0,255]
            PIL.Image.fromarray(np.uint8(img * 255.0)).save(os.path.expanduser(imfile))
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

    imfile = os.path.expanduser(imfile)
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


def print_and_return(x):
    print(x)
    return x


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


def _json_class_registry():
    import vipy.video
    import vipy.image
    registry = {"<class 'vipy.video.Scene'>":vipy.video.Scene.from_json,
                "<class 'vipy.video.Video'>":vipy.video.Video.from_json,
                "<class 'vipy.video.VideoCategory'>":vipy.video.VideoCategory.from_json,
                "<class 'vipy.image.Image'>":vipy.image.Image.from_json,
                "<class 'vipy.image.ImageCategory'>":vipy.image.ImageCategory.from_json,
                "<class 'vipy.image.ImageDetection'>":vipy.image.ImageDetection.from_json,            
                "<class 'vipy.image.Scene'>":vipy.image.Scene.from_json,
                "<class 'vipy.geometry.BoundingBox'>":vipy.geometry.BoundingBox.from_json,
                "<class 'vipy.object.Track'>":vipy.object.Track.from_json,
                "<class 'vipy.object.Detection'>":vipy.object.Detection.from_json,
                "<class 'vipy.activity.Activity'>":vipy.activity.Activity.from_json}
    try:
        import pycollector.video
        registry.update( {"<class 'pycollector.video.Video'>":pycollector.video.Video.from_json} )
    except:
        registry.update( {"<class 'pycollector.video.Video'>":lambda x: exec("raise ValueError(\"<class 'pycollector.video.Video'> not found - Run 'pip install pycollector' \")")})        
    try:
        import pycollector.admin.video
        registry.update( {"<class 'pycollector.admin.video.Video'>":pycollector.admin.video.Video.from_json} )
    except:
        registry.update( {"<class 'pycollector.admin.video.Video'>":lambda x: exec("raise ValueError(\"<class 'pycollector.admin.video.Video'> not found - This is for admin use only \")")})        

    return registry
            

def saveas(vars, outfile=None, format='dill'):
    """Save variables as a dill pickled file"""
    outfile = temppickle() if outfile is None else outfile

    remkdir(filepath(outfile))
    if format in ['dill']:
        dill.dump(vars, open(outfile, 'wb'))
        return outfile
    
    elif format == 'json':
        saveobj = vars
        class_registry = _json_class_registry()
        if isinstance(saveobj, list) and all([str(type(d)) in class_registry for d in saveobj]):
            j = [{str(type(d)):d.json(encode=False)} for d in saveobj] if isinstance(saveobj, list) else ({str(type(d)):d.json(encode=False)} for d in saveobj)
        elif str(type(saveobj)) in class_registry:
            j = {str(type(saveobj)):saveobj.json(encode=False)}
        else:
            j = saveobj

        s = json.dumps(j, ensure_ascii=False)  # load to memory (faster than json.dump), will throw exception if it cannot serialize
        with open(outfile, 'w') as f:
            f.write(s)            
        return outfile
        
    else:
        raise ValueError('unknown serialization format "%s"' % format)

    return outfile


def loadas(infile, format='dill'):
    """Load variables from a dill pickled file"""
    
    if format in ['dill', 'pkl']:
        return dill.load(open(infile, 'rb'))
    elif format == 'json':
        with open(infile, 'r') as f:
            loadobj = json.load(f)
        class_registry = _json_class_registry()
        assert isinstance(loadobj, list) or isinstance(loadobj, dict), "invalid vipy JSON serialization format"
        if isinstance(loadobj, list) and all([isinstance(d, dict) for d in loadobj]) and all([c in class_registry for d in loadobj for (c,v) in d.items()]):
            return [class_registry[c](v) for d in loadobj for (c,v) in d.items()]
        elif isinstance(loadobj, dict) and all([c in class_registry for (c,d) in loadobj.items()]):
            obj = [class_registry[c](v) for (c,v) in loadobj.items()]
            return obj[0] if len(obj) == 1 else obj
        else:
            return loadobj
    else:
        raise ValueError('unknown serialization format "%s"' % format)


def load(infile, abspath=False):
    """Load variables from a relocatable archive file format, either Dill Pickle or JSON.
       
       Loading is performed by attemping the following:

       1. load the pickle or json file
       2. if abspath=true, then convert relative paths to absolute paths for object when loaded
       3. If the loaded object is a vipy object (or iterable) and the relocatable path /$PATH is present, try to repath it to the directory containing this archive (this has been deprecated)
       4. If the resulting files are not found, throw a warning
       5. If a large number of objects are loaded, disable garbage collection.

    """
    infile = os.path.abspath(os.path.expanduser(infile))
    if ispkl(infile):
        format = 'dill'
    elif isjsonfile(infile):
        format = 'json'
    else:
        raise ValueError('unknown file type')
    
    obj = loadas(infile, format=format)
    try:
        testobj = tolist(obj)[0]
    except:
        raise ValueError('Invalid archive "%s"' % infile)

    # Relocatable object?
    if hasattr(testobj, 'filename') and testobj.filename() is not None and '/$PATH' in testobj.filename():
        testobj = repath(copy.deepcopy(testobj), '/$PATH', filepath(os.path.abspath(infile)))  # attempt to rehome /$PATH/to/me.jpg -> /NEWPATH/to/me.jpg where NEWPATH=filepath(infile)
        if os.path.exists(testobj.filename()):       # file found
            obj = repath(obj, '/$PATH', filepath(os.path.abspath(infile)))      # rehome everything to the same root path as the pklfile
        else:
            warnings.warn('Loading "%s" that contains redistributable paths - Use vipy.util.distload("%s", datapath="/path/to/your/data") to rehome absolute file paths' % (infile, infile))
    elif hasattr(testobj, 'filename') and testobj.filename() is not None and not os.path.exists(testobj.filename()):
        if hasattr(testobj, 'hasurl') and testobj.hasurl():
            warnings.warn('Loading archive "%s" that contains filename "%s" which does not exist - This must be downloaded from the provided url() using download()' % (infile, testobj.filename()))
        elif not os.path.isabs(testobj.filename()):
            if not abspath:
                warnings.warn('Loading archive "%s" with relative paths.  Changing directory to "%s".  Disable this warning by setting vipy.util.load(..., abspath=True).' % (infile, filepath(infile)))
                os.chdir(filepath(infile))
            else:
                # Absolute path?  The loaded archive will no longer be relocatable, and the videos directory cannot be moved
                pwd = os.getcwd()  
                os.chdir(filepath(infile))  # change to archive directory
                objout = [o.abspath() if o.hasfilename() else o for o in tolist(obj)]  # set absolute paths
                obj = objout if isinstance(obj, list) else objout[0]
                if any([not o.hasfilename() for o in tolist(obj)]):
                    warnings.warn('Loading "%s" that contains absolute path "%s" which does not exist' % (infile, tolist(obj)[0]))
                os.chdir(pwd)  # set it back
        else:
            warnings.warn('Loading "%s" that contains absolute path "%s" which does not exist' % (infile, testobj.filename()))

    # Large vipy object?  Disable garbage collection.
    #   - Python uses reference counting for the primary garbage collection mechanism, but also uses reference cycle checks to search for dependencies between objects.
    #   - All vipy objects are self contained, and do not have reference cycles.  However, there is no way to mark an individual object which does not participate in reference cycle counting.
    #   - This means that a large number of vipy objects, garbage collection can take minutes searching for cycles which are never there.  To fix this, globally disable the garbage collector.
    #   - Note that refernece counting is still performed, we are just disabling reference *cycle* counting using the generational garbage collector.
    #   - This can be re-enabled at any time by "import gc; gc.enable()"
    #   - If you use %autoreload iPython magic command, note that this will be very slow.  You should set %sutoreload 0
    #   - Alternatively, load as JSON and all attributes will be unpacked on demand and stored in a packed format that is not tracked (e.g. tuple of strings) by the reference cycle counter
    if format != 'json' and hasattr(testobj, 'filename') and len(tolist(obj)) > 10000:
        # Should we warn the user?  Probably doesn't matter ...
        #warnings.warn('Loading "%s" that contains a large number of vipy objects - disabling reference cycle checks.  Re-enable at any time using "import gc; gc.enable()"' % (infile))
        import gc; gc.disable()
    return obj


def save(vars, outfile, mode=None):
    """Save variables to an archive file"""
    allowable = set(['.pkl', '.json'])
    if ispkl(outfile):
        format = 'dill'
    elif isjsonfile(outfile):
        format = 'json'
    else:
        raise ValueError('Unknown file extension "%s" - must be in %s' % (fileext(outfile), str(allowable)))
    
    outfile = saveas(vars, outfile, format=format)
    if mode is not None:
        chmod(outfile, mode)
    return outfile


def distload(infile, datapath, srcpath='/$PATH'):
    """Load a redistributable pickle file that replaces absolute paths in datapath with srcpath.  See also vipy.util.distsave(),
       This function has been deprecated, all archives should be distributed with relative paths
    """
    return repath(load(infile), srcpath, datapath)


def distsave(vars, datapath, outfile=None, mode=None, dstpath='/$PATH'):
    """Save a archive file for redistribution, where datapath is replaced by dstpath.  Useful for redistribuing pickle files with absolute paths.  See also vipy.util.distload().
       This function has been deprecated, all archives should be distributed with relative paths       
    """
    vars = vars if (datapath is None or not isvipyobject(vars)) else repath(vars, datapath, dstpath) 
    return save(vars, outfile, mode)


def repath(v, srcpath, dstpath):
    """Change the filename with prefix srcpath to dstpath, for any element in v that supports the filename() api"""
    if not islist(v) and (hasattr(v, 'filename') and hasattr(v, 'clone')):
        vc = v.filename( v.filename().replace(os.path.normpath(srcpath), os.path.normpath(dstpath))) if v.filename() is not None else v
    elif islist(v) and all([(hasattr(vv, 'filename') and hasattr(vv, 'clone')) for vv in v]):
        vc = [vv.filename( vv.filename().replace(os.path.normpath(srcpath), os.path.normpath(dstpath))) if vv.filename() is not None else vv for vv in v ]
    elif isstring(v):
        vc = v.replace(os.path.normpath(srcpath), os.path.normpath(dstpath))
    else:
        raise ValueError('Input must be a singleton or list of vipy.image.Image() or vipy.video.Video() objects, not type "%s"' % (str(type(v))))
    return vc
    

def scpsave(V):
    import vipy.image
    import vipy.video
    if (isinstance(V, vipy.image.Image) or isinstance(V, vipy.video.Video)) and V.hasfilename():        
        v = V
        v = v.clone().url('scp://%s:%s' % (socket.gethostname(), v.filename())).nofilename()
    elif islist(V) and all([isinstance(v, vipy.image.Image) or isinstance(v, vipy.video.Video) for v in V]):
        v = [v.clone().url('scp://%s:%s' % (socket.gethostname(), v.abspath().filename())).nofilename() for v in V]
    else:
        v = V # no vipy objects

    pklfile = 'scp://%s:%s' % (socket.gethostname(), save(v, temppkl()))
    cmd = "V = vipy.util.scpload('%s')" % pklfile
    print('[vipy.util.scpsave]: On a remote machine where you have public key ssh access to this machine run:\n>>> %s\n' % cmd)
    return pklfile


def scpload(url):
    import vipy.downloader
    return load(vipy.downloader.scp(url, templike(url)))


def load_opencv_yaml(yamlfile):
    """Load a numpy array from YAML file exported from OpenCV"""
    return np.squeeze(np.array(cv.Load(yamlfile)))


def matrix_to_opencv_yaml(yamlfile, mtxlist, mtxname=None):
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


def save_opencv_yaml(yamlfile, mat):
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

def ishtml(filename):
    """Is the file an HTMLfile"""
    return fileext(filename).lower() == '.html'

def ispickle(filename):
    """Is the file a pickle archive file"""
    return isfile(filename) and os.path.exists(filename) and (((fileext(filename) is not None) and fileext(filename).lower() in ['.pk', '.pkl']) or (filename[-4:] == '.pkl'))


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


def filepath(filename, depth=0):
    """Return /a/b/c for filename /a/b/c/d.ext, /a/b for filename /a/b/c/d.ext if depth=1, etc"""
    (head, tail) = os.path.split(filename)
    for k in range(depth):
        (head, tail) = os.path.split(head)           
    return head


def delpath(indir, filename):
    """Return c/d.ext for filename /a/b/c/d.ext and indir /a/b"""
    assert indir in filename, 'Path "%s" not found in filename "%s"' % (indir, filename)
    indir = os.path.join(indir, '')  # /a/b -> /a/b/
    return filename.split(indir)[1]

    
def newpath(filename, newdir):
    """Return /d/e/c.ext for filename /a/b/c.ext and newdir /d/e/"""
    (head, tail) = os.path.split(filename)
    return os.path.join(newdir, tail)

def newprefix(filename, newprefix, depth=0):
    """Return /a/b/c/h/i.ext for filename /f/g/h/i.ext and prefix /a/b/c and depth=1"""
    p = filepath(filename, depth=depth)
    return os.path.normpath(filename.replace(p, newprefix))

def newpathdir(filename, olddir, newdir, n=1):
    """Return /a/b/n/d/e.ext for filename=/a/b/c/d/e.ext, olddir=c, newdir=n"""
    p = pathlib.PurePath(filename)
    assert sum([d == olddir for d in p.parts]) == n, "Path must have exactly %s directory matches" % n
    return os.path.join(*[d.replace(olddir, newdir) for d in list(p.parts)])


def newpathroot(filename, newroot):
    """Return /r/b/c.ext for filename /a/b/c.ext and new root directory r"""
    p = pathlib.PurePath(filename)
    path = list(p.parts)    
    if len(p.root) == 0:
        path[0] = newroot
    else:
        path[1] = newroot
    return os.path.join(*path)

def topath(filename, newdir):
    """Alias for newpath"""
    return newpath(filename, newdir)


def filefull(f):
    """Return /a/b/c for filename /a/b/c.ext"""
    ext = fileext(f, multidot=True, withdot=True)
    return f.replace(ext, '') if ext is not None else None


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


def dirlist_sorted_bycreation(indir):
    """Sort the directory list from newest first to oldest last by creation date"""
    return sorted(dirlist(indir), key=lambda d: os.stat(d).st_ctime, reverse=True)


def extlist(indir, ext):
    """return list of files with absolute path in a directory that have
    the provided extension (with the prepended dot, ext='.mp4')"""
    return [os.path.abspath(os.path.join(indir, item))
            for item in os.listdir(indir)
            if fileext(item) is not None
            and (fileext(item).lower() == ext.lower())]

def listext(indir, ext):
    """Alias for extlist"""
    return extlist(indir, ext)

def jsonlist(indir):
    """return list of fJSON iles with absolute path in a directory"""
    #return extlist(indir, ext='.json')  # FIXME: broken.for.wonky.filenames.with.dots.json
    return [os.path.abspath(os.path.join(indir, item))
            for item in os.listdir(indir)
            if len(item) > 5 and item[-5:] == '.json']

def listjson(indir):
    """Alias for jsonlist"""
    return jsonlist(indir)

def writelist(mylist, outfile, mode='w'):
    """Write list of strings to an output file with each row an element of
    the list"""
    outfile = os.path.abspath(os.path.expanduser(outfile))
    with open(outfile, mode) as f:
        for s in mylist:
            f.write(str(s) + '\n')
    return(outfile)


def readlist(infile):
    """Read each row of file as an element of the list"""
    with open(infile, 'r') as f:
        list_of_rows = [r.strip() for r in f.readlines()]
    return list_of_rows


def readtxt(infile):
    """Read a text file one string per row"""
    return readlist(infile)


def writecsv(list_of_tuples, outfile, mode='w', separator=','):
    """Write list of tuples to an output csv file with each list element
    on a row and tuple elements separated by comma"""
    list_of_tuples = list_of_tuples if not isnumpy(list_of_tuples) else list_of_tuples.tolist()
    outfile = os.path.abspath(os.path.expanduser(outfile))
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


def pklist(indir):
    """Return a list of absolute paths of *.pk files in current directory"""
    return listpkl(indir)

def listpkl(indir):
    """Return a list of absolute paths of *.pk files in current directory"""
    return [os.path.join(indir, item) for item in os.listdir(indir)
            if ispickle(os.path.join(indir, item))]


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

def shortuuid(n=16):
    return hashlib.sha256(uuid.uuid1().hex.encode('utf-8')).hexdigest()[0:n] 


def isimageurl(path):
    """Is a path a URL with image extension?"""
    return isurl(path) and isimg(path)


def isvideourl(path):
    """Is a path a URL with video extension?"""
    return isurl(path) and isvideo(path)


def isS3url(path):
    """Is a path a URL for an S3 object?"""
    return isurl(path) and urlparse(path).scheme == 's3'


def isyoutubeurl(path):
    """Is a path a youtube URL?"""
    return isurl(path) and 'youtube.com' in path


def checkerboard(m=8,n=256):
    """m=number of square by column, n=size of final image"""
    return np.array(PIL.Image.fromarray(np.uint8(255 * np.random.rand(m,m,3))).resize((n,n), PIL.Image.NEAREST))


def islist(x):
    """Is an object a python list"""
    return type(x) is list


def islistoflists(x):
    """Is an object a python list of lists x=[[1,2], [3,4]]"""
    return type(x) is list and type(x[0]) is list


def istupleoftuples(x):
    """Is an object a python list of lists x=[[1,2], [3,4]]"""
    return type(x) is tuple and type(x[0]) is tuple


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

def isvideoobject(x):
    return isvideotype(x)


def isvipyobject(x):
    import vipy.image
    import vipy.video
    return ((isinstance(x, vipy.image.Image) or isinstance(x, vipy.video.Video)) 
            or (islist(x) or istuple(x) and all([isinstance(v, vipy.image.Image) or isinstance(v, vipy.video.Video) for v in x]))
            or (isinstance(x, dict) and all([isinstance(v, vipy.image.Image) or isinstance(v, vipy.video.Video) for (k,v) in x.items()])))


def istuple(x):
    """Is an object a python tuple?"""
    return isinstance(x, tuple)


def tolist(x):
    """Convert a python tuple or singleton object to a list if not already a list """
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):        
        return list(x)
    elif isinstance(x, set):        
        return list(x)
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

def isimage(path):
    return isimg(path)
    

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

def iswebp(path):
    return hasextension(path) and fileext(path).lower() == '.webp'

def ispng(path):
    return hasextension(path) and (fileext(path).lower() == '.png' or fileext(path).lower() == '.apng')

def isgif(path):
    return hasextension(path) and fileext(path).lower() == '.gif'


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
    """Is a filename in path a video with a known video extension
    ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']?"""
    (filename, ext) = os.path.splitext(path)
    if ext.lower() in ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']:
        return True
    else:
        return False


def isnumpy(obj):
    """Is a python object a numpy object?"""
    return ('numpy' in str(type(obj)))

def isnumpyarray(obj):
    """Is a python object a numpy array?"""
    return isnumpy(obj) and 'numpy.ndarray' in str(type(obj))


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

def istgz(filename):
    return filename[-4:] == '.tgz' or filename[-7:] == '.tar.gz'

def isbz2(filename):
    return filename[-4:] == '.bz2' or filename[-8:] == '.tar.bz2'

def tempfilename(suffix):
    """Create a temporary filename $TEMPDIR/$UUID.suffix, suffix should include the dot such as suffix='.jpg', """
    return os.path.join(tempfile.gettempdir(), '%s%s' % (shortuuid(), suffix))


def totempdir(filename):
    """Convert a filename '/patj/to/filename.ext' to '/tempdir/filename.ext'"""
    return os.path.join(tempfile.gettempdir(), filetail(filename))


def templike(filename):
    """Create a new temporary filename with the same extension as filename"""
    return tempfilename(fileext(filename))


def cached(filename):
    """Create a new filename in the cache, or tempdir if not found"""
    if 'VIPY_CACHE' in os.environ:
        return os.path.join(remkdir(os.environ['VIPY_CACHE']), filetail(filename))
    else:
        return totempdir(filename)


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
    return tempfilename(suffix='.mp4')


def tmpjpg():
    """Create a temporary JPG file in /tmp"""
    return '/tmp/%s.jpg' % uuid.uuid4().hex


def tempcsv():
    """Create a temporary CSV file"""
    return tempfilename(suffix='.csv')

def temphtml():
    """Create a temporary HTMLfile"""
    return tempfilename(suffix='.html')


def temppkl():
    """Create a temporary pickle file"""
    return temppickle()


def tempyaml():
    """Create a temporary YAML file"""
    return tempfilename(suffix='.yml')


def tempjson():
    """Create a temporary JSON file"""
    return tempfilename(suffix='.json')


def temppdf():
    """Create a temporary PDF file"""
    return tempfilename(suffix='.pdf')


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

    def duration(self):
        """Time in seconds since last reset"""
        return time.time() - self.start

    
class Timer(object):
    """Pretty print elapsed system time in seconds between calls to enter and exit

       >>> t = Timer():
       >>> [some code]
       >>> print(t)
       >>> [some more code]
       >>> print(t)

       >>> with Timer():
       >>>    [some code]
       

    """
    def __enter__(self):
        self._begin = time.time()
        self._last = self._begin
        return self
        
    def __exit__(self, *args):
        print(self.__repr__())

    def __init__(self, sprintf_next=None, sprintf_first=None):
        self._sprintf_next = '[vipy.util.timer]: elapsed=%1.6fs, total=%1.6fs' if sprintf_next is None else sprintf_next
        self._sprintf_first = '[vipy.util.timer]: elapsed=%1.6fs' if sprintf_first is None else sprintf_first
        self._begin = time.time()
        self._last = self._begin
        self._laps = 0        
        try:
            self._sprintf_next % (1.0, 1.0)
            self._sprintf_first % (1.0)            
        except:
            raise ValueError('Printed display string must be a sprintf style string with one or two number variable like "Elapsed=%1.6f since=%1.6f"')
            
    def __repr__(self):
        s = str(self._sprintf_next % (time.time() - self._last, (time.time() - self._begin))) if self._laps > 0 else str(self._sprintf_first % (time.time() - self._begin))
        self._last = time.time()
        self._laps += 1
        return s

        
def isfile(path):
    """Wrapper for os.path.isfile"""
    return os.path.isfile(str(path))


def isstring(s):
    """Is an object a python string or unicode string?"""
    return isinstance(s, str)  # python3


def timestamp():
    """Return date and time string in form DDMMMYY_HHMMSS"""
    return str.upper(strftime("%d%b%y_%I%M%S%p", localtime()))

def clockstamp():
    """Datetime stamp in local timezone with second resolution with format Year-Month-Day Hour:Minute:Second"""    
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")    

def minutestamp():
    """Return date and time string in form DDMMMYY_HHMM"""
    return str.upper(strftime("%d%b%y_%I%M%p", localtime()))


def datestamp():
    """Return date and time string in form DDMMMYY"""
    return str.upper(strftime("%d%b%y", localtime()))



def remkdir(path, flush=False):
    """Create a given directory if not already exists"""
    if os.path.isdir(path) is False and len(path) > 0:
        os.makedirs(path)
    elif flush is True:
        shutil.rmtree(path)
        os.makedirs(path)
    return os.path.abspath(os.path.expanduser(path))


def rermdir(path):
    """Recursively delete a given directory (if exists), and remake it"""
    return remkdir(path, flush=True)


def premkdir(filename):
    """create directory /path/to/subdir if not exist for outfile=/path/to/subdir/file.ext"""
    return remkdir(filepath(filename))


def newbase(filename, base):
    """Convert filename=/a/b/c.ext base=d -> /a/b/d.ext"""
    return os.path.join(filepath(filename), '%s.%s' % (base, fileext(filename, withdot=False)))

def toextension(filename, newext):
    """Convert filenam='/path/to/myfile.ext' to /path/to/myfile.xyz, such that newext='xyz' or newext='.xyz'"""
    if '.' in newext:
        newext = newext.split('.')[-1]
    (filename, oldext) = splitextension(filename)
    return filename + '.' + str(newext)


def splitextension(filename):
    """Given /a/b/c.ext return tuple of strings ('/a/b/c', '.ext')"""
    (head, tail) = os.path.split(filename)
    try:
        (base, ext) = str.rsplit(tail, '.', 1)  # for .tar.gz
        ext = '.' + ext
    except:
        base = tail
        ext = None
    return (os.path.join(head, base), ext)  # for consistency with splitext


def hasextension(filename):
    """Does the provided filename have a file extension (e.g. /path/to/file.ext) or not (e.g. /path/to/file)"""
    return fileext(filename) is not None


def fileext(filename, multidot=True, withdot=True):
    """Given filename /a/b/c.ext return '.ext', or /a/b/c.tar.gz return '.tar.gz'.   If multidot=False, then return '.gz'.  If withdot=False, return 'ext'"""
    (head, tail) = os.path.split(filename)
    try:
        parts = str.rsplit(tail, '.', 2)
        if len(parts) == 3 and multidot:
            ext = '.%s.%s' % (parts[1], parts[2])  # .tar.gz
        elif len(parts) == 3 and not multidot:
            ext = '.%s' % (parts[2])  # .gz            
        else:
            ext = '.' + parts[1]  # .mp4

    except:
        base = tail
        ext = None
    return ext if withdot else ext[1:]

def mediaextension(filename):
    """Return '.mp4' for filename='/a/b/c.mp4'"""
    return fileext(filename, multidot=False)

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


def string_to_pil_interpolation(interp):
    """Internal function to convert interp string to interp object"""
    assert interp in ['bilinear', 'bicubic', 'nearest'], "Invalid interp - Must be in ['bilinear', 'bicubic', 'nearest']"
    if interp == 'bilinear':
        return PIL.Image.BILINEAR
    elif interp == 'bicubic':
        return PIL.Image.BICUBIC
    elif interp == 'nearest':
        return PIL.Image.NEAREST
    else:
        raise  # should never get here
