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
import re
import uuid
import builtins
import pickle as cPickle
import PIL
import matplotlib.pyplot as plt
import itertools
from itertools import tee, chain
import importlib
import pathlib
import socket
import warnings
import copy
import bz2
import random
import gc

from vipy.globals import log

ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

try:
    import ujson as json  # faster
except ImportError:
    import json

try:
    import dill as default_pickle
except:
    import pickle as default_pickle
    
def class_registry():
    """Return a dictionary mapping str(type(obj)) to a JSON loader for all vipy objects.

    This function is useful for JSON loading of vipy objects to map to the correct deserialization method.
    """

    import vipy.video
    import vipy.image
    import vipy.dataset

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

    registry.update( {None: cPickle.loads} )  # fallback on generic pickle dumps
    return registry
            

def save(vars, outfile=None, backup=False):
    """Save variables to an archive file.

    This function allows vipy objects to be serialized to disk for later loading.

    ```python
    im = vipy.image.owl()
    im = vipy.util.load(vipy.util.save(im))   # round trip
    ```

    Args:
        vars: A python object to save.  This can be any serializable python object
        outfile:  An output file to save.  Must have extension [.pkl, .json, .pkl.bz2].  If None, will save to a temporary JSON file.
        backup [bool]:  If true and the outfile already exists, make a copy and save as outfile.bak before overwriting
    Returns
        A path to the saved archive file.  Load using `vipy.util.load`. 

    .. note:: JSON is preferred as an archive format for vipy.  Be sure to install the excellent ultrajson library (pip install ujson) for fast serialization.
    """
    allowable = set(['.pkl', '.json', '.pkl.bz2'])
    outfile = tempjson() if outfile is None else outfile

    if backup and os.path.exists(outfile):
        shutil.copyfile(outfile, outfile+'.bak')
    remkdir(filepath(outfile))
    if ispkl(outfile):
        with open(outfile, 'wb') as f:
            default_pickle.dump(vars, f)

    elif isjsonfile(outfile):
        saveobj = vars
        registry = class_registry()
        if isinstance(saveobj, list) and all([str(type(d)) in registry for d in saveobj]):
            j = [{str(type(d)):d.json(encode=False)} for d in saveobj] if isinstance(saveobj, list) else ({str(type(d)):d.json(encode=False)} for d in saveobj)
        elif str(type(saveobj)) in registry:
            j = {str(type(saveobj)):saveobj.json(encode=False)}
        else:
            j = saveobj

        s = json.dumps(j, ensure_ascii=False)  # load to memory (faster than json.dump), will throw exception if it cannot serialize
        with open(outfile, 'w') as f:
            f.write(s)            

    elif ispklbz2(outfile):
        return pklbz2(outfile, vars)
    else:
        raise ValueError('Unknown file extension for save file "%s" - must be in %s' % (fileext(outfile), str(allowable)))
    
    return os.path.abspath(outfile)


def load(infile, abspath=True, freeze=True, relocatable=True):
    """Load variables from a relocatable archive file format, either dill pickle, JSON format or JSON directory format.
       
       Loading is performed by attemping the following:

       1. If the input file is a directory, return a `vipy.dataset.Dataset` with lazy loading of all pkl or json files recursively discovered in this directory.
       2. If the input file is a pickle or json file, load it
       3. if abspath=true, then convert relative paths to absolute paths for object when loaded
       4. If freeze=True, then disable the python reference cycle garbage collector for the object loaded by this file
    
    ```python
    im = vipy.image.owl()
    f = vipy.util.save(im)
    im = vipy.util.load(im)
    ```

       Args:
           infile: [str] file saved using `vipy.util.save` with extension [.pkl, .json].  This may also be a directory tree containing json or pkl files 
           abspath: [bool] If true, then convert all vipy objects with relative paths to absolute paths. If False, then preserve relative paths and warn user.
           freeze: [bool] If True, then disable python reference cycle garbage collector for this loaded object. 
           relocatable: [bool] If True, then perform relocatable relative and absolute paths for vipy objects containing filenames
       Returns:
           The object in the archive file
    """
    if freeze:
        gc.disable()
        
    infile = os.path.abspath(os.path.expanduser(infile))

    if ispkl(infile):
        with open(infile, 'rb') as f:
            obj = default_pickle.load(f)
    elif isjsonfile(infile):
        with open(infile, 'r') as f:
            loadobj = json.load(f)
        registry = class_registry()
        assert isinstance(loadobj, list) or isinstance(loadobj, dict), "invalid vipy JSON serialization format"
        if isinstance(loadobj, list) and all([isinstance(d, dict) for d in loadobj]) and all([c in registry for d in loadobj for (c,v) in d.items()]):
            obj = [registry[c](v) for d in loadobj for (c,v) in d.items()]
        elif isinstance(loadobj, dict) and all([c in registry for (c,d) in loadobj.items()]):
            obj = [registry[c](v) for (c,v) in loadobj.items()]
            obj = obj[0] if len(obj) == 1 else obj
        else:
            obj = loadobj
    elif ispklbz2(infile):
        return pklbz2(infile)
    elif os.path.isdir(infile):        
        import vipy.dataset
        return vipy.dataset.Dataset.from_directory(infile)
    else:
        raise ValueError('unknown file type')
    
    # Relocatable vipy object?
    testobj = tolist(obj)[0] if len(tolist(obj)) > 0 else None
    if relocatable and testobj is not None and hasattr(testobj, 'filename') and testobj.filename() is not None:
        if not os.path.isabs(testobj.filename()):
            if not abspath:
                warnings.warn('Loading archive "%s" with relative paths.  Changing directory to "%s".  Disable this warning with vipy.util.load(..., abspath=True).' % (infile, filepath(infile)))
                os.chdir(filepath(infile))
            else:
                # Absolute path?  The loaded archive will no longer be relocatable if you save this to a new archive, and the videos directory cannot be moved
                pwd = os.getcwd()  # save current directory
                os.chdir(filepath(infile))  # change to archive directory
                objout = [o.abspath() if o.filename() is not None else o for o in tolist(obj)]  # set absolute paths relative to archive directory
                obj = objout if isinstance(obj, list) else objout[0]
                os.chdir(pwd)  # restore current directory
        elif not testobj.hasfilename():
            warnings.warn('Loading "%s" that contains path (e.g. "%s") which does not exist' % (infile, testobj.filename()))

    # Large vipy object?  Disable garbage collection.
    #   - Python uses reference counting for the primary garbage collection mechanism, but also uses reference cycle checks to search for dependencies between objects.
    #   - All vipy objects are self contained, and do not have reference cycles.  However, there is no way to mark an individual object which does not participate in reference cycle counting.
    #   - This means that a large number of vipy objects, garbage collection can take minutes searching for cycles which are never there.  To fix this, globally disable the garbage collector.
    #   - Note that refernece counting is still performed, we are just disabling reference *cycle* counting using the generational garbage collector.
    #   - This can be re-enabled at any time by "import gc; gc.enable()"
    #   - If you use %autoreload iPython magic command, note that this will be very slow.  You should set %sutoreload 0
    #   - Alternatively, load as JSON and all attributes will be unpacked on demand and stored in a packed format that is not tracked (e.g. tuple of strings) by the reference cycle counter
    if freeze:
        gc.enable()
        gc.collect()
        gc.freeze() 
    return obj


def is_jsonable(obj):
    """Return true if can be successfully converted to json (without actually doing it) by recursive type checking"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return True  # JSON types
    elif isinstance(obj, (list, tuple)):
        return all(is_jsonable(item) for item in obj)
    elif isinstance(obj, dict):
        # JSON object keys *must* be strings
        return all(isinstance(key, str) and is_jsonable(value) for key, value in obj.items())
    else:
        return False
    
def dirload(indir):
    """Load a directory by recursively searching for loadable archives and loading them into a flat list"""
    return [x for f in findloadable(indir) for x in load(f)]

def dedupe(inlist, f):
    """Deduplicate the list using the provided lambda function which transforms an element to a dedupe key, such that all elements with the same key are duplicates"""
    assert callable(f)
    assert isinstance(inlist, list)
    return list({f(x):x for x in inlist}.values())


def pklbz2(filename, obj=None):
    """Read/Write a bz2 compressed pickle file"""
    assert filename[-8:] == '.pkl.bz2', "Invalid filename - must be '*.pkl.bz2'"
    if obj is not None:
        f = bz2.BZ2File(filename, 'wb')
        default_pickle.dump(obj, f)
        f.close()
        return filename
    else:
        f = bz2.BZ2File(filename, 'rb')
        obj = default_pickle.load(f)
        f.close()
        return obj
        

def catcher(f, *args, **kwargs):
    """Call the function f with the provided arguments, and return (True, result) on success and (False, exception) if there is any thrown exception.

    Useful for parallel processing
    Useful for wrapping a function where execptions are silent.

    For example, attempting to remove a file where the filename may be None or not present

    >>> vipy.util.catcher(lambda f: os.remove(f), None)
    >>> vipy.util.catcher(lambda f: os.remove(f), '/path/to/missing.txt'))

    """
    assert callable(f)
    try:
        return (True, f(*args, **kwargs))
    except Exception as e:
        return (False, str(e))


def mergedict(*args):
    """Combine keys of two or more dictionaries and return a dictionary deep copy.
    
    ```python
    d1 = {1:2}
    d2 = {3:4}
    d3 = mergedict(d1,d2)
    assert d3 == {1:2, 3:4}
    ```

    """
    assert all(isinstance(d, dict) for d in args)
    assert len(args) > 0
    d = copy.deepcopy(args[0])
    for o in args[1:]:
        d.update(o)
    return d


def env(var=None):
    """Return the VIPY environment variable var, returning None if not present, or all environment variables if var=None.  Var is optionally prepended with 'VIPY_'"""
    env = {k:v for (k,v) in os.environ.items() if k.startswith('VIPY_')}
    var = ('VIPY_'+var) if var is not None and not var.startswith('VIPY_') else var
    return env if var is None else (env[var] if var in env else None)

def hascache():
    """Is the VIPY_CACHE environment variable set?"""
    return 'VIPY_CACHE' in os.environ

def cache():
    """If the VIPY_CACHE environment variable set, return it otherwise return tempdir()"""
    return remkdir(os.path.expanduser(os.environ['VIPY_CACHE'])) if hascache() else tempdir()

def tocache(filename):
    """If the VIPY_CACHE environment variable is set, then return the filename=subpath/to/file.ext in the cache as VIPY_CACHE/subpath/to/file.ext.  Otherwise, return the file in the system temp"""
    return os.path.join(cache(), filename)

def seconds_to_MMSS_colon_notation(sec):
    """Convert integer seconds into MM:SS colon format.  If sec=121, then return '02:01'. """
    assert isinstance(sec, int) and sec <= 99*60 + 59 and sec >= 0
    return '%02d:%02d' % (int(sec/60.0), sec % 60)

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

def findpickle(basedir):
    """Return a list of absolute paths to pkl files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.pickle')]

def findpklbz2(basedir):
    """Return a list of absolute paths to .pkl.bz2 files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.pkl.bz2')]

def findpdf(basedir):
    """Return a list of absolute paths to pdf files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.pdf')]

def findpng(basedir):
    """Return a list of absolute paths to png files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.png')]

def findjpg(basedir):
    """Return a list of absolute paths to jpg files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.jpg')]

def findjson(basedir):
    """Return a list of absolute paths to json files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.json')]

def findtxt(basedir):
    """Return a list of absolute paths to txt files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.txt')]

def findtar(basedir):
    """Return a list of absolute paths to tar files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.tar')]

def findtargz(basedir):
    """Return a list of absolute paths to .pkl.bz2 files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.tar.gz')]

def findimage(basedir):
    """Return a list of absolute paths to image files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*') if isimage(str(path.resolve()))]

def findimages(basedir):
    """Alias for `vipy.util.findimage`"""
    return findimage(basedir)

def findvideo(basedir):
    """Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*') if isvideo(str(path.resolve()))]

def findwebp(basedir):
    """Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*') if iswebp(str(path.resolve()))]

def findxml(basedir):
    """Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir"""
    return [str(path.resolve()) for path in pathlib.Path(basedir).rglob('*.xml')]

def findvideos(basedir):
    """Alias for `vipy.util.findvideo`"""
    return findvideo(basedir)

def findloadable(basedir):
    """Return a list of absolute paths to any archive file loadable by `vipy.load` (*.pkl, *.json, *.pkl.bz2).  Recursively search starting from basedir"""
    return findpkl(basedir) + findjson(basedir) + findpklbz2(basedir) + findpickle(basedir)

def readyaml(yamlfile):
    """Read a yaml file and return a parsed dictionary, this is slow for large yaml files"""
    try_import('yaml', 'pyyaml')
    import yaml    
    try:
            from yaml import CLoader as Loader
    except ImportError:
            from yaml import Loader

    with open(yamlfile, 'r') as f:
        return yaml.load(f.read(), Loader=Loader)  # yaml.CLoader is faster, but not installed via pip


def count_images_in_subdirectories(indir):
    """Count the total number of images in indir/subdir1, indir/subdir2, go down only one level and no further..."""
    num_files = 0
    for d in dirlist(outdir):
        num_files += len(imlist(d))
    return num_files


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
    return isinstance(filename, str) and len(filename) > 5 and filename[-5:] == '.json'


def writejson(d, outfile):
    with open(outfile, 'w') as f:
        json.dump(d, f)
    return outfile


def readjson(jsonfile, strict=True):
    """Read jsonfile=/path/to/file.json and return the json parsed object, issue warning if jsonfile does not have .json extension and strict=True"""
    if not isjsonfile(jsonfile) and strict:
        warnings.warn('Attempting to read JSON file "%s" without .json extension' % jsonfile)
    with open(jsonfile) as f:
        data = json.loads(f.read())
    return data

def tryjson(jsonfile):
    """Attempt to load the json file, return True if loadable, False if not"""
    try:
        readjson(jsonfile, strict=False)
        return True
    except Exception as e:
        return False

def groupby(initer, keyfunc):
    """groupby on unsorted input iterable (initer)"""
    return itertools.groupby(sorted(initer, key=keyfunc), keyfunc)


def vipy_groupby(inset, keyfunc):
    """groupby on unsorted inset"""
    return groupby(inset, keyfunc)


def groupbyasdict(togroup, keyfunc, valuefunc=lambda x: x):
    """Return dictionary of keys and lists from groupby on unsorted inset, where keyfunc is a lambda function on elements in inset
    
    Args:
        togroup: an iteraable of elements to group
        keyfunc:  a lambda function to operate on elements of togroup such that the value returned from the lambda is the equality key for grouping
        valuefunc: a lambda function to operate on elements of to group such that the value returned from the lambda is a transform of the element to be grouped
    Returns:
        A dictionary with unique keys returned from keyfunc, and values are lists of elements in togroup with the same key

    """
    return {k: [valuefunc(vi) for vi in v] for (k, v) in groupby(togroup, keyfunc)}

def countby(inlist, keyfunc=lambda x: x):
    """Return dictionary of keys and group sizes for a grouping of the input list by keyfunc lambda function, sorted by increasing count""" 
    return {k:v for (k,v) in sorted({k:len(v) for (k,v) in groupbyasdict(inlist, keyfunc).items()}.items(), key=lambda x: x[1])}

def sumby(inlist, keyfunc=lambda x: x[0], valuefunc=lambda x: x[1]):
    """Given an inlist of tuples [('a',1), ('a',2), ('b',4)], group by the keyfunc, then sum over the values in valuefunc.  Returns ductionary over keys, sum reduced over valuefunc.  Example returns {'a':3,'b':4}."""
    return {k:sum([valuefunc(vi) for vi in v]) for (k,v) in groupbyasdict(inlist, keyfunc).items()}

def most_frequent(inlist, topk=1):
    """Return the most frequent element as determined by element equality"""
    ranked = list(countby(inlist).keys())
    return ranked[-topk:] if topk is not None else ranked

def countbyasdict(inlist, keyfunc):
    """Alias for `vipy.util.countby`"""
    return countby(inlist, keyfunc)

def softmax(x, temperature=1.0):
    """Row-wise softmax"""
    assert x.ndim == 2
    z = np.exp((x - np.max(x, axis=1).reshape(x.shape[0], 1)) / temperature)
    return z / np.sum(z, axis=1).reshape(x.shape[0], 1)


def permutelist(inlist, seed=None):
    """randomly permute list order.  Permutation is deterministic (same permutation on multiple calls) if specified.  Shuffle is not in place"""
    if seed is not None:
        np.random.seed(seed)  # deterministic        
    outlist = [inlist[k] for k in np.random.permutation(list(range(0, len(inlist))))]
    if seed is not None:
        np.random.seed()  # re-init randomness
    return outlist

def shufflelist(inlist):
    """Randomly shuffle a list, returning the shuffled list. Shuffle is not in-place"""
    return random.sample(inlist, len(inlist))  # sample without replacement

def flatlist(inlist):
    """Convert list of tuples into a list expanded by concatenating tuples.  If the input is already flat, return it unchanged."""
    return [x for r in inlist for x in (r if isinstance(r, (list, tuple, set)) else (r,))]


def rmdir(indir):
    """Recursively remove directory and all contents (if the directory exists)"""
    if os.path.exists(indir) and os.path.isdir(indir):
        shutil.rmtree(indir)
    return indir

def dividelist(inlist, fractions):
    """Divide inlist into a list of lists such that the size of each sublist is the requseted fraction of the original list. 

       This operation is deterministic and generates the same division in multiple calls.
       
    Args:
        inlist: [list]
        fractions: [tuple] such as (0.1, 0.7, 0.2)   An iterable of fractions that must be non-negative and sum to one
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


def pairwise(iterable, prepad=False, postpad=False, padval=None):
    """Equivalent to python-3.10 itertools.pairwise. 
    
    >>> pairwise('ABCD') --> (A,B), (B,C), (C,D)
    >>> pairwise('ABCD', prepad=True, padval=0) --> (0,A), (A,B), (B,C), (C,D)
    >>> pairwise('ABCD', postpad=True) --> (A,B), (B,C), (C,D), (D,None)
    >>> pairwise([(1,1),(2,2)], prepad=True, postpad=True, padval=(None,None)) --> [((None, None), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (None, None))]
    """
    
    a, b = tee(iterable, 2)
    if prepad:
        a = chain([padval], a)
    else:
        b0 = next(b, None)
    if postpad:
        b = chain(b, [padval])
    return zip(a, b)


def chunklist(inlist, num_chunks):
    """Convert list into a list of lists of length num_chunks, such that each element is a list containing a sequential chunk of the original list.
    
    ```python
    (A,B,C) = vipy.util.chunklist(inlist, num_chunks=3)
    assert len(A) == len(inlist) // 3
    ```

    .. note::  The last chunk will be larger for ragged chunks
    """
    (m, n) = (num_chunks, int(np.ceil(float(len(inlist)) / float(num_chunks))))
    return [inlist[i * n:min(i * n + n, len(inlist))] for i in range(0, m)]


def chunkgen(inlist, num_chunks):
    """Yield a list of lists of length num_chunks, such that each element is a list containing a sequential chunk of the original list.
    
    ```python
    A = next(vipy.util.chunkgen(inlist, num_chunks=3))
    assert len(A) == len(inlist) // 3
    ```
    .. note::  The last chunk will be larger for ragged chunks
    """
    (m, n) = (num_chunks, int(np.ceil(float(len(inlist)) / float(num_chunks))))
    for i in range(0,m):
        yield inlist[i * n:min(i * n + n, len(inlist))]


def chunklistbysize(inlist, size_per_chunk):
    """Convert list into a list of lists such that each element is a list
    containing a sequential chunk of the original list of length
    size_per_chunk"""
    assert size_per_chunk >= 1
    return [inlist[i:i+size_per_chunk] for i in range(0,len(inlist),size_per_chunk)]

def chunkgenbysize(ingen, size_per_chunk):
    """Yield a list of lists such that each element is a list
    containing a sequential chunk of the original list of length
    size_per_chunk"""
    assert size_per_chunk >= 1

    if sys.version_info >= (3,12):
        for b in itertools.batched(ingen, size_per_chunk):
            yield b  
    else:        
        for i in range(0,len(ingen),size_per_chunk):
            yield ingen[i:i+size_per_chunk]
    
def triplets(inlist):
    """Yield triplets (1,2,3), (4,5,6), ...  from list inlist=[1,2,3,4,5,6,...]"""
    for k in range(0, len(inlist), 3):
        yield (inlist[k], inlist[k+1] if (k+1)<len(inlist) else None, inlist[k+2] if (k+2)<len(inlist) else None)
        
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


def imwritegray(img, imfile=None, quality=75):
    """Write a floating point grayscale numpy image in [0,1] as [0,255] grayscale"""
    if imfile is None:
        imfile = temppng()
    if isnumpy(img):
        if img.dtype == np.dtype('uint8'):
            # Assume that uint8 is in the range [0,255]
            PIL.Image.fromarray(img).save(os.path.expanduser(imfile), quality=quality)
        elif img.dtype == np.dtype('float32'):
            # Convert [0, 1.0] to uint8 [0,255]
            PIL.Image.fromarray(np.uint8(img * 255.0)).save(os.path.expanduser(imfile), quality=quality)
        else:
            raise ValueError('Unsupported datatype - '
                             'Numpy array must be uint8 or float32')
    else:
        raise ValueError('Input must be numpy array')
    return imfile


def imwrite(img, imfile=None, writeas=None, quality=75):
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
            PIL.Image.fromarray(rgb2bgr(img)).save(imfile, quality=quality)  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            # convert to uint8 then BGR
            PIL.Image.fromarray(rgb2bgr(np.uint8(255.0 * img))).save(imfile)
    elif writeas in ['bgr']:
        if img.ndim != 3:
            raise ValueError('numpy array must be 3D')
        if img.dtype == np.dtype('uint8'):
            PIL.Image.fromarray(img).save(imfile, quality=quality)  # convert to BGR
        elif img.dtype == np.dtype('float32'):
            # convert to uint8 then BGR
            PIL.Image.fromarray(np.uint8(255.0 * img)).save(imfile, quality=quality)
    else:
        raise ValueError('unsupported writeas')

    return imfile


def print_and_return(x):
    log.info(x)
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


def is_email_address(email):
    """Is the provided string an email address?"""
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(regex, email) is not None


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


def take(inlist, k):
    """Take k elements at random from inlist"""
    return [inlist[i] for i in np.random.permutation(range(len(inlist)))[0:k]] if len(inlist)>k else inlist

def takeone(inlist):
    """Take one element at random from inlist or return None if empty"""
    return take(list(inlist), k=1)[0] if len(inlist)>=1 else None   # -> random.sample()?

def takelast(inlist):
    """Take last element from inlist or return None if empty"""
    return tolist(inlist)[-1] if len(tolist(inlist))>=1 else None

def tryload(infile, abspath=False):
    """Attempt to load a pkl file, and return the value if successful and None if not"""
    try:
        return load(infile, abspath=abspath)
    except:
        return None

def canload(infile):
    """Attempt to load an archive file, and return true if it can be successfully loaded, otherwise False"""
    try:
        load(infile, abspath=True)
        return True
    except:
        return False




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
    

def scpsave(obj, username=None, format='json'):
    """Save an archive file to load via SCP.

    Use case:

    - This archive format is useful to allow access to videos and images that are accessible behind a remote server for which you have access via SSH key-based authentication.
    - You create this archive on the remote server, and all vipy objects are replaced with references to remote media.
    - Every video or image is replaced with a URL of the format 'scp://USER@HOST:/path/to.mp4'.  
    - Vipy will use your SSH keys to SCP these media files from USER@HOST on demand, so that the videos are cached for you on your local machine when you need them.
    - This is useful for transparently visualizing large datasets that are hidden behind an SSH-only accessible server

    Usage:
    
    ```python
    outfile = vipy.util.scpsave([vipy.video.Video(filename='/path/to.mp4)])  # run on remote machine that you have SSH key access
    V = vipy.util.scpload(outfile)  # run on local machine that has SSH key access to remote machine
    V[0].load()  # this will SCP the videos from 'scp:///path/to.mp4' to $VIPY_CACHE/to.mp4 transparently and on demand
    ```

    Args:
        V: [vipy objects] A list of vipy objects or `vipy.dataset.Dataset`
        username: [str] Your username on the remote machine to select the proper SSH key
        format: [str] pkl or json
    Returns:
        A temp archive file stored on the remote machine that will be downloaded and loaded via SCP, such that each element in the list will be fetched via scp when pixels are loaded.

    """
    
    import vipy.image
    import vipy.video

    try:
        # Connect to an external IP (doesn't send data, just determines route)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's DNS
        hostname = s.getsockname()[0]
        s.close()
    except:
        hostname = socket.gethostname()

    if (isinstance(obj, vipy.image.Image) or isinstance(obj, vipy.video.Video)) and obj.hasfilename():        
        obj = obj.clone().url('scp://%s%s:%s' % (('%s@' % username) if username is not None else '', hostname, obj.filename())).clear_filename()
    elif islist(obj) and all([isinstance(o, vipy.image.Image) or isinstance(o, vipy.video.Video) for o in obj]):
        obj = [o.clone().url('scp://%s%s:%s' % (('%s@' % username) if username is not None else '', hostname, o.abspath().filename())).clear_filename() for o in obj]
    else:
        raise ValueError('vipy objects only')

    archivefile = 'scp://%s%s:%s' % (('%s@' % username) if username is not None else '', hostname, save(obj, temppkl() if format == 'pkl' else tempjson()))
    cmd = "vipy.util.scpload('%s')" % archivefile
    log.info('[vipy.util.scpsave]: On a local machine where you have public key ssh access to this remote machine run:\n>>> %s' % cmd)
    return archivefile


def scpload(url):
    """Load an archive file saved using `vipy.util.scpsave`"""
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


def isextension(filename, ext):
    """Does the filename end with the extension ext? 
    
    ```python
    isextension('/path/to/myfile.json', 'json') == True
    isextension('/path/to/myfile.json', '.json') == True
    isextension('/path/to/myfile.json', '.pkl') == False
    ```

    """
    return filename is not None and filename.endswith(ext)

def ispkl(filename):
    """Is the file a pickle archive file"""
    return filename[-4:] == '.pkl' if isstring(filename) and len(filename) >= 4 else False

def ispklbz2(filename):
    """Is the file a pickle bz2 archive file"""
    return filename[-8:] == '.pkl.bz2' if isstring(filename) and len(filename) >= 8 else False

def is_pkl_gz(filename):
    """Is the file a pickle gzip archive file"""
    return filename[-7:] == '.pkl.gz' if isstring(filename) and len(filename) >= 7 else False

def ispklfile(filename):
    """Is the file a pickle archive file"""
    return ispkl(filename)

def ishtml(filename):
    """Is the file an HTMLfile"""
    return filename.lower()[-5:] == '.html'

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
    """Return c for filename /a/b/c.ext
    
    .. warning:: Will return /a/b/c.d for multidot filenames wth more than two trailing dots like /a/b/c.d.e.f (e.g. /a/b/my.filename.tar.gz)
    """
    (head, tail) = os.path.split(filename)
    (base, ext) = splitext(tail)
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
    """Alias for `vipy.util.newpath`"""
    return newpath(filename, newdir)


def filefull(f):
    """Return /a/b/c for filename /a/b/c.ext"""
    ext = fileext(f, multidot=True, withdot=True)
    return f.replace(ext, '') if ext is not None else f


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
    """return list of absolute paths to subdirectories in a directory"""
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
        list_of_rows = f.read().splitlines()
    return list_of_rows


def readtxt(infile):
    """Read a text file one string per row"""
    return readlist(infile)


def writecsv(list_of_tuples, outfile=None, mode='w', separator=',', header=None, comment='# '):
    """Write list of tuples to an output csv file with each list element on a row and tuple elements separated by commas.

    Examples:
    ```python
    vipy.util.writecsv([(1,2,3), (4,5,6)], '/tmp/out.csv')
    vipy.util.writecsv([(1,2,3), (4,5,6)], '/tmp/out.csv', separator=';'))
    vipy.util.writecsv([(1,2,3), (4,5,6)], '/tmp/out.csv', header=('h1','h2','h3'))
    ```

    Args:
        list_of_tuples: a list of tuples each tuple is a row
        outfile: the csv file output
        mode: 'w' for overwrite, 'a' for append
        separator: a string specifying the separator between columns.  defaults to ','
        header: a tuple containing strings to be appended to the first row of the csv file
        comment:  the comment symbol to be prepended to the header row 

    Returns:
        the outfile path
    """
    
    list_of_tuples = list_of_tuples if not isnumpy(list_of_tuples) else list_of_tuples.tolist()
    list_of_tuples = list_of_tuples if header is None else [tuple([h if k>0 else comment+h for (k,h) in enumerate(header)])]+list_of_tuples  # prepend header with comment symbol
    outfile = os.path.abspath(os.path.expanduser(outfile)) if outfile is not None else tempcsv()
    with open(outfile, mode) as f:
        for u in list_of_tuples:
            n = len(u)
            for (k, v) in enumerate(u):
                if (k + 1) < n:
                    f.write(str(v) + separator)
                else:
                    f.write(str(v) + '\n')
    return(outfile)


def readcsv(infile, separator=',', ignoreheader=False, comment=None, ignore_header=False):
    """Read a csv file into a list of lists, ignore any rows prepended with comment symbol, ignore first row if ignoreheader=True

    Args:
        infile: the csv file input
        separator: a string specifying the separator between columns.  defaults to ','
        ignoreheader: if true, ignore the first row of the csv file
        ignore_header: if true, ignore the first row of the csv file (argument synonym)
        comment:  if provided, ignore all rows with this comment symbol prepended

    Returns:
        a list of lists, each list element containing a list of elements in the corresponding line of the csv file, parsed by separator

    .. note:: this parser does not escape delimiters enclosed in double quotes, as may be assumed by some csv writers
    """

    with open(infile, 'r') as f:
        list_of_rows = [[x.strip() for x in r.split(separator)] for r in f]
    list_of_rows = list_of_rows if (len(list_of_rows)==0 or not (ignoreheader or ignore_header)) else list_of_rows[1:]
    list_of_rows = list_of_rows if comment is None else [r for r in list_of_rows if len(r)==0 or r[0][0] != comment]
    return list_of_rows


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
    """Is a path a URL?  It requires a url scheme and url netloc without any common unallowed characters"""
    try:
        url = urlparse(path)
        return not any([c in path for c in ('>','<','"')]) and bool(url.scheme) and bool(url.netloc)
    except:
        return False

def shortuuid(n=8):
    """Generate a short UUID with n charaters sampled uniformly at random from lowercase|uppercase|numbers"""
    return ''.join(random.sample(ALPHABET, n))

def stringhash(s, n=16):
    """Generate a repeatable hash with n characters for a string s"""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[0:n]

def isimageurl(path):
    """Is a path a URL with image extension?"""
    return path is not None and isurl(path) and isimg(path)


def isvideourl(path):
    """Is a path a URL with video extension?"""
    return isurl(path) and isvideo(path)


def isS3url(path):
    """Is a path a URL for an S3 object?"""
    return isurl(path) and urlparse(path).scheme == 's3'


def isyoutubeurl(path):
    """Is a path a youtube URL?"""
    return isurl(path) and ('youtube.com' in path or 'youtu.be' in path)

def isRTSPurl(path):
    return isurl(path) and path.startswith('rtsp://')

def isRTMPurl(path):
    return is_rtmp_url(path)

def is_rtmp_url(path):
    return isurl(path) and (path.startswith('rtmp://') or path.startswith('rtmps://'))


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


def totuple(x):
    """Convert an object to a python tuple?"""
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, tuple):        
        return x
    elif isinstance(x, set):        
        return tuple(x)
    else:
        return (x,)

def to_iterable(x): 
    """Convert an object to a singleton tuple if not already a list, tuple or set iterable"""
    return x if isinstance(x, (list, tuple, set)) else (x,)
        
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

def singletonlist(x):
    """Convert a singleton list to a singleton, otherwise return the list"""
    return x[0] if isinstance(x, list) and len(x)==1 else x

def toset(x):
    """Convert a python iterable to a set of not already a set"""
    if isinstance(x, set):        
        return x    
    elif isinstance(x, list) or isinstance(x, tuple):
        return set(x)
    else:
        return set([x])
    
    
def tolist_or_singleton(x):
    """Return list(x) if length of iterator x is not equal to one, else return x or None.  This is useful to return single elements instead of single element lists."""
    y = tolist(x)
    return y if len(y)>1 else (y[0] if len(y)==1 else None)


def isimg(path):
    """Is an object an image with a supported image extension ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?"""    
    if path is not None and os.path.splitext(path)[1].lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pgm', '.ppm', '.gif', '.bmp']:
        return True
    else:
        return False

def isimage(path):
    """Alias for `vipy.util.isimg`"""
    return isimg(path)
    
def isvideofile(path):
    """Alias for `vipy.util.isvideo`"""
    return isvideo(path)

def isimgfile(path):
    """Alias for `vipy.util.isimg`"""
    return isimg(path)

def has_image_extension(path):
    """Alias for `vipy.util.isimg`"""
    return isimg(path)

def isimagefile(path):
    """Alias for `vipy.util.isimg`"""
    return isimg(path)


def isjpeg(path):
    """is the file a .jpg or .jpeg extension?"""
    return hasextension(path) and fileext(path).lower() == '.jpg' or fileext(path).lower() == '.jpeg'

def iswebp(path):
    """is the file a .webp extension?"""
    return path is not None and hasextension(path) and fileext(path).lower() == '.webp'

def ispng(path):
    """is the file a .png or .apng extension?"""
    return hasextension(path) and (fileext(path).lower() == '.png' or fileext(path).lower() == '.apng')

def isgif(path):
    """is the file a .gif extension?"""
    return hasextension(path) and fileext(path).lower() == '.gif'

def isjpg(path):
    """Alias for `vipy.util.isjpeg`"""
    return isjpeg(path)


def iscsv(path):
    """Is a file a CSV file extension?"""

    (filename, ext) = (os.path.splitext(path) if path is not None else ('',''))
    if ext.lower() in ['.csv', '.CSV']:
        return True
    else:
        return False

def isvideo(path):
    """Is a filename in path a video with a known video extension ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm', '3gp']?"""
    if path is not None and os.path.splitext(path)[1].lower() in ['.avi','.mp4','.mov','.wmv','.mpg', '.mkv', '.webm', '.3gp']:
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
    (filename, ext) = (os.path.splitext(path) if path is not None else ('',''))
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
    """Convert a numpy array in BGR order to HSV"""
    return np.array(PIL.Image.fromarray(bgr2rgb(im_bgr)).convert('HSV'))  # BGR -> RGB -> HSV


def gray2hsv(im_gray):
    """Convert a numpy array in floating point single channel greyscale order to HSV"""
    return np.array(PIL.Image.fromarray(gray2rgb(im_gray)).convert('HSV'))  # Gray -> RGB -> HSV


def isarchive(filename):
    """Is filename a zip or gzip compressed tar archive?"""
    (filebase, ext) = splitext(filename)
    if (ext is not None) and (len(ext) > 0) and (ext.lower() in [
            '.egg', '.jar', '.tar', '.tar.bz2', '.tar.gz',
            '.tgz', '.tz2', '.zip', '.gz']):
        return True
    else:
        (filebase, ext) = splitext(ext[1:])
        if (ext is not None) and (len(ext) > 0) and (ext.lower() in ['.bz2']):
            return True
        else:
            return False

def istgz(filename):
    """Is the filename a .tgz or .tar.gz extension?"""
    return filename[-4:] == '.tgz' or filename[-7:] == '.tar.gz'

def istar(filename):
    """Is the filename a .tar extension?"""
    return filename[-4:] == '.tar'

def istarbz2(filename):
    """Is the filename a .bz2 or .tar.bz2 extension?"""
    return filename[-8:] == '.tar.bz2'

def tempfilename(suffix=''):
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

def tempWEBP():
    """Create a temporary WEBP file in system temp directory"""
    return tempfilename(suffix='.webp')


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

def temppklbz2():
    """Create a temporary .pkl.bz2 file"""
    return temppickle()+'.bz2'


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
    
    ```python 
       t = Timer():
       [some code]
       print(t)
       [some more code]
       print(t)

       with Timer():
          [some code]
    ```
       
    """
    def __enter__(self):
        self._begin = time.time()
        self._last = self._begin
        return self
        
    def __exit__(self, *args):
        log.info(self.__repr__())

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
    if os.path.isdir(path) is False and len(str(path)) > 0:
        os.makedirs(path)
    elif flush is True:
        shutil.rmtree(path)
        os.makedirs(path)
    return os.path.abspath(os.path.expanduser(path))


def rermdir(path):
    """Recursively delete a given directory (if exists), and remake it"""
    return remkdir(path, flush=True)


def premkdir(filename):
    """pre-create directory /path/to/subdir using `vipy.util.remkdir` if it does not exist for outfile=/path/to/subdir/file.ext, and return filename"""
    remkdir(filepath(filename))
    return filename


def newbase(filename, base):
    """Convert filename=/a/b/c.ext base=d -> /a/b/d.ext"""
    return os.path.join(filepath(filename), '%s.%s' % (base, fileext(filename, withdot=False)))

def toextension(filename, newext):
    """Convert filename='/path/to/myfile.ext' to /path/to/myfile.xyz, such that newext='xyz' or newext='.xyz'"""
    if '.' in newext:
        newext = newext.split('.')[-1]
    (filename, oldext) = splitext(filename)
    return filename + '.' + str(newext)

def noextension(filename, ext=None):
    """Convert filename='/path/to/myfile.ext' or filename='/path/to/myfile.ext1.ext2.ext3' to /path/to/myfile with no extension, removing the appended string past the first dot"""
    return filename.split('.')[0] if ext is None else filename.replace(ext, '')

def topkl(filename):
    """Convert filename='/path/to/myfile.ext' to /path/to/myfile.pkl"""
    return toextension(filename, '.pkl')

def splitext(filename):
    """Given /a/b/c.ext return tuple of strings ('/a/b/c', '.ext'), handling multi-dot extensions like .tar.gz"""
    (head, tail) = os.path.split(filename)
    ext = fileext(filename, multidot=True, withdot=True)
    base = tail.replace(ext,'') if ext is not None else tail
    return (os.path.join(head, base), ext)  # for consistency with splitext


def hasextension(filename):
    """Does the provided filename have a file extension (e.g. /path/to/file.ext) or not (e.g. /path/to/file)"""
    return fileext(filename) is not None


def fileext(filename, multidot=True, withdot=True):
    """Given filename /a/b/c.ext return '.ext', or /a/b/c.tar.gz return '.tar.gz'.   If multidot=False, then return '.gz'.  If withdot=False, return 'ext'.  Multidot support at most two trailing dots"""
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

def symlink(src, dst, overwrite=False):
    """Create a symlink from src to dst, overwriting the existing symlink at dst if overwrite=True"""
    if overwrite and os.path.islink(dst):
        os.unlink(dst)
    os.symlink(src, dst)
    return dst


def truncate_string(s, maxlen):
    """If string s is greater than maxlen, truncate and append an ellipsis"""
    return s if len(s) <= maxlen else str(s)[0:maxlen]+'...'

def escape_string_for_innerHTML(s, escape=(('\n','<br>'),('{',"&#123;"),('}','&#125;'),('"', '&quot;'),("'","&#39;"))):
    """Convert a string by replacing escape characters with equivalents suitable for copying into an innerHTML element in html.  

    The escaping characters are provided as ((character, replacemant), ...) 

    Given an html file with the format:
    
    <html><body><pre>INNER_HTML</pre></body></html>
    
    This function converts a string s to an escaped_string such that INNER_HTML replaced with the escaped string will render properly as the string s in-browser.

    This is useful for `vipy.visualize.scene_explorer` to escape json prior to copying into the template
    
    This is pretty hacky, there has got to be a better way ...
    """
    for (c,e) in escape:
        s = s.replace(c,e)
    return s
    
