"""Object cache for URLs with garbage collection"""

import os
from os import path
import hashlib
import numpy
import urllib.parse
from strpy.bobo.util import isarchive, isurl, isimg, ishdf5, isfile, quietprint, isnumpy, isstring, remkdir, splitextension, filepath
import strpy.bobo.viset.download
import string
import shutil
import h5py


class Cache():
    _maxsize = None
    _verbose = None
    _strategy = None
    _free_maxctr = 100
    _free_ctr = _free_maxctr
    _cachesize = None 
    _prettyhash = True
    
    def __init__(self, maxsize=10E9, verbose=True, strategy='lru', refetch=False):
        self._maxsize = maxsize
        self._verbose = verbose
        self._strategy = strategy
        self._refetch = refetch
        #quietprint('[bobo.cache]: initializing cache with root directory "%s"' % self.root(), verbose)
        
    def __len__(self):
        if self._cachesize is not None:
            return self._cachesize.get()
        else:
            return self.size()
        
    def __repr__(self):
        return str('<bobo.cache: cachedir=' + str(self.root()) + '\'>')
    
    def __getitem__(self, key):
        return self.get(key)

    def _getroot(self):
        if os.environ.get('JANUS_DATA') is not None:
            cachedir = (os.environ.get('JANUS_DATA'))
        elif os.environ.get('JANUS_CACHE') is not None:
            cachedir = (os.environ.get('JANUS_CACHE'))
        elif os.environ.get('BOBO_CACHE') is not None:
            cachedir = (os.environ.get('BOBO_CACHE'))
        else:
            cachedir = (path.join(os.environ['HOME'],'.janus'))
        return remkdir(cachedir)
        
    def _hash(self, url):
        """Compute a SHA1 hash of a url to generate a unique cache filename for a given url"""
        p = urllib.parse.urlsplit(url)
        urlquery = urllib.parse.urlunsplit([p[0],p[1],p[2],p[3],None])
        urlpath = urllib.parse.urlunsplit([p[0],p[1],p[2],None,None])        
        (filename, ext) = splitextension(urlpath)
        #urlopt = self._url_fragment_options(url)
        urlhash = hashlib.sha1(urlquery).hexdigest()
        if self._prettyhash:    
            return path.basename(filename) + '_' + urlhash[0:7]
        else:
            return urlhash 

    def _free(self):
        """FIXME: Garbage collection"""
        if self._free_ctr == 0:
            if self._cachesize is not None:
                if self._cachesize.get() > self._maxsize:
                    print('[bobo.cache][WARNING]: cachesize is larger than maximum.  Clean resources!')
            quietprint('[bobo.cache]: spawning cache garbage collection process', verbosity=2)
            self._cachesize = Pool(1).apply_async(self.size(), self.root())
            self._free_ctr = self._free_maxctr
        self._free_ctr -= 1


    # ---- PUBLIC FUNCTIONS --------------------------------------------------------
    def put(self, obj, key=None, timeout=None, sha1=None):
        """Put an native python object into cache with the provided cache key"""
        if key is None:
            key = self.key(obj)        
        if self.iscached(key):
            raise CacheError('[bobo.cache][Error]: Key collision! Existing object in cache with key "%s"' % key)
            
        # Numpy object - export to file in cache with provided key
        if isnumpy(obj):
            quietprint('[bobo.cache][PUT]: Exporting numpy object to cache with key "' + key + '"', verbosity=3)                                                                             
            f = h5py.File(self.abspath(key), 'a')
            f[key] = obj
            f.close()

        # URL - download and save to cache with provided key
        elif isurl(obj):
            quietprint('[bobo.cache][PUT]: "%s" key "%s"' % (obj, key), verbosity=2)                                                                                             
            filename = self._download(obj, timeout=timeout)
            shutil.move(filename, self.abspath(key))
            
        # Unsupported type!
        else:
            raise CacheError('[bobo.cache][ERROR]: Unsupported object type for PUT')
            
        # Return cache key
        return key        

    # ------------------------------------------------------------    
    def get(self, url, key=None, timeout=None, sha1=None, username=None, password=None):
        """Download url and store downloaded file to provided key in cache, return absolute filename"""

        # File already exists?
        if os.path.isfile(url):
            return url
                
        # Key provided?
        if key is None:
            key = self.key(urllib.parse.urldefrag(url)[0])
        else:
            remkdir(filepath(self.abspath(key)))        
        if self.iscached(key):
            quietprint('[bobo.cache][HIT]:  "%s" to "%s" ' % (url, self.abspath(key)), verbosity=3)                
            return self.abspath(key)
        
        # Download me!
        quietprint('[bobo.cache][MISS]: "%s" key "%s" ' % (url, key), verbosity=3)                        
        filename = self.abspath(key)        
        url_scheme = urllib.parse.urlparse(url)[0]
        if url_scheme in ['http', 'https']:
            quietprint('[bobo.cache][MISS]: downloading "%s" to "%s"' % (url, filename), verbosity=2)                                            
            bobo.viset.download.download(url, filename, verbose=False, timeout=timeout, sha1=None, username=username, password=password)                       
        elif url_scheme == 'file':
            shutil.copyfile(url, filename)
        elif url_scheme == 'hdfs':
            raise NotImplementedError('FIXME: support for hadoop distributed file system')                
        else:
            raise NotImplementedError('file not found - %s' % url)

        # Verify me
        if sha1 is not None:
            quietprint('[bobo.cache]: Verifying SHA1... ', verbosity=3)                          
            if not bobo.viset.download.verify_sha1(filename, sha1):
                raise CacheError('[bobo.cache][ERROR]: invalid SHA1 ')  

        #self._free()  # garbage collection time?

        # Return cache key
        return self.abspath(key)


    # ------------------------------------------------------------    
    def discard(self, key):
        """Delete single key from cache"""
        if self.iscached(key):
            quietprint('[bobo.cache]: Removing key "%s" ' % key, verbosity=1)
            if os.path.isfile(self.abspath(key)):
                os.remove(self.abspath(key))
        elif os.path.isdir(self.abspath(key)):
            quietprint('[bobo.cache]: Removing cached directory "%s" ' % (self.abspath(key)), verbosity=1)
            shutil.rmtree(self.abspath(key))
        else:
            raise CacheError('[bobo.cache][ERROR]: Key not found "%s" ' % (key))

            
    # ------------------------------------------------------------                
    def delete(self):
        """Delete entire cache"""
        quietprint('[bobo.cache]: Deleting all cached data in "' + self.root() + '"', verbosity=1)
        shutil.rmtree(self.root())
        os.makedirs(self.root())        

    # ------------------------------------------------------------            
    def clean(self):
        """Delete entire cache"""
        self.delete()

    # ------------------------------------------------------------            
    def size(self, key=None):
        """Recursively compute the size in bytes of a cache directory: http://snipplr.com/view/47686/"""
        if key is None:
            total_size = os.path.getsize(self.root())
            for item in os.listdir(self.root()):
                itempath = os.path.join(self.root(), item)
                if os.path.isfile(itempath):
                    total_size += os.path.getsize(itempath)
                elif os.path.isdir(itempath):
                    total_size += self.size(itempath)
            return total_size
        else:
            # return size of key only
            if os.path.isfile(self.abspath(self.key(key))):            
                return os.path.getsize(self.abspath(self.key(key)))
            else:
                return 0
            
    # ------------------------------------------------------------    
    def iscached(self, key):
        """Return true if a key is in the cache"""
        return path.isfile(self.abspath(key)) or path.isdir(self.abspath(key))

    # ------------------------------------------------------------    
    def key(self, obj):        
        """Return a unique cache key (relative path to file in cache) for an object"""
        if isnumpy(obj):
            # Key is byte view sha1 hash with .h5 extension in cache subdir of cacheroot
            byteview = obj.view(numpy.uint8)
            key = os.path.join('cache', str(hashlib.sha1(byteview).hexdigest()) + '.h5')
        elif isurl(obj):
            # key is URL filename with an appended hash (for uniqueness)
            p = urllib.parse.urlsplit(obj)
            urlquery = urllib.parse.urlunsplit([p[0],p[1],p[2],p[3],None])        
            urlpath = urllib.parse.urlunsplit([p[0],p[1],p[2],None,None])
            urlhash = self._hash(obj)
            (filename, ext) = splitextension(path.basename(urlpath))
            key = str(urlhash) + str(ext)
        elif os.path.isfile(obj):
            key = obj  # just use absolute path as key 
            
            # within cache?
            #filebase = obj.split(self.root(),1)
            #if len(filebase) == 2:
            #    # key is subpath within cache
            #    key = filebase[1][1:]
            #else:
            #    # key is filename with unique appended hash
            #    (head, tail) = os.path.split(obj)
            #    (filename, ext) = splitextension(tail)                 
            #    namehash = hashlib.sha1(tail).hexdigest()                 
            #    key = filename + '_' + str(namehash[0:7]) + ext
            
        elif (path.isfile(self.abspath(obj)) or path.isdir(self.abspath(obj))):
            key = obj   # Already a cache key
        elif isstring(obj):
            key = obj   # Use arbitrary string if not file or url
        else:
            raise CacheError('[bobo.cache][ERROR]: Unsupported object for constructing key')
        return key

    # ------------------------------------------------------------                
    def abspath(self, key):
        """The absolute file path for a cache key"""
        return os.path.join(self.root(), key)

    # ------------------------------------------------------------        
    def root(self):
        return self._getroot()

    # ------------------------------------------------------------            
    def ls(self):
        print(os.listdir(self.root()))

    # ------------------------------------------------------------            
    def unpack(self, pkgkey, unpackto=None, sha1=None, cleanup=False):
        """Extract archive file to unpackdir directory, delete archive file and return archive directory"""
        if not self.iscached(pkgkey):
            raise CacheError('[bobo.cache][ERROR]: Key not found "%s" ' % pkgkey)
        filename = self.abspath(pkgkey)
        if isarchive(filename):
            # unpack directory is the same directory as filename
            if unpackto is None:
                unpackdir = self.root()
            else:
                unpackdir = self.abspath(unpackto)
            if not path.exists(unpackdir):
                os.makedirs(unpackdir)
            bobo.viset.download.extract(filename, unpackdir, sha1=sha1, verbose=self._verbose)                
            if cleanup:
                quietprint('[bobo.cache]: Deleting archive "%s" ' % (pkgkey), verbosity=2)                                
                os.remove(filename)
            return unpackdir
        else:
            raise CacheError('[bobo.cache][ERROR]: Key not archive "%s" ' % pkgkey)            

class CacheError(Exception):
    pass

class DownloadError(Exception):
    pass


def sha1(filename):
    """Generate SHA1 hash for filename or URL"""
    if isurl(filename):
        filename = Cache().get(filename)
    sha1 = hashlib.sha1()
    f = open(filename, 'rb')
    try:
        sha1.update(f.read())
    finally:
        f.close()
    return sha1.hexdigest()

        
    
