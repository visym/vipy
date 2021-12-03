import os
import numpy as np
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv, newpathroot, listjson, extlist, filefull, tempdir, groupbyasdict
import random
import vipy
import vipy.util
import shutil
import uuid
import warnings
import copy 
from vipy.util import is_email_address
import hashlib
import pickle
import time
import json
import dill
from vipy.show import colorlist
import matplotlib.pyplot as plt
import gc 
import vipy.metrics


class Dataset():
    """vipy.dataset.Dataset() class
    
    Common class to manipulate large sets of vipy objects in parallel

    ```python
    D = vipy.dataset.Dataset([vipy.video.RandomScene(), vipy.video.RandomScene()], id='random_scene')
    with vipy.globals.parallel(2):
        D = D.map(lambda v: v.frame(0))
    list(D)
    ```

    Create dataset and export as a directory of json files 

    ```python
    D = vipy.dataset.Dataset([vipy.video.RandomScene(), vipy.video.RandomScene()])
    D.tojsondir('/tmp/myjsondir')
    ```
    
    Create dataset from all json or pkl files recursively discovered in a directory and lazy loaded

    ```python
    D = vipy.dataset.Dataset('/tmp/myjsondir')  # lazy loading
    ```

    Create dataset from a list of json or pkl files and lazy loaded

    ```python
    D = vipy.dataset.Dataset(['/path/to/file1.json', '/path/to/file2.json'])  # lazy loading
    ```
    
    .. notes:: Be warned that using the jsondir constructor will load elements on demand, but there are some methods that require loading the entire dataset into memory, and will happily try to do so
    """

    def __init__(self, objlist, id=None, abspath=True):

        self._saveas_ext = ['pkl', 'json']
        self._id = id if id is not None else vipy.util.shortuuid(8)
        self._loader = lambda x: x
        self._istype_strict = True
        self._lazy_loader = False
        self._abspath = abspath

        if isinstance(objlist, str) and (vipy.util.isjsonfile(objlist) or vipy.util.ispklfile(objlist)):
            self._objlist = vipy.util.load(objlist, abspath=abspath)
        elif isinstance(objlist, str) and os.path.isdir(objlist):
            self._objlist = vipy.util.findjson(objlist) + vipy.util.findpkl(objlist)  # recursive
            self._loader = lambda x,b=abspath:  vipy.util.load(x, abspath=b) if (vipy.util.ispkl(x) or vipy.util.isjsonfile(x)) else x
            self._istype_strict = False
            self._lazy_loader = True
        elif isinstance(objlist, list) and all([(vipy.util.ispkl(x) or vipy.util.isjsonfile(x)) for x in objlist]):
            self._objlist = objlist 
            self._loader = lambda x,b=abspath:  vipy.util.load(x, abspath=b) if (vipy.util.ispkl(x) or vipy.util.isjsonfile(x)) else x            
            self._istype_strict = False
            self._lazy_loader = True
        else:
            self._objlist = objlist

        self._objlist = tolist(self._objlist)        
        assert len(self._objlist) > 0, "Empty dataset"

        if self._lazy_loader:
            try:
                self[0]
            except Exception as e:
                raise ValueError('Invalid dataset - Lazy load failed with error "%s"' % str(e))

    def __repr__(self):
        return str('<vipy.dataset: id="%s", len=%d, type=%s>' % (self.id(), len(self), str(type(self[0])) if len(self)>0 else 'None'))

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __getitem__(self, k):
        if isinstance(k, int) or isinstance(k, np.uint64):
            assert k>=0 and k<len(self._objlist), "invalid index"
            return self._loader(self._objlist[int(k)])
        elif isinstance(k, slice):
            return [self._loader(x) for x in self._objlist[k.start:k.stop:k.step]]
        else:
            raise ValueError()
            
    def __len__(self):
        return len(self._objlist)
        
    def json(self, encode=True):
        r = vipy.util.class_registry()
        d = {k:v for (k,v) in self.__dict__.items() if not k == '_loader'}
        d['_objlist'] = [(str(type(v)), v.json(encode=False)) if str(type(v)) in r else v for v in self._objlist]
        return json.dumps(d) if encode else d
        
    @classmethod
    def from_json(cls, s):
        r = vipy.util.class_registry()
        d = json.loads(s) if not isinstance(s, dict) else s  
        return cls(objlist=[r[x[0]](x[1]) if (isinstance(x, tuple) and x[0] in r) else x for x in d['_objlist']],
                   id=d['_id'],
                   abspath=d['_abspath'])                            

    def id(self, n=None):
        """Set or return the dataset id"""
        if n is None:
            return self._id
        else:
            self._id = n
            return self

    def list(self):
        """Return the dataset as a list"""
        return list(self)
    def tolist(self):
        """Alias for self.list()"""
        return list(self)

    def flatten(self):
        """Convert dataset stored as a list of lists into a flat list"""
        self._objlist = [o for objlist in self._objlist for o in vipy.util.tolist(objlist)]
        return self

    def istype(self, validtype):
        """Return True if the all elements (or just the first element if strict=False) in the dataset are of type 'validtype'"""
        return all([any([isinstance(v,t) for t in tolist(validtype)]) for v in self]) if self._istype_strict else any([isinstance(self[0],t) for t in tolist(validtype)])

    def _isvipy(self):
        """Return True if all elements in the dataset are of type `vipy.video.Video` or `vipy.image.Image`"""        
        return self.istype([vipy.image.Image, vipy.video.Video])

    def _is_vipy_video(self):
        """Return True if all elements in the dataset are of type `vipy.video.Video`"""                
        return self.istype([vipy.video.Video])

    def _is_vipy_scene(self):
        """Return True if all elements in the dataset are of type `vipy.video.Scene`"""                        
        return self.istype([vipy.video.Scene])

    def clone(self):
        """Return a deep copy of the dataset"""
        return copy.deepcopy(self)

    def archive(self, tarfile, delprefix, mediadir='', format='json', castas=vipy.video.Scene, verbose=False, extrafiles=None, novideos=False, md5=True):
        """Create a archive file for this dataset.  This will be archived as:

           /path/to/tarfile.{tar.gz|.tgz|.bz2}
              tarfilename
                 tarfilename.{json|pkl}
                 mediadir/
                     video.mp4
                 extras1.ext
                 extras2.ext
        
            Args:
                tarfile: /path/to/tarfilename.tar.gz
                delprefix:  the absolute file path contained in the media filenames to be removed.  If a video has a delprefix='/a/b' then videos with path /a/b/c/d.mp4' -> 'c/d.mp4', and {JSON|PKL} will be saved with relative paths to mediadir.  This may be a list of delprefixes.
                mediadir:  the subdirectory name of the media to be contained in the archive.  Usually "videos".             
                extrafiles: list of tuples or singletons [(abspath, filename_in_archive_relative_to_root), 'file_in_root_and_in_pwd', ...], 
                novideos [bool]:  generate a tarball without linking videos, just annotations
                md5 [bool]:  If True, generate the MD5 hash of the tarball using the system "md5sum", or if md5='vipy' use a slower python only md5 hash 
                castas [class]:  This should be a vipy class that the vipy objects should be cast to prior to archive.  This is useful for converting priveledged superclasses to a base class prior to export.

            Example:  

              - Input files contain /path/to/oldvideos/category/video.mp4
              - Output will contain relative paths videos/category/video.mp4

        ```python
        d.archive('out.tar.gz', delprefix='/path/to/oldvideos', mediadir='videos')
        ```
        
            Returns:

                The absolute path to the tarball 
        """
        assert self._isvipy(), "Source dataset must contain vipy objects for staging"
        assert all([os.path.isabs(v.filename()) for v in self]), "Input dataset must have only absolute media paths"
        assert len([v for v in self if any([d in v.filename() for d in tolist(delprefix)])]) == len(self), "all media objects must have a provided delprefix for relative path construction"
        assert vipy.util.istgz(tarfile) or vipy.util.isbz2(tarfile), "Allowable extensions are .tar.gz, .tgz or .bz2"
        assert shutil.which('tar') is not None, "tar not found on path"        
        
        D = self.clone()
        stagedir = remkdir(os.path.join(tempdir(), filefull(filetail(tarfile))))
        print('[vipy.dataset]: creating staging directory "%s"' % stagedir)
        delprefix = [[d for d in tolist(delprefix) if d in v.filename()][0] for v in self]  # select the delprefix per video
        D._objlist = [v.filename(v.filename().replace(os.path.normpath(p), os.path.normpath(os.path.join(stagedir, mediadir))), symlink=not novideos) for (p,v) in zip(delprefix, D.list())]
        pklfile = os.path.join(stagedir, '%s.%s' % (filetail(filefull(tarfile)), format))
        D.save(pklfile, relpath=True, nourl=True, sanitize=True, castas=castas, significant_digits=2, noemail=True, flush=True)
    
        # Copy extras (symlinked) to staging directory
        if extrafiles is not None:
            assert all([((isinstance(e, tuple) or isinstance(e, list)) and len(e) == 2) or isinstance(e, str) for e in extrafiles])
            extrafiles = [e if (isinstance(e, tuple) or isinstance(e, list)) else (e,e) for e in extrafiles]  # tuple-ify files in pwd() and should be put in the tarball root
            for (e, a) in tolist(extrafiles):
                assert os.path.exists(os.path.abspath(e)), "Invalid extras file '%s' - file not found" % e
                os.symlink(os.path.abspath(e), os.path.join(stagedir, filetail(e) if a is None else a))

        # System command to run tar
        cmd = ('tar %scvf %s -C %s --dereference %s %s' % ('j' if vipy.util.isbz2(tarfile) else 'z', 
                                                           tarfile,
                                                           filepath(stagedir),
                                                           filetail(stagedir),
                                                           ' > /dev/null' if not verbose else ''))

        print('[vipy.dataset]: executing "%s"' % cmd)        
        os.system(cmd)  # too slow to use python "tarfile" package
        print('[vipy.dataset]: deleting staging directory "%s"' % stagedir)        
        shutil.rmtree(stagedir)

        if md5:
            if shutil.which('md5sum') is not None:
                cmd = 'md5sum %s' % tarfile
                print('[vipy.dataset]: executing "%s"' % cmd)        
                os.system(cmd)  # too slow to use python "vipy.downloader.generate_md5(tarball)" for huge datasets
            else:
                print('[vipy.dataset]: %s, MD5=%s' % (tarfile, vipy.downloader.generate_md5(tarfile)))  # too slow for large datasets, but does not require md5sum on path
        return tarfile
        
    def save(self, outfile, nourl=False, castas=None, relpath=False, sanitize=True, strict=True, significant_digits=2, noemail=True, flush=True):
        """Save the dataset to the provided output filename stored as pkl or json
        
        Args:
            outfile: [str]: The /path/to/out.pkl or /path/to/out.json
            nourl: [bool]: If true, remove all URLs from the media (if present)
            castas: [type]:  Cast all media to the provided type.  This is useful for downcasting to `vipy.video.Scene` from superclasses
            relpath: [bool]: If true, define all file paths in objects relative to the /path/to in /path/to/out.json
            sanitize: [bool]:  If trye, call sanitize() on all objects to remove all private attributes with prepended '__' 
            strict: [bool]: Unused
            significant_digits: [int]: Assign the requested number of significant digits to all bounding boxes in all tracks.  This requires dataset of `vipy.video.Scene`
            noemail: [bool]: If true, scrub the attributes for emails and replace with a hash
            flush: [bool]:  If true, flush the object buffers prior to save

        Returns:        
            This dataset that is quivalent to vipy.dataset.Dataset('/path/to/outfile.json')
        """
        n = len([v for v in self if v is None])
        if n > 0:
            print('[vipy.dataset]: removing %d invalid elements' % n)
        objlist = [v for v in self if v is not None]  
        if relpath or nourl or sanitize or flush or noemail or (significant_digits is not None):
            assert self._isvipy(), "Invalid input"
        if relpath:
            print('[vipy.dataset]: setting relative paths')
            objlist = [v.relpath(filepath(outfile)) if os.path.isabs(v.filename()) else v for v in objlist]
        if nourl: 
            print('[vipy.dataset]: removing URLs')
            objlist = [v.nourl() for v in objlist]           
        if sanitize:
            print('[vipy.dataset]: sanitizing attributes')                        
            objlist = [v.sanitize() for v in objlist]  # removes all attributes with '__' keys
        if castas is not None:
            assert hasattr(castas, 'cast'), "Invalid cast"
            print('[vipy.dataset]: casting as "%s"' % (str(castas)))
            objlist = [castas.cast(v) for v in objlist]                     
        if significant_digits is not None:
            assert self._is_vipy_scene()
            assert isinstance(significant_digits, int) and significant_digits >= 1, "Invalid input"
            objlist = [o.trackmap(lambda t: t.significant_digits(significant_digits)) if o is not None else o for o in objlist]
        if noemail:
            print('[vipy.dataset]: removing emails')            
            for o in objlist:
                for (k,v) in o.attributes.items():
                    if isinstance(v, str) and is_email_address(v):
                        o.attributes[k] = hashlib.sha1(v.encode("UTF-8")).hexdigest()[0:10]
        if flush:
            objlist = [o.flush() for o in objlist]  

        print('[vipy.dataset]: Saving %s to "%s"' % (str(self), outfile))
        vipy.util.save(objlist, outfile)
        return self

    def classlist(self):
        """Return a sorted list of categories in the dataset"""
        assert self._isvipy(), "Invalid input"
        return sorted(list(set(self.map(lambda v: v.category()))))

    def classes(self):
        """Alias for classlist"""
        return self.classlist()
    def categories(self):
        """Alias for classlist"""
        return self.classlist()
    def num_classes(self):
        """Return the number of unique categories in this dataset"""
        return len(self.classlist())
    def num_labels(self):
        """Alias for num_classes"""
        return self.num_classes()
    def num_categories(self):
        """Alias for num_classes"""
        return self.num_classes()
    
    
    def class_to_index(self):
        """Return a dictionary mapping the unique classes to an integer index.  This is useful for defining a softmax index ordering for categorization"""
        return {v:k for (k,v) in enumerate(self.classlist())}

    def index_to_class(self):
        """Return a dictionary mapping an integer index to the unique class names.  This is the inverse of class_to_index, swapping keys and values"""
        return {v:k for (k,v) in self.class_to_index().items()}

    def label_to_index(self):
        """Alias for class_to_index"""
        return self.class_to_index()

    def powerset(self):
        return list(sorted(set([tuple(sorted(list(a))) for v in self for a in v.activitylabel() if len(a) > 0])))        

    def powerset_to_index(self):        
        assert self._isvipy(), "Invalid input"
        return {c:k for (k,c) in enumerate(self.powerset())}

    def dedupe(self, key):
        self._objlist = list({key(v):v for v in self}.values())
        return self
        
    def countby(self, f):
        return len([v for v in self if f(v)])

    def union(self, other, key=None):
        assert isinstance(other, Dataset), "invalid input"
        if len(other) > 0:
            try:
                other._loader(self._objlist[0])
                self._loader(other._objlist[0])
                self._objlist = self._objlist + other._objlist  # compatible loaders
            except:
                self._objlist = self.list() + other.list()  # incompatible loaders
                self._loader = lambda x: x
        return self.dedupe(key) if key is not None else self
    
    def difference(self, other, key):
        assert isinstance(other, Dataset), "invalid input"
        idset = set([key(v) for v in self]).difference([key(v) for v in other])   # in A but not in B
        self._objlist = [v for v in self if key(v) in idset]
        return self
        
    def has(self, val, key):
        return any([key(obj) == val for obj in self])

    def replace(self, other, key):
        """Replace elements in self with other with equality detemrined by the key lambda function"""
        assert isinstance(other, Dataset), "invalid input"
        d = {key(v):v for v in other}
        self._objlist = [v if key(v) not in d else d[key(v)] for v in self]
        return self

    def merge(self, outdir):
        """Merge a dataset union into a single subdirectory with symlinked media ready to be archived.

        ```python
        D1 = vipy.dataset.Dataset('/path1/dataset.json')
        D2 = vipy.dataset.Dataset('/path2/dataset.json')
        D3 = D1.union(D2).merge(outdir='/path3')
        ```

        Media in D1 are in /path1, media in D2 are in /path2, media in D3 are all symlinked to /path3.
        We can now create a tarball for D3 with all of the media files in the same relative path.
        """
        
        outdir = vipy.util.remkdir(os.path.abspath(os.path.normpath(outdir)))
        return self.clone().localmap(lambda v: v.filename(os.path.join(outdir, filetail(v.filename())), copy=False, symlink=True))

    def augment(self, f, n_augmentations):
        assert n_augmentations >= 1
        self._objlist = [f(v.clone()) for v in self for k in range(n_augmentations)]  # This will remove the originals
        return self

    def filter(self, f):
        """In place filter with lambda function f"""
        self._objlist = [v for v in self if f(v)]
        return self

    def valid(self):
        return self.filter(lambda v: v is not None)

    def takefilter(self, f, n=1):
        """Apply the lambda function f and return n elements in a list where the filter returns true
        
        Args:
            f: [lambda] If f(x) returns true, then keep
            n: [int >= 0] The number of elements to take
        
        Returns:
            [n=0] Returns empty list
            [n=1] Returns singleton element
            [n>1] Returns list of elements of at most n such that each element f(x) is True            
        """
        objlist = [obj for obj in self if f(obj)]
        return [] if (len(objlist) == 0 or n == 0) else (objlist[0] if n==1 else objlist[0:n])

    def jsondir(self, outdir=None, verbose=True, rekey=False, bycategory=False, byfilename=False, abspath=True):
        """Export all objects to a directory of JSON files.
    
           Usage:

        ```python
        D = vipy.dataset.Dataset(...).jsondir('/path/to/jsondir')
        D = vipy.util.load('/path/to/jsondir')   # recursively discover and lazy load all json files 
        ```

           Args:

               outdir [str]:  The root directory to store the JSON files
               verbose [bool]: If True, print the save progress
               rekey [bool] If False, use the instance ID of the vipy object as the filename for the JSON file, otherwise assign a new UUID_dataset-index
               bycategory [bool]: If True, use the JSON structure '$OUTDIR/$CATEGORY/$INSTANCEID.json'
               byfilename [bool]: If True, use the JSON structure '$FILENAME.json' where $FILENAME is the underlying media filename of the vipy object
               abspath [bool]: If true, store absolute paths to media in JSON.  If false, store relative paths to media from JSON directory

           Returns:
               outdir: The directory containing the JSON files.
        """
        assert self._isvipy()
        assert outdir is not None or byfilename 
        assert not byfilename and bycategory

        if outdir is not None:
            vipy.util.remkdir(outdir) 
        if bycategory:
            tojsonfile = lambda v,k: os.path.join(outdir, v.category(), ('%s.json' % v.instanceid()) if not rekey else ('%s_%d.json' % (uuid.uuid4().hex, k)))
        elif byfilename:
            tojsonfile = lambda v,k: vipy.util.toextension(v.filename(), '.json')
        else:
            tojsonfile = lambda v,k: os.path.join(outdir, ('%s.json' % v.instanceid()) if not rekey else '%s_%d.json' % (uuid.uuid4().hex, k))
        
        for (k,v) in enumerate(self):            
            f = vipy.util.save(v.clone().relpath(start=filepath(tojsonfile(v,k))) if not abspath else v.clone().abspath(), tojsonfile(v,k))
            if verbose:
                print('[vipy.dataset.Dataset][%d/%d]: %s' % (k, len(self), f))
        return outdir

    def tojsondir(self, outdir=None, verbose=True, rekey=False, bycategory=False, byfilename=False, abspath=True):
        """Alias for `vipy.dataset.Dataset.jsondir`"""
        return self.jsondir(outdir, verbose=verbose, rekey=rekey, bycategory=bycategory, byfilename=byfilename, abspath=abspath)
    
    def takelist(self, n, category=None, canload=False):
        """Take n elements of selected category and return list"""
        assert n >= 0, "Invalid length"
        D = self if category is None else self.clone().filter(lambda v: v.category() == category())
        return [D[int(k)] for k in np.random.permutation(range(len(D)))][0:n]  # native python int

    def load(self):
        """Load the entire dataset into memory.  This is useful for creating in-memory datasets from lazy load datasets"""
        self._objlist = self.list()
        self._loader = lambda x: x
        return self

    def take(self, n, category=None, canload=False):
        return Dataset(self.takelist(n, category=category, canload=canload))

    def take_per_category(self, n, id=None, canload=False):
        return Dataset([v for c in self.categories() for v in self.takelist(n, category=c, canload=canload)], id=id)
    
    def shuffle(self):
        """Randomly permute elements in this dataset"""
        self._objlist.sort(key=lambda x: random.random())  # in place
        return self

    def split(self, trainfraction=0.9, valfraction=0.1, testfraction=0, seed=42):
        """Split the dataset by category by fraction so that video IDs are never in the same set"""
        assert self._isvipy(), "Invalid input"
        assert trainfraction >=0 and trainfraction <= 1
        assert valfraction >=0 and valfraction <= 1
        assert testfraction >=0 and testfraction <= 1
        assert trainfraction + valfraction + testfraction == 1.0
        np.random.seed(seed)  # deterministic
        
        # Video ID assignment
        A = self.list()
        videoid = list(set([a.videoid() for a in A]))
        np.random.shuffle(videoid)
        (testid, valid, trainid) = vipy.util.dividelist(videoid, (testfraction, valfraction, trainfraction))        
        (testid, valid, trainid) = (set(testid), set(valid), set(trainid))
        d = groupbyasdict(A, lambda a: 'testset' if a.videoid() in testid else 'valset' if a.videoid() in valid else 'trainset')
        (trainset, testset, valset) = (d['trainset'] if 'trainset' in d else [], 
                                       d['testset'] if 'testset' in d else [], 
                                       d['valset'] if 'valset' in d else [])

        print('[vipy.dataset]: trainset=%d (%1.2f)' % (len(trainset), trainfraction))
        print('[vipy.dataset]: valset=%d (%1.2f)' % (len(valset), valfraction))
        print('[vipy.dataset]: testset=%d (%1.2f)' % (len(testset), testfraction))
        np.random.seed()  # re-initialize seed

        return (Dataset(trainset, id='trainset'), Dataset(valset, id='valset'), Dataset(testset, id='testset') if len(testset)>0 else None)

    def tocsv(self, csvfile=None):
        csv = [v.csv() for v in self.list]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])

    def map(self, f_map, model=None, dst=None, id=None, strict=False, ascompleted=True, chunks=128, ordered=False):        
        """Distributed map.

        To perform this in parallel across four processes:

        ```python
        D = vipy.dataset.Dataset(...)
        with vipy.globals.parallel(4):
            D.map(lambda v: ...)
        ```

        Args:
            f_map: [lambda] The lambda function to apply in parallel to all elements in the dataset.  This must return a JSON serializable object
            model: [torch.nn.Module] The model to scatter to all workers
            dst: [str] The ID to give to the resulting dataset
            id: [str] The ID to give to the resulting dataset (parameter alias for dst)
            strict: [bool] If true, raise exception on map failures, otherwise the map will return None for failed elements
            ascompleted: [bool] If true, return elements as they complete
            ordered: [bool] If true, preserve the order of objects in dataset as returned from distributed processing

        Returns:
            A `vipy.dataset.Dataset` containing the elements f_map(v).  This operation is order preserving if ordered=True.

        .. note:: 
            - This dataset must contain vipy objects of types defined in `vipy.util.class_registry` or JSON serializable objects
            - Serialization of large datasets can take a while, kick it off to a distributed dask scheduler and go get lunch
            - This method uses dask distributed and `vipy.batch.Batch` operations
            - All vipy objects are JSON serialized prior to parallel map to avoid reference cycle garbage collection which can introduce instabilities
            - Due to chunking, all error handling is caught by this method.  Use `vipy.batch.Batch` to leverage dask distributed futures error handling.
            - Operations must be chunked and serialized because each dask task comes with overhead, and lots of small tasks violates best practices
            - Serialized results are deserialized by the client and returned a a new dataset
        """
        assert callable(f_map)
        from vipy.batch import Batch   # requires pip install vipy[all]

        # Distributed map using vipy.batch
        f_serialize = lambda v,d=vipy.util.class_registry(): (str(type(v)), v.json()) if str(type(v)) in d else (None, pickle.dumps(v))  # fallback on PKL dumps/loads
        f_deserialize = lambda x,d=vipy.util.class_registry(): d[x[0]](x[1])  # with closure capture
        f_catcher = vipy.util.catcher  # catch exceptions when executing lambda and return (True, result) or (False, exception)
        S = [f_serialize(v) for v in self._objlist]  # local serialization
        B = Batch(vipy.util.chunklist(S, chunks), strict=strict, as_completed=ascompleted, warnme=False, minscatter=chunks, ordered=ordered)        
        if model is None:
            f = lambda x, f_loader=self._loader, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher: f_serializer(f_catcher(f_map, f_loader(f_deserializer(x))))  # with closure capture
            S = B.map(lambda X,f=f: [f(x) for x in X]).result()  # chunked, with caught exceptions
        else:
            f = lambda net, x, f_loader=self._loader, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher: f_serializer(f_catcher(f_map, net, f_loader(f_deserializer(x))))  # with closure capture
            S = B.scattermap((lambda net, X, f=f: [f(net, x) for x in X]), model).result()  # chunked, scattered, caught exceptions
        V = [f_deserialize(x) for s in S for x in s]  # Local deserialization and chunk flattening
        (good, bad) = ([r for (b,r) in V if b], [r for (b,r) in V if not b])  # catcher returns (True, result) or (False, exception string)
        if len(bad) > 0:
            print('[vipy.dataset.Dataset.map]: Exceptions in map distributed processing:\n%s' % str(bad))
            print('[vipy.dataset.Dataset.map]: %d/%d items failed' % (len(bad), len(self)))
        return Dataset(good, id=dst if dst is not None else id)

    def localmap(self, f):
        self._objlist = [f(v) for v in self]
        return self

    def flatmap(self, f):
        self._objlist = [x for v in self for x in f(v)]
        return self
    
    def count(self, f=None):
        """Counts for each label.  
        
        Args:
            f: [lambda] if provided, count the number of elements that return true.  This is the same as len(self.filter(f)) without modifying the dataset.

        Returns:
            A dictionary of counts per category [if f is None]
            A length of elements that satisfy f(v) = True [if f is not None]
        """
        assert self._isvipy()
        assert f is None or callable(f)
        return len([v for v in self if f is None or f(v)])

    def countby(self, f=lambda v: v.category()):
        """Count the number of elements that return the same value from the lambda function"""
        assert self._isvipy()
        assert f is None or callable(f)
        return vipy.util.countby(self, f)
        
    def frequency(self):
        return self.count()

    def histogram(self, outfile=None, fontsize=6, category_to_barcolor=None, category_to_xlabel=None):
        assert self._isvipy()
        assert category_to_barcolor is None or all([c in category_to_barcolor for c in self.categories()])
        assert category_to_xlabel is None or callable(category_to_xlabel) or all([c in category_to_xlabel for c in self.categories()])
        f_category_to_xlabel = category_to_xlabel if callable(category_to_xlabel) else ((lambda c: category_to_xlabel[c]) if category_to_xlabel is not None else (lambda c: c))
        
        d = self.countby(lambda v: v.category())
        if outfile is not None:
            (categories, freq) = zip(*reversed(sorted(list(d.items()), key=lambda x: x[1])))  # decreasing frequency
            barcolors = ['blue' if category_to_barcolor is None else category_to_barcolor[c] for c in categories]
            xlabels = [f_category_to_xlabel(c) for c in categories]
            print('[vipy.dataset]: histogram="%s"' % vipy.metrics.histogram(freq, xlabels, barcolors=barcolors, outfile=outfile, ylabel='Instances', fontsize=fontsize))
        return d
    
    def percentage(self):
        """Fraction of dataset for each label"""
        d = self.count()
        n = sum(d.values())
        return {k:v/float(n) for (k,v) in d.items()}

    def multilabel_inverse_frequency_weight(self):
        """Return an inverse frequency weight for multilabel activities, where label counts are the fractional label likelihood within a clip"""
        assert self._is_vipy_video()

        def _multilabel_inverse_frequency_weight(v):
            lbl_likelihood = {}
            if len(v.activities()) > 0:
                (ef, sf) = (max([a.endframe() for a in v.activitylist()]), min([a.startframe() for a in v.activitylist()]))  # clip length 
                lbl_frequency = vipy.util.countby([a for A in v.activitylabel(sf, ef) for a in A], lambda x: x)  # frequency within clip
                for (k,f) in lbl_frequency.items():
                    if k not in lbl_likelihood:
                        lbl_likelihood[k] = 0
                    lbl_likelihood[k] += f/(ef-sf)
            return lbl_likelihood
                    
        lbl_likelihood  = {}
        for d in self.map(lambda v: _multilabel_inverse_frequency_weight(v)):
            for (k,v) in d.items():
                if k not in lbl_likelihood:
                    lbl_likelihood[k] = 0
                lbl_likelihood[k] += v

        # Inverse frequency weight on label likelihood per clip
        d = {k:1.0/max(v,1) for (k,v) in lbl_likelihood.items()}
        n = sum(d.values())  
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def inverse_frequency_weight(self):
        """Return inverse frequency weight for categories in dataset.  Useful for unbalanced class weighting during training"""
        d = {k:1.0/max(v,1) for (k,v) in self.count().items()}
        n = sum(d.values())
        return {k:len(d)*(v/float(n)) for (k,v) in d.items()}

    def duration_in_frames(self, outfile=None):
        assert self._isvipy()
        d = {k:np.mean([v[1] for v in v]) for (k,v) in groupbyasdict([(a.category(), len(a)) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            vipy.metrics.histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (frames)', fontsize=6)            
        return d

    def duration_in_seconds(self, outfile=None, fontsize=6):
        assert self._isvipy()
        d = {k:np.mean([v[1] for v in v]) for (k,v) in groupbyasdict([(a.category(), len(a)/v.framerate()) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            vipy.metrics.histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (seconds)', fontsize=fontsize)            
        return d

    def framerate(self, outfile=None):
        assert self._isvipy()
        d = vipy.util.countby([int(round(v.framerate())) for v in self.list()], lambda x: x)
        if outfile is not None:
            vipy.metrics.pie(d.values(), ['%d fps' % k for k in d.keys()], explode=None, outfile=outfile,  shadow=False)
        return d
        
        
    def density(self, outfile=None, max=None):
        """Compute the frequency that each video ID is represented.  This counts how many activities are in a video, truncated at max"""
        assert self._isvipy()
        d = [len(v) if (max is None or len(v)<= max) else max for (k,v) in groupbyasdict(self.list(), lambda v: v.videoid()).items()]
        d = {k:v for (k,v) in sorted(vipy.util.countby(d, lambda x: x).items(), key=lambda x: x[1], reverse=True)}
        if outfile is not None:
            vipy.metrics.histogram(d.values(), d.keys(), outfile=outfile, ylabel='Frequency', xlabel='Activities per video', fontsize=6, xrot=None)            
        return d

    def boxsize(self, outfile=None, category_to_color=None, categories=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks]) if categories is None else categories

        d = {}        
        for c in object_categories:
            xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category().lower() == c.lower()) and (t.meanshape() is not None))]
            d[c] = xcyc
        
        if outfile is not None:            
            plt.clf()
            plt.figure()
            plt.grid(True)
            for c in object_categories:
                xcyc = d[c]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.scatter(xc, yc, c=category_to_color[c] if category_to_color is not None else 'blue', label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 1000, 0, 1000])                
            plt.legend()
            plt.gca().set_axisbelow(True)        
            plt.savefig(outfile)
        return d

    def boxsize_by_category(self, outfile=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks])
        
        # Mean track size per video category
        d_category_to_xy = {k:np.mean([t.meanshape() for v in vlist for t in v.tracklist()], axis=0) for (k,vlist) in groupbyasdict(self.list(), lambda v: v.category()).items()}

        if outfile is not None:
            plt.clf()
            plt.figure()
            plt.grid(True)
            colors = colorlist()            
            d_category_to_color = {c:colors[k % len(colors)] for (k,c) in enumerate(d_category_to_xy.keys())}
            for c in d_category_to_xy.keys():
                (xc, yc) = d_category_to_xy[c]
                plt.scatter(xc, yc, c=d_category_to_color[c], label=c)
            plt.xlabel('bounding box (width)')
            plt.ylabel('bounding box (height)')
            plt.axis([0, 600, 0, 600])                
            plt.gca().set_axisbelow(True)        
            lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.savefig(outfile, bbox_extra_artists=(lgd,), bbox_inches='tight')
        return d_category_to_xy

    def boxsize_histogram(self, outfile=None):
        # Scatterplot of object box sizes
        tracks = [t for s in self.list() for t in s.tracks().values()]        
        (x, y) = zip(*[(t.meanshape()[1], t.meanshape()[0]) for t in tracks])
        object_categories = set([t.category() for t in tracks])

        
        # 2D histogram of object box sizes
        for c in object_categories:
            xcyc = [(t.meanshape()[1], t.meanshape()[0]) for t in tracks if ((t.category() == c) and (t.meanshape() is not None))]
            d[c] = xcyc

        if outfile is not None:
            for c in object_categories:            
                xcyc = d[c]
                if len(xcyc) > 0:
                    (xc, yc) = zip(*xcyc)
                    plt.clf()
                    plt.figure()
                    plt.hist2d(xc, yc, bins=10)
                    plt.xlabel('Bounding box (width)')
                    plt.ylabel('Bounding box (height)')                    
                    plt.savefig(outfile % c)
        return d

    
    def to_torch(self, f_video_to_tensor):
        """Return a torch dataset that will apply the lambda function f_video_to_tensor to each element in the dataset on demand"""
        import vipy.torch
        return vipy.torch.TorchDataset(f_video_to_tensor, self)

    def to_torch_tensordir(self, f_video_to_tensor, outdir, n_augmentations=20, sleep=3):
        """Return a TorchTensordir dataset that will load a pkl.bz2 file that contains one of n_augmentations (tensor, label) pairs.
        
        This is useful for fast loading of datasets that contain many videos.

        """
        import vipy.torch    # lazy import, requires vipy[all] 
        from vipy.batch import Batch   # requires pip install vipy[all]

        assert self._is_vipy_scene()
        outdir = vipy.util.remkdir(outdir)
        vipy.batch.Batch(self.list(), as_completed=True).map(lambda v, f=f_video_to_tensor, outdir=outdir, n_augmentations=n_augmentations: vipy.util.bz2pkl(os.path.join(outdir, '%s.pkl.bz2' % v.instanceid()), [f(v.print(sleep=sleep).clone()) for k in range(0, n_augmentations)]))
        return vipy.torch.Tensordir(outdir)

    def annotate(self, outdir, mindim=512):
        assert self._isvipy()
        f = lambda v, outdir=outdir, mindim=mindim: v.mindim(mindim).annotate(outfile=os.path.join(outdir, '%s.mp4' % v.videoid())).print()
        return self.map(f, dst='annotate')

    def tohtml(self, outfile, mindim=512, title='Visualization', fraction=1.0, display=False, clip=True, activities=True, category=True):
        """Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from"""
    
        assert ishtml(outfile), "Output file must be .html"
        assert fraction > 0 and fraction <= 1.0, "Fraction must be between [0,1]"
        
        import vipy.util  # This should not be necessary, but we get "UnboundLocalError" without it, not sure why..
        import vipy.batch  # requires pip install vipy[all]

        dataset = self.list()
        assert all([isinstance(v, vipy.video.Video) for v in dataset])
        dataset = [dataset[int(k)] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
        #dataset = [v for v in dataset if all([len(a) < 15*v.framerate() for a in v.activitylist()])]  # remove extremely long videos

        quicklist = vipy.batch.Batch(dataset, strict=False, as_completed=True, minscatter=1).map(lambda v: (v.load().quicklook(), v.flush().print())).result()
        quicklist = [x for x in quicklist if x is not None]  # remove errors
        quicklooks = [imq for (imq, v) in quicklist]  # keep original video for HTML display purposes
        provenance = [{'clip':str(v), 'activities':str(';'.join([str(a) for a in v.activitylist()])), 'category':v.category()} for (imq, v) in quicklist]
        (quicklooks, provenance) = zip(*sorted([(q,p) for (q,p) in zip(quicklooks, provenance)], key=lambda x: x[1]['category']))  # sorted in category order
        return vipy.visualize.tohtml(quicklooks, provenance, title='%s' % title, outfile=outfile, mindim=mindim, display=display)


    def video_montage(self, outfile, gridrows, gridcols, mindim=64, bycategory=False, category=None, annotate=True, trackcrop=False, transpose=False, max_duration=None, framerate=30, fontsize=8):
        """30x50 activity montage, each 64x64 elements.

        Args:
            outfile: [str] The name of the outfile for the video.  Must have a valid video extension. 
            gridrows: [int, None]  The number of rows to include in the montage.  If None, infer from other args
            gridcols: [int] The number of columns in the montage
            mindim: [int] The square size of each video in the montage
            bycategory: [bool]  Make the video such that each row is a category 
            category: [str, list] Make the video so that every element is of category.  May be a list of more than one categories
            annotate: [bool] If true, include boxes and captions for objects and activities
            trackcrop: [bool] If true, center the video elements on the tracks with dilation factor 1.5
            transpose: [bool] If true, organize categories columnwise, but still return a montage of size (gridrows, gridcols)
            max_duration: [float] If not None, then set a maximum duration in seconds for elements in the video.  If None, then the max duration is the duration of the longest element.

        Returns:
            A clone of the dataset containing the selected videos for the montage, ordered rowwise in the montage

        .. notes::  
            - If a category does not contain the required number of elements for bycategory, it is removed prior to visualization
            - Elements are looped if they exit prior to the end of the longest video (or max_duration)
        """
        assert self._is_vipy_video()
        assert vipy.util.isvideo(outfile)
        assert gridrows is None or (isinstance(gridrows, int) and gridrows >= 1)
        assert gridcols is None or (isinstance(gridcols, int) and gridcols >= 1)
        assert isinstance(mindim, int) and mindim >= 1
        assert category is None or isinstance(category, str)

        D = self.clone()
        if bycategory:
            (num_categories, num_elements) = (gridrows, gridcols) if not transpose else (gridcols, gridrows)
            assert num_elements is not None
            requested_categories = sorted(D.classlist()) if (num_categories is None) else sorted(D.classlist())[0:num_categories]             
            categories = [c for c in requested_categories if D.count()[c] >= num_elements]  # filter those categories that do not have enough
            if set(categories) != set(requested_categories):
                warnings.warn('[vipy.dataset.video_montage]: removing "%s" without at least %d examples' % (str(set(requested_categories).difference(set(categories))), num_elements))
            vidlist = sorted(D.filter(lambda v: v.category() in categories).take_per_category(num_elements, canload=True).tolist(), key=lambda v: v.category())
            vidlist = vidlist if not transpose else [vidlist[k] for k in np.array(range(0, len(vidlist))).reshape( (len(categories), num_elements) ).transpose().flatten().tolist()] 
            (gridrows, gridcols) = (len(categories), num_elements) if not transpose else (num_elements, len(categories))
            assert len(vidlist) == gridrows*gridcols

        elif category is not None:
            vidlist = D.filter(lambda v: v.category() in vipy.util.tolist(category)).take(gridrows*gridcols, canload=True).tolist()            
        elif len(D) != gridrows*gridcols:
            vidlist = D.take(gridrows*gridcols, canload=True).tolist()
        else:
            vidlist = D.tolist()

        vidlist = [v.framerate(framerate) for v in vidlist]  # resample to common framerate (this may result in jittery tracks
        montage = Dataset(vidlist, id='video_montage').clone()  # for output
        vidlist = [v.trackcrop(dilate=1.5, maxsquare=True) if (v.trackbox() is not None) else v for v in vidlist] if trackcrop else vidlist  # may be None, if so return the video
        vidlist = [v.mindim(mindim) for v in vidlist]  # before annotate for common font size
        vidlist = [vipy.video.Video.cast(v) for v in vidlist] if not annotate else [v.annotate(verbose=False, fontsize=fontsize) for v in vidlist]  # pre-annotate
            
        vipy.visualize.videomontage(vidlist, mindim, mindim, gridrows=gridrows, gridcols=gridcols, framerate=framerate, max_duration=max_duration).saveas(outfile)
        return montage        
        
    def zip(self, other, sortkey=None):
        """Zip two datasets.  Equivalent to zip(self, other).

        ```python
        for (d1,d2) in D1.zip(D2, sortkey=lambda v: v.instanceid()):
            pass
        
        for (d1, d2) in zip(D1, D2):
            pass
        ```

        Args:
            other: [`vipy.dataset.Dataset`] 
            sortkey: [lambda] sort both datasets using the provided sortkey lambda.
        
        Returns:
            Generator for the tuple sequence ( (self[0], other[0]), (self[1], other[1]), ... )
        """ 
        assert isinstance(other, Dataset)
        assert len(self) == len(other)

        for (vi, vj) in zip(self.sort(sortkey), other.sort(sortkey)):
            yield (vi, vj)

    def sort(self, key):
        """Sort the dataset in-place using the sortkey lambda function"""
        if key is not None:
            self._objlist.sort(key=lambda x: key(self._loader(x)))
        return self
                
