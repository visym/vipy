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
import atexit
from vipy.util import is_email_address
from vipy.batch import Batch       
import hashlib
import pickle
import time
import json
import dill
from vipy.show import colorlist
import matplotlib.pyplot as plt
import vipy.metrics
import gc 

class Dataset():
    """vipy.dataset.Dataset() class
    
    Common class to manipulate large sets of vipy objects in parallel

    >>> D = vipy.dataset.Dataset([vipy.video.RandomScene(), vipy.video.RandomScene()], id='random_scene')
    >>> with vipy.globals.parallel(2):
    >>>     D = D.map(lambda v: v.frame(0))
    >>> list(D)

    """

    def __init__(self, objlist_or_filename, id=None, abspath=True):
        objlist = vipy.util.load(objlist_or_filename, abspath=abspath) if (vipy.util.isjsonfile(objlist_or_filename) or vipy.util.ispklfile(objlist_or_filename)) else objlist_or_filename
        assert isinstance(objlist, list), "Invalid input"
        self._saveas_ext = ['pkl', 'json']
        self._id = id if id is not None else (vipy.util.filetail(objlist_or_filename) if isinstance(objlist_or_filename, str) else uuid.uuid4().hex)
        self._objlist = tolist(objlist)
        assert len(self._objlist) > 0, "Empty dataset"

    def __repr__(self):
        if len(self) > 0:
            return str('<vipy.dataset: id="%s", len=%d, type=%s>' % (self.id(), len(self), str(type(self._objlist[0]))))
        else:
            return str('<vipy.dataset: id="%s", len=0>' % (self.id()))

    def __iter__(self):
        for k in range(len(self)):
            yield self._objlist[k]

    def __getitem__(self, k):
        if isinstance(k, int):
            assert k>=0 and k<len(self._objlist), "invalid index"
            return self._objlist[k]
        elif isinstance(k, slice):
            return self._objlist[k.start:k.stop:k.step]
        else:
            raise
            

    def __len__(self):
        return len(self._objlist)

    def id(self, n=None):
        """Set or return the dataset id"""
        if n is None:
            return self._id
        else:
            self._id = n
            return self

    def list(self):
        """Return the dataset as a list"""
        return self._objlist
    def tolist(self):
        """Alias for self.list()"""
        return self._objlist

    def flatten(self):
        """Convert dataset stored as a list of lists into a flat list"""
        self._objlist = [o for objlist in self._objlist for o in vipy.util.tolist(objlist)]
        return self

    def istype(self, validtype):
        """Return True if all elements in the dataset are of type 'validtype'"""
        return all([any([isinstance(v,t) for t in tolist(validtype)]) for v in self._objlist])

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

    def archive(self, tarfile, delprefix, mediadir='', format='json', castas=vipy.video.Scene, verbose=False, extrafiles=None, novideos=False):
        """Create a archive file for this dataset.  This will be archived as:

           /path/to/tarfile.{tar.gz|.tgz|.bz2}
              tarfilename
                 tarfilename.{json|pkl}
                 mediadir/
                     video.mp4
                 extras1.ext
                 extras2.ext
        
            Inputs:
              - tarfile: /path/to/tarfilename.tar.gz
              - delprefix:  the absolute file path contained in the media filenames to be removed.  If a video has a delprefix='/a/b' then videos with path /a/b/c/d.mp4' -> 'c/d.mp4', and {JSON|PKL} will be saved with relative paths to mediadir
              - mediadir:  the subdirectory name of the media to be contained in the archive.  Usually "videos".             
              - extrafiles: list of tuples or singletons [(abspath, filename_in_archive_relative_to_root), 'file_in_root_and_in_pwd', ...], 

            Example:  

              - Input files contain /path/to/oldvideos/category/video.mp4
              - Output will contain relative paths videos/category/video.mp4

              >>> d.archive('out.tar.gz', delprefix='/path/to/oldvideos', mediadir='videos')
        
        """
        assert self._isvipy(), "Source dataset must contain vipy objects for staging"
        assert all([os.path.isabs(v.filename()) for v in self]), "Input dataset must have only absolute media paths"
        assert self.countby(lambda v: delprefix in v.filename()) > 0, "delprefix not found"
        assert self.countby(lambda v: delprefix in v.filename()) == len(self), "all media objects must have the same delprefix for relative path construction"
        assert vipy.util.istgz(tarfile) or vipy.util.isbz2(tarfile), "Allowable extensions are .tar.gz, .tgz or .bz2"
        assert shutil.which('tar') is not None, "tar not found on path"        

        D = self.clone()
        stagedir = remkdir(os.path.join(tempdir(), filefull(filetail(tarfile))))
        print('[vipy.dataset]: creating staging directory "%s"' % stagedir)        
        D._objlist = [v.filename(v.filename().replace(os.path.normpath(delprefix), os.path.normpath(os.path.join(stagedir, mediadir))), symlink=not novideos) for v in D.list()]
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
        print('[vipy.dataset]: %s, MD5=%s' % (tarfile, vipy.downloader.generate_md5(tarfile)))
        return tarfile
        
    def save(self, outfile, nourl=False, castas=None, relpath=False, sanitize=True, strict=True, significant_digits=2, noemail=True, flush=True):
        """Save the dataset to the provided output filename stored as pkl or json
        
        Args:
            outfile [str]: The /path/to/out.pkl or /path/to/out.json
            nourl [bool]: If true, remove all URLs from the media (if present)
            castas [type]:  Cast all media to the provided type.  This is useful for downcasting to `vipy.video.Scene` from superclasses
            relpath [bool]: If true, define all file paths in objects relative to the /path/to in /path/to/out.json
            sanitize [bool]:  If trye, call sanitize() on all objects to remove all private attributes with prepended '__' 
            strict [bool]: Unused
            significant_digits [int]: Assign the requested number of significant digits to all bounding boxes in all tracks.  This requires dataset of `vipy.video.Scene`
            noemail [bool]: If true, scrub the attributes for emails and replace with a hash
            flush [bool]:  If true, flush the object buffers prior to save

        Returns:        
            This dataset that is quivalent to vipy.dataset.Dataset('/path/to/outfile.json')
        """
        n = len([v for v in self._objlist if v is None])
        if n > 0:
            print('[vipy.dataset]: removing %d invalid elements' % n)
        objlist = [v for v in self._objlist if v is not None]  
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
        return sorted(list(set([v.category() for v in self._objlist])))

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
        return list(sorted(set([tuple(sorted(list(a))) for v in self._objlist for a in v.activitylabel() if len(a) > 0])))        

    def powerset_to_index(self):        
        assert self._isvipy(), "Invalid input"
        return {c:k for (k,c) in enumerate(self.powerset())}

    def dedupe(self, key):
        self._objlist = list({key(v):v for v in self._objlist}.values())
        return self
        
    def countby(self, f):
        return len([v for v in self._objlist if f(v)])

    def union(self, other, key=None):
        assert isinstance(other, Dataset), "invalid input"
        self._objlist = self._objlist + other._objlist
        return self.dedupe(key) if key is not None else self
    
    def difference(self, other, key):
        assert isinstance(other, Dataset), "invalid input"
        idset = set([key(v) for v in self._objlist]).difference([key(v) for v in other._objlist])   # in A but not in B
        self._objlist = [v for v in self._objlist if key(v) in idset]
        return self
        
    def has(self, val, key):
        return any([key(obj) == val for obj in self._objlist])

    def replace(self, other, key):
        """Replace elements in self with other with equality detemrined by the key lambda function"""
        assert isinstance(other, Dataset), "invalid input"
        d = {key(v):v for v in other}
        self._objlist = [v if key(v) not in d else d[key(v)] for v in self._objlist]
        return self

    def merge(self, other, outdir, selfdir, otherdir):
        assert isinstance(other, Dataset), "invalid input"
        (selfdir, otherdir, outdir) = (os.path.normpath(selfdir), os.path.normpath(otherdir), vipy.util.remkdir(os.path.normpath(outdir)))
        assert all([selfdir in v.filename() for v in self._objlist])
        assert all([otherdir in v.filename() for v in other._objlist])

        D1 = self.clone().localmap(lambda v: v.filename(v.filename().replace(selfdir, outdir), copy=False, symlink=True))
        D2 = other.clone().localmap(lambda v: v.filename(v.filename().replace(otherdir, outdir), copy=False, symlink=True))
        return D1.union(D2)

    def augment(self, f, n_augmentations):
        assert n_augmentations >= 1
        self._objlist = [f(v.clone()) for v in self._objlist for k in range(n_augmentations)]  # This will remove the originals
        return self

    def filter(self, f):
        self._objlist = [v for v in self._objlist if f(v)]
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
        objlist = [obj for obj in self._objlist if f(obj)]
        return [] if (len(objlist) == 0 or n == 0) else (objlist[0] if n==1 else objlist[0:n])

    def to_jsondir(self, outdir):
        print('[vipy.dataset]: exporting %d json files to "%s"...' % (len(self), outdir))
        vipy.util.remkdir(outdir)  # to avoid race condition
        Batch(vipy.util.chunklist([(k,v) for (k,v) in enumerate(self._objlist)], 64), as_completed=True, minscatter=1).map(lambda X: [vipy.util.save(x[1].clone(), os.path.join(outdir, '%s_%d.json' % (x[1].clone().videoid(), x[0]))) for x in X]).result()
        return outdir

    def takelist(self, n, category=None, canload=False):
        assert n >= 0, "Invalid length"

        outlist = []
        objlist = self._objlist if category is None else [v for v in self._objlist if v.category() == category]
        for k in np.random.permutation(range(0, len(objlist))).tolist():
            if not canload or objlist[k].isloadable():
                outlist.append(objlist[k])  # without replacement
            if len(outlist) == n:
                break
        return outlist

    def take(self, n, category=None, canload=False):
        return Dataset(self.takelist(n, category=category, canload=canload))

    def take_per_category(self, n, id=None, canload=False):
        return Dataset([v for c in self.categories() for v in self.takelist(n, category=c, canload=canload)], id=id)
    
    def split(self, trainfraction=0.9, valfraction=0.1, testfraction=0, seed=42):
        """Split the dataset by category by fraction so that video IDs are never in the same set"""
        assert self._isvipy(), "Invalid input"
        assert trainfraction >=0 and trainfraction <= 1
        assert valfraction >=0 and valfraction <= 1
        assert testfraction >=0 and testfraction <= 1
        assert trainfraction + valfraction + testfraction == 1.0

        np.random.seed(seed)
        A = self.list()
        
        # Video ID assignment
        videoid = list(set([a.videoid() for a in A]))
        np.random.shuffle(videoid)
        (testid, valid, trainid) = vipy.util.dividelist(videoid, (testfraction, valfraction, trainfraction))        
        (testid, valid, trainid) = (set(testid), set(valid), set(trainid))
        d = groupbyasdict(A, lambda a: 'testset' if a.videoid() in testid else 'valset' if a.videoid() in valid else 'trainset')
        (trainset, testset, valset) = (d['trainset'] if 'trainset' in d else [], 
                                       d['testset'] if 'testset' in d else [], 
                                       d['valset'] if 'valset' in d else [])

        print('[vipy.dataset]: trainset=%d (%1.1f)' % (len(trainset), trainfraction))
        print('[vipy.dataset]: valset=%d (%1.1f)' % (len(valset), valfraction))
        print('[vipy.dataset]: testset=%d (%1.1f)' % (len(testset), testfraction))
        
        return (Dataset(trainset, id='trainset'), Dataset(valset, id='valset'), Dataset(testset, id='testset') if len(testset)>0 else None)

    def tocsv(self, csvfile=None):
        csv = [v.csv() for v in self.list]        
        return vipy.util.writecsv(csv, csvfile) if csvfile is not None else (csv[0], csv[1:])

    def map(self, f_transform, model=None, dst=None, id=None, checkpoint=False, strict=False, ascompleted=True):        
        """Distributed map.

        To perform this in parallel across four processes:

        >>> with vipy.globals.parallel(4):
        >>>     self.map(lambda v: ...)

        Args:
            f_transform: [lambda] The lambda function to apply in parallel to all elements in the dataset
            model: [torch.nn.Module] The model to scatter to all workers
            dst: [str] The ID to give to the resulting dataset
            id: [str] The ID to give to the resulting dataset (parameter alias for dst)
            checkpoint: [bool] If trye, checkpoint the map operation
            strict: [bool] If true, raise exception on map failures, otherwise the map will return None for failed elements
            ascompleted: [bool] If true, return elements as they complete

        Returns:
            A `vipy.dataset.Dataset` containing the elements f_transform(v).  This operation is order preserving.

        """
        assert callable(f_transform)
        B = Batch(self.list(), strict=strict, as_completed=ascompleted, checkpoint=checkpoint, warnme=False, minscatter=1000000)
        V = B.map(f_transform).result() if not model else B.scattermap(f_transform, model).result()
        D = Dataset(V, id=dst if dst is not None else id)
        return D

    def localmap(self, f):
        self._objlist = [f(v) for v in self._objlist]
        return self

    def flatmap(self, f):
        self._objlist = [x for v in self._objlist for x in f(v)]
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
        return vipy.util.countby(self.list(), lambda v: v.category()) if f is None else len([v for v in self if f(v)])

    def frequency(self):
        return self.count()

    def histogram(self, outfile=None, fontsize=6, category_to_barcolor=None):
        assert self._isvipy()
        assert category_to_barcolor is None or all([c in category_to_barcolor for c in self.categories()])
        
        d = self.count()
        if outfile is not None:
            from vipy.metrics import histogram            
            (categories, freq) = zip(*reversed(sorted(list(d.items()), key=lambda x: x[1])))  # decreasing frequency
            barcolors = ['blue' if category_to_barcolor is None else category_to_barcolor[c] for c in categories]
            print('[vipy.dataset]: histogram="%s"' % vipy.metrics.histogram(freq, categories, barcolors=barcolors, outfile=outfile, ylabel='Instances', fontsize=fontsize))
        return d
    
    def percentage(self):
        """Fraction of dataset for each label"""
        d = self.count()
        n = sum(d.values())
        return {k:v/float(n) for (k,v) in d.items()}

    def multilabel_inverse_frequency_weight(self):
        """Return an inverse frequency weight for multilabel activities, where label counts are the fractional label likelihood within a clip"""
        assert self.is_vipy_video()

        lbl_likelihood  = {k:0 for k in self.classlist()}
        for v in self.list():
            if len(v.activities()) > 0:
                (ef, sf) = (max([a.endframe() for a in v.activitylist()]), min([a.startframe() for a in v.activitylist()]))  # clip length 
                lbl_frequency = vipy.util.countby([a for A in v.activitylabel(sf, ef) for a in A], lambda x: x)  # frequency within clip
                for (k,f) in lbl_frequency.items():
                    lbl_likelihood[k] += f/(ef-sf)

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
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (frames)', fontsize=6)            
        return d

    def duration_in_seconds(self, outfile=None, fontsize=6):
        assert self._isvipy()
        d = {k:np.mean([v[1] for v in v]) for (k,v) in groupbyasdict([(a.category(), len(a)/v.framerate()) for v in self.list() for a in v.activitylist()], lambda x: x[0]).items()}
        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Duration (seconds)', fontsize=fontsize)            
        return d

    def framerate(self, outfile=None):
        assert self._isvipy()
        d = vipy.util.countby([int(round(v.framerate())) for v in self.list()], lambda x: x)
        if outfile is not None:
            from vipy.metrics import pie
            pie(d.values(), ['%d fps' % k for k in d.keys()], explode=None, outfile=outfile,  shadow=False)
        return d
        
        
    def density(self, outfile=None):
        assert self._isvipy()
        d = [len(v) for (k,v) in groupbyasdict(self.list(), lambda v: v.videoid()).items()]
        d = vipy.util.countby(d, lambda x: x)
        if outfile is not None:
            from vipy.metrics import histogram
            histogram(d.values(), d.keys(), outfile=outfile, ylabel='Frequency', xlabel='Activities per video', fontsize=6, xrot=None)            
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

    def to_torch_tensordir(self, f_video_to_tensor, outdir, n_augmentations=20, n_chunks=512):
        """Return a TorchTensordir dataset that will load a pkl.bz2 file that contains one of n_augmentations (tensor, label) pairs.
        
        This is useful for fast loading of datasets that contain many videos.

        """
        import vipy.torch
        assert self.is_vipy_scene()
        outdir = vipy.util.remkdir(outdir)
        B = vipy.util.chunklist(self._objlist, n_chunks)
        vipy.batch.Batch(B, as_completed=True, minscatter=1).map(lambda V, f=f_video_to_tensor, outdir=outdir, n_augmentations=n_augmentations: [vipy.util.bz2pkl(os.path.join(outdir, '%s.pkl.bz2' % v.instanceid()), [f(v.clone()) for k in range(0, n_augmentations)]) for v in V])
        return vipy.torch.TorchTensordir(outdir)

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
        dataset = [dataset[k] for k in np.random.permutation(range(len(dataset)))[0:int(len(dataset)*fraction)]]
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

        >>> for (d1,d2) in D1.zip(D2, sortkey=lambda v: v.instanceid()):
        >>>     pass
        
        >>> for (d1, d2) in zip(D1, D2):
        >>>     pass

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
            self._objlist.sort(key=key)
        return self
                
