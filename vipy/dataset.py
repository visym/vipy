import os
import numpy as np
from vipy.util import findpkl, toextension, filepath, filebase, jsonlist, ishtml, ispkl, filetail, temphtml, env, cache
from vipy.util import listpkl, listext, templike, tempdir, remkdir, tolist, fileext, writelist, tempcsv, to_iterable
from vipy.util import newpathroot, listjson, extlist, filefull, tempdir, groupbyasdict, try_import, shufflelist, catcher
from vipy.util import is_email_address, findjson, findimages, isimageurl, countby, chunkgen, chunkgenbysize, dividelist, chunklist
from vipy.util import findvideos, truncate_string
from vipy.globals import log
import random
import shutil
import copy 
import gc 
import itertools
from pathlib import Path
import functools
import concurrent.futures as cf
import vipy.parallel


class Dataset():
    """vipy.dataset.Dataset() class
    
    Common class to manipulate large sets of objects in parallel

    Args:
        - dataset [list, tuple, set, obj]: a python built-in type that supports indexing or a generic object that supports indexing and has a length
        - id [str]: an optional id of this dataset, which provides a descriptive name of the dataset
        - loader [callable]: a callable loader that will construct the object from a raw data element in dataset.  This is useful for custom deerialization or on demand transformations

    Datasets can be indexed, shuffled, iterated, minibatched, sorted, sampled, partitioned.
    Datasets constructed of vipy objects are lazy loaded, delaying loading pixels until they are needed

    ```python
    (trainset, valset, testset) = vipy.dataset.registry('mnist')

    (trainset, valset) = trainset.partition(0.9, 0.1)
    categories = trainset.set(lambda im: im.category())
    smaller = testset.take(1024)
    preprocessed = smaller.map(lambda im: im.resize(32, 32).gain(1/256))
    
    for b in preprocessed.minibatch(128):
        print(b)

    # visualize the dataset 
    (trainset, valset, testset) = vipy.dataset.registry('pascal_voc_2007')
    for im in trainset:
        im.mindim(1024).show().print(sleep=1).close()
    
    ```

    Datasets can be constructed from directories of json files or image files (`vipy.dataset.Dataset.from_directory`)
    Datasets can be constructed from a single json file containing a list of objects (`vipy.dataset.Dataset.from_json`)
    
    ..note:: that if a lambda function is provided as loader then this dataset is not serializable.  Use self.load() then serialize
    """

    __slots__ = ('_id', '_ds', '_idx', '_loader', '_type')
    def __init__(self, dataset, id=None, loader=None):
        assert loader is None or callable(loader)
        
        self._id = id
        self._ds = dataset if not isinstance(dataset, Dataset) else dataset._ds
        self._idx = None if not isinstance(dataset, Dataset) else dataset._idx   # random access on-demand
        self._loader = loader if not isinstance(dataset, Dataset) else dataset._loader  # not serializable if lambda is provided

        try:
            self._type = str(type(self._loader(dataset[0]) if self._loader else dataset[0]))  # peek at first element, cached
        except:
            self._type = None


    @classmethod
    def from_directory(cls, indir, filetype='json', id=None):
        """Recursively search indir for filetype, construct a dataset from all discovered files of that type"""
        if filetype == 'json':
            return cls([x for f in findjson(indir) for x in to_iterable(vipy.load(f))], id=id)
        elif filetype.lower() in ['jpg','jpeg','images']:
            return cls([vipy.image.Image(filename=f) for f in findimages(indir)], id=id)            
        elif filetype.lower() in ['mp4','videos']:
            return cls([vipy.image.Video(filename=f) for f in findvideos(indir)], id=id)            
        else:
            raise ValueError('unsupported file type "%s"' % filetype)

    @classmethod
    def from_image_urls(cls, urls, id=None):
        """Construct a dataset from a list of image URLs"""
        return cls([vipy.image.Image(url=url) for url in to_iterable(urls) if isimageurl(url)], id=id)
        
    @classmethod
    def from_json(cls, jsonfile, id=None):
        return cls([x for x in to_iterable(vipy.load(jsonfile))], id=id)

    @classmethod
    def cast(cls, obj):
        return cls(obj) if not isinstance(obj, Dataset) else obj
    
    def __repr__(self):
        fields = ['id=%s' % truncate_string(self.id(), maxlen=80)] if self.id() else []
        fields += ['len=%d' % self.len()] if self.len() is not None else []
        fields += ['type=%s' % self._type] if self._type else []
        return str('<vipy.dataset.Dataset: %s>' % ', '.join(fields))

    def __iter__(self):            
        if self.is_streaming():
            for x in self._ds:  # iterable access (faster)
                yield self._loader(x) if self._loader is not None else x                 
        else:
            for k in range(len(self)):
                yield self[k]   # random access (slower)                


    def __getitem__(self, k):
        assert self.len() is not None, "dataset does not support indexing"
        
        idx = self.index()  # convert to random access on demand
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(idx), "invalid index"
            x = self._ds[idx[int(k)]]
            x = self._loader(x) if self._loader is not None else x
            return x
        elif isinstance(k, slice):
            X = [self._ds[k] for k in idx[k.start:k.stop:k.step]]
            X = [self._loader(x) for x in X] if self._loader is not None else X
            return X
        else:
            raise ValueError('invalid slice "%s"' % type(k))            

    def raw(self):
        """Return a view of this dataset without the loader"""
        return Dataset(self._ds, loader=None)
    
    def is_streaming(self):
        return self._idx is None

    def len(self):
        return len(self._idx) if self._idx is not None else (len(self._ds) if hasattr(self._ds, '__len__') else None)

    def __len__(self):
        len = self.len()
        if len is None:
            raise ValueError('dataset has no length')
        return len
    
    def __or__(self, other):
        assert isinstance(other, Dataset)
        return Union(self, other, id=self.id())
    
    def id(self, new_id=None):
        """Change the dataset ID to the provided ID, or return it if None"""
        if new_id is not None:
            self._id = new_id
            return self
        return self._id

    def index(self, index=None, strict=False):
        """Update the index, useful for filtering of large datasets"""
        if index is not None:
            assert not strict or index is None or (len(index)>0 and len(index)<=len(self) and max(index)<len(self) and min(index)>=0)            
            self._idx = index
            return self
        if self._idx is None:
            self._idx = list(range(len(self._ds)))  # on-demand index, only if underlying dataset has known length
        return self._idx
    
        
    def clone(self, deep=False):
        """Return a copy of the dataset object"""
        if not deep:
            return copy.copy(self) 
        else:
            return copy.deepcopy(self)
    
    def shuffle(self, shuffler=None):
        """Permute elements in this dataset uniformly at random in place using the optimal shuffling strategy for the dataset structure to maximize performance.
           This method will use either Dataset.streaming_shuffler (for iterable datasets) or Dataset.uniform_shuffler (for random access datasets)
        """
        assert shuffler is None or callable(shuffler)
        shuffler = shuffler if shuffler is not None else (Dataset.streaming_shuffler if self.is_streaming() else Dataset.uniform_shuffler)
        return shuffler(self)

    def repeat(self, n):
        """Repeat the dataset n times.  If n=0, the dataset is unchanged, if n=1 the dataset is doubled in length, etc."""
        assert n>=0
        return self.index( self.index()*(n+1) )
    
    def tuple(self, mapper=None, flatten=False, reducer=None):
        """Return the dataset as a tuple, applying the optional mapper lambda on each element, applying optional flattener on sequences returned by mapper, and applying the optional reducer lambda on the final tuple, return a generator"""
        assert mapper is None or callable(mapper)
        assert reducer is None or callable(reducer)
        mapped = self.map(mapper) if mapper else self
        flattened = (y for x in mapped for y in x) if flatten else (x for x in mapped)
        reduced = reducer(flattened) if reducer else flattened
        return reduced

    def list(self, mapper=None, flatten=False):
        """Return a tuple as a list, loading into memory"""
        return self.tuple(mapper, flatten, reducer=list)

    def set(self, mapper=None, flatten=False):
        """Return the dataset as a set.  Mapper must be a lambda function that returns a hashable type"""
        return self.tuple(mapper=mapper, reducer=set, flatten=flatten)        

    def all(self, mapper):
        return self.tuple(mapper=mapper, reducer=all)
    
    def frequency(self, f):
        """Frequency counts for which lambda returns the same value.  For example f=lambda im: im.category() returns a dictionary of category names and counts in this category"""
        return countby(self.tuple(mapper=f))

    def balanced(self, f):
        """Is the dataset balanced (e.g. the frequencies returned from the lambda f are all the same)?"""
        return len(set(self.frequency(f).values())) == 1
    
    def count(self, f):
        """Counts for each element for which lamba returns true.  
        
        Args:
            f: [lambda] if provided, count the number of elements that return true.  

        Returns:
            A length of elements that satisfy f(v) = True [if f is not None]
        """
        return len(self.tuple(f, reducer=lambda X: [x for x in X if x is True]))

    def countby(self, f):
        return self.frequency(f)
    
    def filter(self, f):
        """In place filter with lambda function f, keeping those elements obj in-place where f(obj) evaluates true.  Callable should return bool"""
        assert callable(f)
        return self.index( [i for (b,i) in zip(self.localmap(f), self.index()) if b] )
    
    def take(self, n, inplace=False):
        """Randomly Take n elements from the dataset, and return a dataset (in-place or cloned). If n is greater than the size of the dataset, sample with replacement, if n is less than the size of the dataset, sample without replacement"""
        assert isinstance(n, int) and n>0
        D = self.clone() if not inplace else self
        return D.index(list((random.sample if n<= len(self) else random.choices)(D.index(), k=n)) )


    def groupby(self, f):
        """Group the dataset according to the callable f, returning dictionary of grouped datasets."""
        assert callable(f)        
        return {k:self.clone().index([x[1] for x in v]).id('%s:%s' % (self.id(),str(k))) for (k,v) in itertools.groupby(enumerate(self.sort(f).index()), lambda x: f(self[x[0]]))}

    def takeby(self, f, n):
        """Filter the dataset according to the callable f, take n from each group and return a dataset.  Callable should return bool.  If n==1, return a singleton"""
        d = self.clone().filter(f)
        return d.take(n) if n>1 else d.takeone()

    def takelist(self, n):
        """Take n elements and return list.  The elements are loaded and not cloned."""
        return self.take(n).list()

    def takeone(self):
        """Randomly take one element from the dataset and return a singleton"""
        return self[random.randint(0, len(self)-1)]

    def sample(self):
        """Return a single element sampled uniformly at random"""
        return self.takeone()
    
    def take_fraction(self, p, inplace=False):
        """Randomly take a percentage of the dataset, returning a clone or in-place"""
        assert p>=0 and p<=1, "invalid fraction '%s'" % p
        return self.take(n=int(len(self)*p), inplace=inplace)

    def inverse_frequency(self, f):
        """Return the inverse frequency of elements grouped by the callable f.  Returns a dictionary of the callable output to inverse frequency """
        attributes = self.set(f)
        frequency = self.frequency(f)
        return {a:(1/len(attributes))*(len(self)/frequency[a]) for a in attributes}  # (normalized) inverse frequency weight
    
    def load(self):
        """Cache the entire dataset into memory"""
        return Dataset([x for x in self], id=self.id())
    
    def chunk(self, n):
        """Yield n chunks as list.  Last chunk will be ragged."""
        for (k,c) in enumerate(chunkgen(self, n)):
            yield list(c)

    def batch(self, n):
        """Yield batches of size n as datasets.  Last batch will be ragged.  Batches are not loaded.  Batches have appended id equal to the zero-indexed batch order"""
        for (k,b) in enumerate(chunkgenbysize(self, n)):  
            yield Dataset(b).id('%s:%d' % (self.id() if self.id() else '', k))
                                
    def minibatch(self, n, ragged=True, loader=None, bufsize=1024, accepter=None, preprocessor=None):
        """Yield preprocessed minibatches of size n of this dataset.

        To yield chunks of this dataset, suitable for minibatch training/testing

        ```python
        D = vipy.dataset.Dataset(...)
        for b in D.minibatch(n):
           print(b)
        ```
        
        To perform minibatch image downloading in parallel across four processes with the context manager:

        ```python
        D = vipy.dataset.registry('yfcc100m_url:train').take(128)
        with vipy.globals.parallel(4):
            for b in D.minibatch(16, loader=vipy.image.Transform.download, accepter=lambda im: im.is_downloaded()):
                print(b)  # complete minibatch that passed accepter
        ```

        Args:
            n [int]: The size of the minibatch
            ragged [bool]: If ragged=true, then the last chunk will be ragged with len(chunk)<n, else skipped
            bufsize [int]:  The size of the buffer used in parallel processing of elements.  Useful for parallel loading
            accepter [callable]:  A callable that returns true|false on an element, where only elements that return true are included in the minibatch.  useful for parallel loading of elements that may fail to download
            loader [callable]: A callable that is applied to every element of the dataset.  Useful for parallel loading

        Returns:        
            Iterator over `vipy.dataset.Dataset` elements of length n.  Minibatches will be yielded loaded and preprocessed (processing done concurrently if vipy.parallel.executor() is initialized)

        ..note:: The distributed iterator appends the minibatch index to the minibatch.id().  
        ..note:: If there exists a vipy.parallel.exeuctor(), then loading and preprocessing will be performed concurrently

        """
        for (k,b) in enumerate(chunkgenbysize(vipy.parallel.iter(self, mapper=loader, bufsize=max(bufsize,n), accepter=accepter), n)): 
            if ragged or len(b) == n:
                yield Dataset.cast(b).id('%s:%d' % (self.id() if self.id() else '', k))                    
                    
                        
    def shift(self, m):
        """Circular shift the dataset m elements to the left, so that self[k+m] == self.shift(m)[k].  Circular shift for boundary handling so that self.shift(m)[-1] == self[m-1]"""
        return self.clone().index(self.index()[m:] + self.index()[0:m])

    def slice(self, start=0, stop=-1, step=1):
        """Slice the dataset to contain elements defined by slice(start, stop, step)"""
        return self.clone().index(self.index()[start:stop:step])
        
    def truncate(self, m):
        """Truncate the dataset to contain the first m elements only"""
        return self.slice(stop=m)
    
    def pipeline(self, n, m, ragged=True, prepad=True, postpad=True):
        """Yield pipelined minibatches of size n with pipeline length m.

        A pipelined minibatch is a tuple (head, tail) such that (head, tail) are minibatches at different indexes in the dataset.  
        Head corresponds to the current minibatch and tail corresponds to the minibatch left shifted by (m-1) minibatches.

        This structure is useful for yielding datasets for pipelined training where head contains the minibatch that will complete pipeline training on this iteration, and tail contains the 
        next minibatch to be inserted into the pipeline on this iteration.
        
        ```python
        D = vipy.dataset.Dataset(...)
        for (head, tail) in D.pipeline(n, m, prepad=False, postpad=False):
            assert head == D[0:m]
            assert tail == D[n*(m-1): n*(m-1)+n]

        Args:
            n [int]: The size of each minibatch
            m [int]:  The pipeline length in minibatches
            ragged [bool]: If ragged=true, then the last chunk will be ragged with len(chunk)<n, else skipped
            prepad: If true, yield (head, tail) == (None, batch) when filling the pipeline
            postpad: If true, yield (head, tail) == (batch, None) when flushing the pipeline
        
        Returns:        
            Iterator over tuples (head,tail) of `vipy.dataset.Dataset` elements of length n where tail is left shifted by n*(m-1) elements. 
        
        .. note::  The distributed iterator is not order preserving over minibatches and yields minibatches as completed, however the tuple (head, tail) is order preserving within the pipeline
        .. note:: If there exists a vipy.parallel.executor(), then loading and preprocessing will be performed concurrently
        
        """
        pipeline = [] 
        for (k,b) in enumerate(self.minibatch(n, ragged=ragged)):  # not order preserving
            pipeline.append(b)  # order preserving within pipeline                        
            if k < m-1:
                if prepad:
                    yield( (None, b) )  
            else:
                yield( (pipeline.pop(0), b) )  # yield deque-like (minibatch, shifted minibatch) tuples
        for p in pipeline:
            if postpad:
                yield( (p, None) )


    def chunks(self, sizes):
        """Partition the dataset into chunks of size given by the tuple in partitions, and give the dataset suffix if provided"""
        assert sum(sizes) == len(self)

        i = 0
        datasets = []
        for n in sizes:
            datasets.append(self.clone().index(self.index()[i:i+n]))
            i += n
        return datasets
        
    def partition(self, trainfraction=0.9, valfraction=0.1, testfraction=0, trainsuffix=':train', valsuffix=':val', testsuffix=':test'):
        """Partition the dataset into the requested (train,val,test) fractions.  

        Args:
            trainfraction [float]: fraction of dataset for training set
            valfraction [float]: fraction of dataset for validation set
            testfraction [float]: fraction of dataset for test set
            trainsuffix: If not None, append this string the to trainset ID
            valsuffix: If not None, append this string the to valset ID
            testsuffix: If not None, append this string the to testset ID        
        
        Returns:        
            (trainset, valset, testset) such that trainset is the first trainfraction of the dataset.  

        .. note:: This does not permute the dataset.  To randomize split, shuffle dataset first

        """
        assert trainfraction >=0 and trainfraction <= 1, "invalid training set fraction '%f'" % trainfraction
        assert valfraction >=0 and valfraction <= 1, "invalid validation set fraction '%f'" % valfraction
        assert testfraction >=0 and testfraction <= 1, "invalid test set fraction '%f'" % testfraction
        assert abs(trainfraction + valfraction + testfraction - 1) < 1E-6, "fractions must sum to one"
        
        idx = self.index()
        (testidx, validx, trainidx) = dividelist(idx, (testfraction, valfraction, trainfraction))
            
        trainset = self.clone().index(trainidx)
        if trainsuffix and trainset.id():
            trainset.id(trainset.id() + trainsuffix)
        
        valset = self.clone().index(validx)
        if valsuffix and valset.id():
            valset.id(valset.id() + valsuffix)
        
        testset = self.clone().index(testidx)
        if testsuffix and testset.id():
            testset.id(testset.id() + testsuffix)
                
        return (trainset,valset,testset) if testfraction!=0 else (trainset, valset)

    def split(self, size):
        """Split the dataset into two datasets, one of length size, the other of length len(self)-size"""
        assert isinstance(size, int) and size>=0 and size<len(self)
        return self.partition(size/len(self), (len(self)-size)/len(self), 0, '', '', '')

    def even_split(self):
        """Split the dataset into two datasets, each half the size of the dataset.  If the dataset length is odd, then one element will be dropped"""
        return self.chunks((len(self)//2, len(self)//2, len(self)%2))[0:2]
        
    def streaming_map(self, mapper, accepter=None, bufsize=1024):
        """Returns a generator that will apply the mapper and yield only those elements that return True from the accepter.  Performs the map in parallel if used in the vipy.globals.parallel context manager"""
        return vipy.parallel.iter(self, mapper=mapper, accepter=accepter, bufsize=bufsize)
        
    def map(self, f_map, strict=True, oneway=False, ordered=False):        
        """Parallel map.

        To perform this in parallel across four threads:

        ```python
        D = vipy.dataset.Dataset(...)
        with vipy.globals.parallel(4):
            D = D.map(lambda v: ...)
        ```

        Args:
            f_map: [lambda] The lambda function to apply in parallel to all elements in the dataset.  This must return a JSON serializable object (or set oneway=True)
            strict: [bool] If true, raise exception on distributed map failures, otherwise the map will return only those that succeeded
            oneway: [bool] If true, do not pass back results unless exception.  This is useful for distributed processing
        
        Returns:
            A `vipy.dataset.Dataset` containing the elements f_map(v).  This operation is order preserving if ordered=True.

        .. note:: 
            - This method uses dask distributed and `vipy.batch.Batch` operations
            - Due to chunking, all error handling is caught by this method.  Use `vipy.batch.Batch` to leverage dask distributed futures error handling.
            - Operations must be chunked and serialized because each dask task comes with overhead, and lots of small tasks violates best practices
            - Serialized results are deserialized by the client and returned a a new dataset
        """
        assert f_map is None or callable(f_map), "invalid map function"

        # Identity
        if f_map is None:
            return self        

        # Parallel map 
        elif vipy.globals.cf() is not None:
            # This will fail on multiprocessing if dataset contains a loader lambda, or any element in the dataset contains a loader.  Use distributed instead
            assert ordered == False, "not order preserving, use localmap()"
            return Dataset(tuple(vipy.parallel.map(f_map, self)), id=self.id()) 
                                              
        # Distributed map
        elif vipy.globals.dask() is not None:
            from vipy.batch import Batch   # requires pip install vipy[all]                
            f_serialize = lambda x: x
            f_deserialize = lambda x: x
            f_oneway = lambda x, oneway=oneway: x if not x[0] or not oneway else (x[0], None)
            f_catcher = lambda f, *args, **kwargs: catcher(f, *args, **kwargs)  # catch exceptions when executing lambda, return (True, result) or (False, exception)
            f = lambda x, f_serializer=f_serialize, f_deserializer=f_deserialize, f_map=f_map, f_catcher=f_catcher, f_oneway=f_oneway: f_serializer(f_oneway(f_catcher(f_map, f_deserializer(x))))  # with closure capture
            
            S = [f_serialize(v) for v in self]  # local load, preprocess and serialize
            B = Batch(chunklist(S, 128), strict=False, warnme=False, minscatter=128)
            S = B.map(lambda X,f=f: [f(x) for x in X]).result()  # distributed, chunked, with caught exceptions, may return empty list
            V = [f_deserialize(x) for s in S for x in s]  # Local deserialization and chunk flattening
            
            # Error handling
            (good, bad) = ([r for (b,r) in V if b], [r for (b,r) in V if not b])  # catcher returns (True, result) or (False, exception string)
            if len(bad)>0:
                log.warning('Exceptions in distributed processing:\n%s\n\n[vipy.dataset.Dataset.map]: %d/%d items failed' % (str(bad), len(bad), len(self)))
                if strict:
                    raise ValueError('exceptions in distributed processing')
            return Dataset(good, id=self.id()) if not oneway else None

        # Local map
        else:
            return self.localmap(f_map)
        
    def localmap(self, f):
        """A map performed without any parallel processing"""
        return Dataset([f(x) for x in self], id=self.id())  # triggers load into memory        

    def zip(self, iter):
        """Returns a new dataset constructed by applying the callable on elements from zip(self,iter)"""
        return Dataset(zip(self,iter))
    
    def sort(self, f):
        """Sort the dataset in-place using the sortkey lambda function f

        To perform a sort of the dataset using some property of the instance, such as the object category (e.g. for vipy.image.ImageCategory) 

        ```python
        dataset.sort(lambda im: im.category())
        ```
        """
        idx = self.index()
        return self.index( [idx[j] for (j,x) in sorted(zip(range(len(self)), self.tuple(f)), key=lambda x: x[1])] )

    
    @staticmethod
    def uniform_shuffler(D):
        """A uniform shuffle on the dataset elements.  Iterable access will be slow due to random access"""
        idx = D.index()
        random.shuffle(idx)
        return D.index(idx)

    @staticmethod
    def streaming_shuffler(D):
        """A uniform shuffle (approximation) on the dataset elements for iterable access only"""
        assert D._idx is None, "streaming only"
        
        try_import('datasets', 'datasets'); from datasets import Dataset as HuggingfaceDataset;
        
        if isinstance(D._ds, (list, tuple)):
            D._ds = list(D._ds)
            random.shuffle(D._ds)  # in-place shuffle objects
                
        elif isinstance(D._ds, HuggingfaceDataset):
            # Special case: Arrow backed dataset            
            D._ds = D._ds.to_iterable_dataset()  # no random access
            D._ds.shuffle()  # approximate shuffling for IterableDataset is much more efficient for __iter__
        else:
            raise ValueError('shuffle error')
        return D
    
    @staticmethod
    def identity_shuffler(D):
        """Shuffler that does nothing"""
        return D


class Paged(Dataset):
    """ Paged dataset.

    A paged dataset is a dataset of length N=M*P constructed from M archive files (the pages) each containing P elements (the pagesize).  
    The paged dataset must be constructed with tuples of (pagesize, filename).  
    The loader will fetch, load and cache the pages on demand using the loader, preserving the most recently used cachesize pages

    ```python
    D = vipy.dataset.Paged([(64, 'archive1.pkl'), (64, 'archive2.pkl')], lambda x,y: ivy.load(y))
    ```

    .. note :: Shuffling this dataset is biased.  Shuffling will be performed to mix the indexes, but not uniformly at random.  The goal is to preserve data locality to minimize cache misses.
    """
    
    def __init__(self, pagelist, loader, id=None, strict=True, index=None, cachesize=32):        
        super().__init__(dataset=pagelist,
                         id=id,
                         loader=loader).index(index if index else list(range(sum([p[0] for p in pagelist]))))

        assert callable(loader), "page loader required"
        assert not strict or len(set([x[0] for x in self._ds])) == 1  # pagesizes all the same 

        self._cachesize = cachesize
        self._pagecache = {}
        self._ds = list(self._ds)
        self._pagesize = self._ds[0][0]  # (pagesize, pklfile) tuples        

    def shuffle(self, shuffler=None):
        """Permute elements while preserve page locality to minimize cache misses"""
        shuffler = shuffler if shuffler is not None else functools.partial(Paged.chunk_shuffler, chunksize=int(1.5*self._pagesize))
        return shuffler(self)
        
    def __getitem__(self, k):
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(self._idx), "invalid index"
            page = self._idx[int(k)] // self._pagesize
            if page not in self._pagecache:
                self._pagecache[page] = self._loader(*self._ds[page])  # load and cache new page
                if len(self._pagecache) > self._cachesize:
                    self._pagecache.pop(list(self._pagecache.keys())[0])  # remove oldest
            x = self._pagecache[page][int(k) % self._pagesize]
            return x
        elif isinstance(k, slice):
            return [self[i] for i in range(len(self))[k.start if k.start else 0:k.stop if k.stop else len(self):k.step if k.step else 1]]  # expensive
        else:
            raise ValueError('invalid index type "%s"' % type(k))            

    def flush(self):
        self._pagecache = {}
        return self

    @staticmethod
    def chunk_shuffler(D, chunker, chunksize=64):
        """Split dataset into len(D)/chunksize non-overlapping chunks with some common property returned by chunker, shuffle chunk order and shuffle within chunks.  

           - If chunksize=1 then this is equivalent to uniform_shuffler
           - chunker must be a callable of some property that is used to group into chunks
            
        """
        assert callable(chunker)
        return D.randomize().sort(chunker).index([i for I in shufflelist([shufflelist(I) for I in chunkgenbysize(D.index(), chunksize)]) for i in I])
    
    
class Union(Dataset):
    """vipy.dataset.Union() class
    
    Common class to manipulate groups of vipy.dataset.Dataset objects in parallel

    Usage:
    
        >>> cifar10 = vipy.dataset.registry('cifar10')
        >>> mnist = vipy.dataset.registry('mnist')
        >>> dataset = vipy.dataset.Union(mnist, cifar10)
        >>> dataset = mnist | cifar10

    Args:
        Datasets 
    """

    __slots__ = ('_id', '_ds', '_idx', '_loader', '_type')    
    def __init__(self, *args, **kwargs):
        assert all(isinstance(d, (Dataset, )) for d in args), "invalid datasets"
        
        datasets = [d for d in args]  # order preserving
        assert all([isinstance(d, Dataset) for d in datasets]), "Invalid datasets '%s'" % str([type(d) for d in datasets])

        datasets = [j for i in datasets for j in (i.datasets() if isinstance(i, Union) else (i,))]  # flatten unions        
        self._ds = datasets
        self._idx = None
        self._id = kwargs['id'] if 'id' in kwargs else None

        self._loader = None  # individual datasets have loaders
        self._type = None

    def is_streaming(self):
        return self._idx is None and all(d.is_streaming() for d in self.datasets())

    def __len__(self):
        return sum(d.__len__() for d in self.datasets()) if self._idx is None else len(self._idx)
    
    def __iter__(self):
        if self.is_streaming():
            k = -1
            iter = [d.__iter__() for d in self.datasets()]  # round-robin

            for m in range(len(self.datasets())):
                try:
                    while True:
                        k = (k + 1) % len(iter)                                    
                        yield next(iter[k])  # assumes ordered
                except StopIteration:
                    iter.pop(k)
                    k -= 1
            
        else:
            self.index()  # force random access                    
            for (i,j) in self._idx:
                yield self._ds[i][j]  # random access (slower)                

    def __getitem__(self, k):
        self.index()  # force random access        
        if isinstance(k, (int, np.uint64)):
            assert abs(k) < len(self._idx), "invalid index"
            (i,j) = self._idx[int(k)]            
            return self._ds[i][j]
        elif isinstance(k, slice):
            return [self._ds[i][j] for (i,j) in self._idx[k.start:k.stop:k.step]]
        else:
            raise ValueError('invalid index type "%s"' % type(k))

    def __repr__(self):
        fields = ['id=%s' % truncate_string(self.id(), maxlen=64)] if self.id() else []
        fields += ['len=%d' % len(self)]
        fields += ['union=%s' % str(tuple([truncate_string(d.id(), maxlen=80) for d in self._ds]))]
        return str('<vipy.dataset.Dataset: %s>' % (', '.join(fields)))

    def index(self, index=None, strict=False):
        """Update the index, useful for filtering of large datasets"""
        if index is not None:
            self._idx = index
            return self
        if self._idx is None:
            # Index on-demand: zipped (dataset index, element index) tuples, in round-robin dataset order [(0,0),(1,0),...,(0,n),(1,n),...]            
            lengths = [len(d) for d in self.datasets()]            
            self._idx = [c for r in [[(i,j) for i in range(len(self.datasets()))] for j in range(max(lengths))] for c in r if c[1]<lengths[c[0]]]
        return self._idx
    
    def clone(self, deep=False):
        """Return a copy of the dataset object"""
        D = super().clone(deep=deep)
        D._ds =  [d.clone(deep=deep) for d in D._ds]
        return D
    
    def datasets(self):
        """Return the dataset union elements, useful for generating unions of unions"""
        return list(self._ds)

    def shuffle(self, shuffler=None):
        """Permute elements in this dataset uniformly at random in place using the best shuffler for the dataset structure"""
        shuffler = shuffler if shuffler is not None else (Union.streaming_shuffler if self.is_streaming() else Dataset.uniform_shuffler)
        return shuffler(self)
    
    @staticmethod
    def streaming_shuffler(D):
        """A uniform shuffle (approximation) on the dataset elements for iterable access only"""
        assert D._idx is None, "iterable dataset only"
        D._ds = [Dataset.streaming_shuffler(d) for d in D._ds]  # shuffle dataset shards
        random.shuffle(D._ds)  # shuffle union order
        return D
    

def registry(name=None, datadir=None, freeze=True, clean=False, download=False, split='train'):
    """Common entry point for loading datasets by name.

    Usage:
    
        >>> trainset = vipy.dataset.registry('cifar10', split='train')             # return a training split
        >>> valset = vipy.dataset.registry('cifar10:val', datadir='/tmp/cifar10')  # download to a custom location
        >>> datasets = vipy.dataset.registry(('cifar10:train','cifar100:train'))   # return a union
        >>> vipy.dataset.registry()                                                # print allowable datasets

    Args:
       name [str]: The string name for the dataset.  If tuple, return a `vipy.dataset.Union`.  If None, return the list of registered datasets.  Append name:train, name:val, name:test to output the requested split, or use the split keyword.
       datadir [str]: A path to a directory to store data.  Defaults to environment variable VIPY_DATASET_REGISTRY_HOME (then VIPY_CACHE if not found).  Also uses HF_HOME for huggingface datasets.  Datasets will be stored in datadir/name
       freeze [bool]:  If true, disable reference cycle counting for the loaded object (which will never contain cycles anyway) 
       clean [bool]: If true, force a redownload of the dataset to correct for partial download errors
       download [bool]: If true, force a redownload of the dataset to correct for partial download errors.  This is a synonym for clean=True
       split [str]: return 'train', 'val' or 'test' split.  If None, return (trainset, valset, testset) tuple

    Datasets:
       'mnist','cifar10','cifar100','caltech101','caltech256','oxford_pets','sun397', 'food101','stanford_dogs',
       'flickr30k','oxford_fgvc_aircraft','oxford_flowers_102','eurosat','d2d','ethzshapes','coil100','kthactions',
       'yfcc100m','yfcc100m_url','tiny_imagenet','coyo300m','coyo700m','pascal_voc_2007','coco_2014', 'ava',
       'activitynet', 'open_images_v7', 'imagenet', 'imagenet21k', 'visualgenome' ,'widerface','meva_kf1',
       'objectnet','lfw','inaturalist_2021','kinetics','hmdb','places365','ucf101','lvis','kitti',
       'imagenet_localization','laion2b','datacomp_1b','imagenet2014_det','imagenet_faces','youtubeBB',
       'pip_370k','pip_175k','cap','cap_pad','cap_detection','tiny_virat'

    Returns:
       (trainset, valset, testset) tuple where each is a `vipy.dataset.Dataset` or None, or a single split if name has a ":SPLIT" suffix or split kwarg provided
    """
    
    import vipy.data

    datasets = ('mnist','cifar10','cifar100','caltech101','caltech256','oxford_pets','sun397', 'stanford_dogs','coil100',
                'flickr30k','oxford_fgvc_aircraft','oxford_flowers_102', 'food101', 'eurosat','d2d','ethzshapes','kthactions',
                'yfcc100m','yfcc100m_url','tiny_imagenet','coyo300m','coyo700m','pascal_voc_2007','coco_2014', 'ava',
                'activitynet','open_images_v7','imagenet','imagenet21k','visualgenome','widerface', 'youtubeBB',
                'objectnet','lfw','inaturalist_2021','kinetics','hmdb','places365','ucf101','kitti','meva_kf1',
                'lvis','imagenet_localization','laion2b','datacomp_1b','imagenet2014_det','imagenet_faces',
                'pip_175k','pip_370k','cap','cap_pad','cap_detection','tiny_virat')  # Add to docstring too...
    
    if name is None:
        return tuple(sorted(datasets))
    if isinstance(name, (tuple, list)):
        assert all(n.startswith(datasets) for n in name)
        assert split is not None or all(':' in n for n in name)
        return Union(*(registry(n, datadir=datadir, freeze=freeze, clean=clean, download=download, split=split) for n in name))    
    
    (name, split) = name.split(':',1) if name.count(':')>0 else (name, split)
    if name not in datasets:
        raise ValueError('unknown dataset "%s" - choose from "%s"' % (name, ', '.join(sorted(datasets))))
    if split not in [None, 'train', 'test', 'val']:
        raise ValueError('unknown split "%s" - choose from "%s"' % (split, ', '.join([str(None), 'train', 'test', 'val'])))

    datadir = remkdir(datadir if datadir is not None else (env('VIPY_DATASET_REGISTRY_HOME') if 'VIPY_DATASET_REGISTRY_HOME' in env() else cache()))
    namedir = Path(datadir)/name    
    if (clean or download) and name in datasets and os.path.exists(namedir):
        log.info('Removing cached dataset "%s"' % namedir)
        shutil.rmtree(namedir)  # delete cached subtree to force redownload ...
        
    if freeze:
        gc.disable()
        
    (trainset, valset, testset) = (None, None, None)    
    if name == 'mnist':
        (trainset, testset) = vipy.data.hf.mnist()        
    elif name == 'cifar10':
        (trainset, testset) = vipy.data.hf.cifar10()        
    elif name == 'cifar100':
        (trainset, testset) = vipy.data.hf.cifar100()        
    elif name == 'caltech101':
        trainset = vipy.data.caltech101.Caltech101(namedir)        
    elif name == 'caltech256':
        trainset = vipy.data.caltech256.Caltech256(namedir)
    elif name == 'oxford_pets':
        (trainset, testset) = vipy.data.hf.oxford_pets()
    elif name == 'sun397':
        (trainset, valset, testset) = vipy.data.hf.sun397()
    elif name == 'stanford_dogs':
        trainset = vipy.data.stanford_dogs.StanfordDogs(namedir)
    elif name == 'food101':
        trainset = vipy.data.food101.Food101(namedir)
    elif name == 'eurosat':
        trainset = vipy.data.eurosat.EuroSAT(namedir)
    elif name == 'd2d':
        trainset = vipy.data.d2d.D2D(namedir)
    elif name == 'coil100':
        trainset = vipy.data.coil100.COIL100(namedir)
    elif name == 'kthactions':
        (trainset, testset) = vipy.data.kthactions.KTHActions(namedir).split()
    elif name == 'ethzshapes':
        trainset = vipy.data.ethzshapes.ETHZShapes(namedir)        
    elif name == 'flickr30k':
        trainset = vipy.data.hf.flickr30k()
    elif name == 'oxford_fgvc_aircraft':
        trainset = vipy.data.hf.oxford_fgvc_aircraft()
    elif name == 'oxford_flowers_102':
        trainset = vipy.data.oxford_flowers_102.Flowers102(namedir)
    elif name == 'yfcc100m':
        (trainset, _, valset, _) = vipy.data.hf.yfcc100m()  
    elif name == 'yfcc100m_url':
        (_, trainset, _, valset) = vipy.data.hf.yfcc100m()  
    elif name == 'tiny_imagenet':
        (trainset, valset) = vipy.data.hf.tiny_imagenet()
    elif name == 'coyo300m':
        trainset = vipy.data.hf.coyo300m()
    elif name == 'coyo700m':
        trainset = vipy.data.hf.coyo700m()
    elif name == 'datacomp_1b':
        trainset = vipy.data.hf.datacomp_1b()
    elif name == 'pascal_voc_2007':
        (trainset, valset, testset) = vipy.data.hf.pascal_voc_2007()
    elif name == 'coco_2014':
        trainset = vipy.data.coco.COCO_2014(namedir)
    elif name == 'ava':
        ava = vipy.data.ava.AVA(namedir)
        (trainset, valset) = (ava.trainset(), ava.valset())
    elif name == 'activitynet':
        activitynet = vipy.data.activitynet.ActivityNet(namedir)  # ActivityNet 200
        (trainset, valset, testset) = (activitynet.trainset(), activitynet.valset(), activitynet.testset())
    elif name == 'open_images_v7':
        trainset = vipy.data.openimages.open_images_v7(namedir)
    elif name == 'imagenet':
        imagenet = vipy.data.imagenet.Imagenet2012(namedir)
        (trainset, valset) = (imagenet.classification_trainset(), imagenet.classification_valset())
    elif name == 'imagenet_faces':
        trainset = vipy.data.imagenet.Imagenet2012(Path(datadir)/'imagenet').faces()
    elif name == 'imagenet21k':
        trainset = vipy.data.imagenet.Imagenet21K(namedir)
    elif name == 'visualgenome':
        trainset = vipy.data.visualgenome.VisualGenome(namedir)  # visualgenome-1.4
    elif name == 'widerface':
        trainset = vipy.data.widerface.WiderFace(namedir, split='train')
        valset = vipy.data.widerface.WiderFace(namedir, split='val')
        testset = vipy.data.widerface.WiderFace(namedir, split='test')                                      
    elif name == 'objectnet':
        assert split is None or split == 'test', "objectnet is a test set"
        testset = vipy.data.objectnet.Objectnet(namedir)
    elif name == 'lfw':
        trainset = vipy.data.lfw.LFW(namedir)
    elif name == 'inaturalist_2021':
        dataset = vipy.data.inaturalist.iNaturalist2021(namedir)
        (trainset, valset) = (dataset.trainset(), dataset.valset())
    elif name == 'kinetics':
        dataset = vipy.data.kinetics.Kinetics700(namedir)  # Kinetics700
        (trainset, valset, testset) = (dataset.trainset(), dataset.valset(), dataset.testset())
    elif name == 'hmdb':
        trainset = vipy.dataset.Dataset(vipy.data.hmdb.HMDB(namedir).dataset(), id='hmdb')
    elif name == 'places365':
        places = vipy.data.places.Places365(namedir)
        (trainset, valset) = (places.trainset(), places.valset())
    elif name == 'ucf101':
        trainset = vipy.data.ucf101.UCF101(namedir)
    elif name == 'kitti':
        trainset = vipy.data.kitti.KITTI(namedir, split='train')
        testset = vipy.data.kitti.KITTI(namedir, split='test')                                      
    elif name == 'lvis':
        lvis = vipy.data.lvis.LVIS(namedir)
        (trainset, valset) = (lvis.trainset(), lvis.valset())
    elif name == 'imagenet_localization':
        trainset = vipy.data.imagenet.Imagenet2012(Path(datadir)/'imagenet').localization_trainset()
    elif name == 'imagenet2014_det':
        imagenet2014_det = vipy.data.imagenet.Imagenet2014_DET(namedir)
        (trainset, valset, testset) = (imagenet2014_det.trainset(), imagenet2014_det.valset(), imagenet2014_det.testset())
    elif name == 'laion2b':
        trainset = vipy.data.hf.laion2b()
    elif name == 'youtubeBB':
        trainset = vipy.data.youtubeBB.YoutubeBB(namedir)
    elif name == 'meva_kf1':
        trainset = vipy.data.meva.KF1(namedir).dataset()  # consider using "with vipy.globals.multiprocessing(pct=0.5):"
    elif name == 'pip_175k':
        trainset = vipy.data.pip.PIP_175k(namedir)
    elif name == 'pip_370k':
        trainset = vipy.data.pip.PIP_370k_stabilized(namedir)
    elif name == 'cap':
        trainset = vipy.data.cap.CAP_classification_clip(namedir)
    elif name == 'cap_pad':
        trainset = vipy.data.cap.CAP_classification_pad(namedir)        
    elif name == 'cap_detection':
        trainset = vipy.data.cap.CAP_detection(namedir)
    elif name == 'tiny_virat':
        dataset = vipy.data.tiny_virat.TinyVIRAT(namedir)
        (trainset, valset, testset) = (dataset.trainset(), dataset.valset(), dataset.testset())
    else:
        raise ValueError('unknown dataset "%s" - choose from "%s"' % (name, ', '.join(sorted(datasets))))
    
    if freeze:
        gc.enable()
        gc.collect()
        gc.freeze()  # python-3.7

    if split == 'train':
        return trainset
    elif split == 'val':
        return valset
    elif split == 'test':
        return testset
    else:
        return (trainset, valset, testset)
