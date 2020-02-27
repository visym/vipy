import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir
from itertools import repeat
try_import('dask', 'dask distributed torch')
from dask.distributed import Client
from dask.distributed import as_completed, wait
try_import('torch', 'torch');  import torch
import numpy as np
import tempfile
import warnings


class Batch(object):
    """vipy.batch.Batch class

    This class provides a representation of a set of vipy objects.  All of the object types must be the same.  If so, then an operation on the batch is performed on each of the elements in the batch in parallel.

    Examples:

    >>> b = vipy.batch.Batch([Image(filename='img_%06d.png' % k) for k in range(0,100)])
    >>> b.bgr()  # convert all elements in batch to BGR
    >>> b.torch()  # load all elements in batch and convert to torch tensor
    >>> b.map(lambda im: im.bgr())  # equivalent
    >>> b.map(lambda im: np.sum(im.array())) 
    >>> b.map(lambda im, f: im.saveas(f), args=['out%d.jpg' % k for k in range(0,100)])
    
    >>> v = vipy.video.RandomSceneActivity()
    >>> b = vipy.batch.Batch(v, n_processes=16)
    >>> b.map(lambda v,k: v[k], args=[(k,) for k in range(0, len(v))])  # paralle interpolation

    >>> d = vipy.dataset.kinetics.Kinetics700('/path/to/kinetics').download().trainset()
    >>> b = vipy.batch.Batch(d, n_processes=32)
    >>> b.map(lambda v: v.download().save())  # will download and clip dataset in parallel

    """    
             
    def __init__(self, objlist, n_processes=4, dashboard=False):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        objlist = tolist(objlist)
        self._batchtype = type(objlist[0])        
        assert all([isinstance(im, self._batchtype) for im in objlist]), "Invalid input - Must be homogeneous list of the same type"                
        self._objlist = objlist        
        self._client = Client(name='vipy', 
                              scheduler_port=0, 
                              dashboard_address=None if not dashboard else ':0', 
                              processes=True, 
                              threads_per_worker=1, 
                              n_workers=n_processes, 
                              env={'VIPY_BACKEND':'Agg'},
                              direct_to_workers=True,
                              local_directory=tempfile.mkdtemp())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()
        
    def __len__(self):
        return len(self._objlist)

    def __del__(self):
        self.shutdown()
        
    def __repr__(self):
        return str('<vipy.batch: type=%s, len=%d, procs=%d>' % (str(self._batchtype), len(self), self.n_processes()))

    def info(self):
        return self._client.scheduler_info()

    def n_processes(self):
        return len(self.info()['workers'])

    def batch(self, newlist=None):
        if islist(newlist) and not hasattr(newlist[0], 'result'):
            self._objlist = newlist
            return self
        elif islist(newlist) and hasattr(newlist[0], 'result'):
            try:
                wait(newlist)
            except KeyboardInterrupt:
                # warnings.warn('[vipy.batch]: batch cannot be restarted after killing - Recreate Batch()')
                self.shutdown()  # batch must be recreated after ctrl-c
                return None  # is this the right way to handle this??
            except:
                # warnings.warn('[vipy.batch]: batch cannot be restarted after exception - Recreate Batch()')                
                self.shutdown()  # batch must be recreated after ctrl-c                
                raise
            completedlist = [f.result() for f in newlist]
            if isinstance(completedlist[0], self._batchtype):
                self._objlist = completedlist
                return self
            else:
                return completedlist
        elif newlist is None:
            return self._objlist
        else:
            raise ValueError('Invalid input - must be list')
        
    def __iter__(self):
        for im in self._objlist:
            yield im
            
    def __getattr__(self, attr):
        """Call the same method on all Image objects"""
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"                                
        return lambda *args, **kw: self.batch(self.__dict__['_client'].map(lambda im: getattr(im, attr)(*args, **kw), self._objlist))

    def product(self, f_lambda, args, waiting=True):
        """Cartesian product of args and batch, returns an MxN list of N args applied to M batch elements.  Use this with extreme caution, as the memory requirements may be high."""
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"                        
        c = self.__dict__['_client']
        objlist = c.scatter(self._objlist)        
        futures = [c.submit(f_lambda, im, *a) for im in objlist for a in args]
        return self.batch(futures) if waiting else futures
        
    def map(self, f_lambda, args=None):
        """Run the lambda function on each of the elements of the batch. 
        
        If args is provided, then this is a unique argument for the lambda function for each of the elements in the batch, or is broadcastable.
        
        >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)]   
        >>> imb = vipy.image.Batch(iml, n_processes=4) 
        >>> imb.map(lambda im,f: im.saveas(f), args=[('/tmp/out%d.jpg'%k,) for k in range(0,1000)])  
        >>> imb.map(lambda im: im.rgb())  # this is equivalent to imb.rgb()

        """
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"                
        c = self.__dict__['_client']        
        if args is not None:
            if len(self._objlist) > 1:
                assert islist(args) and len(list(args)) == len(self._objlist), "args must be a list of arguments of length %d, one for each element in batch" % len(self._objlist)
                objlist = c.scatter(self._objlist)
                return self.batch([c.submit(f_lambda, im, *a) for (im, a) in zip(objlist, args)])                
            else:
                assert islist(args), "args must be a list"
                obj = c.scatter(self._objlist[0], broadcast=True)
                return self.batch([self.__dict__['_client'].submit(f_lambda, obj, *a) for a in args])
        else:
            return self.batch(self.__dict__['_client'].map(f_lambda, self._objlist))

    def filter(self, f_lambda):
        """Run the lambda function on each of the elements of the batch and filter based on the provided lambda  
        """
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"        
        c = self.__dict__['_client']
        objlist = c.scatter(self._objlist)        
        is_filtered = self.batch(self.__dict__['_client'].map(f_lambda, objlist))
        self._objlist = [obj for (f, obj) in zip(is_filtered, self._objlist) if f is True]
        return self
        
    def torch(self):
        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        return torch.cat(self.map(lambda im: im.torch()))

    def numpy(self):
        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        return np.stack(self.map(lambda im: im.numpy()))
    
    def shutdown(self):
        if '_client' in self.__dict__ and self.__dict__['_client'] is not None:
            # This may still generate some concerning looking execeptions like: 'tornado.iostream.StreamClosedError: Stream is closed'
            # This is a bug with dask, and can be safely ignored ...
            self.__dict__['_client'].shutdown()
        self.__dict__['_client'] = None
    

