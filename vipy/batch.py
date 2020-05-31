import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir, chunklistbysize
from itertools import repeat
try_import('dask', 'dask distributed torch')
from dask.distributed import as_completed, wait
try_import('torch', 'torch');  import torch
import numpy as np
import tempfile
import warnings
import vipy.globals


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
             
    def __init__(self, objlist, n_processes=None, dashboard=False, ngpu=0):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        objlist = tolist(objlist)
        self._batchtype = type(objlist[0])        
        assert all([isinstance(im, self._batchtype) for im in objlist]), "Invalid input - Must be homogeneous list of the same type"                
        self._objlist = objlist
      
        n_processes = ngpu if ngpu > 0 else n_processes
        n_processes = vipy.globals.max_workers() if n_processes is None else n_processes
        assert n_processes is not None, "set vipy.globals.max_workers() or n_processes kwarg"
        if vipy.globals.dask() is None or n_processes != vipy.globals.dask().num_processes():
            vipy.globals.dask(num_processes=n_processes, dashboard=dashboard)
        self._client = vipy.globals.dask().client()  # shutdown using vipy.globals.dask().shutdown(), or let python garbage collect it
    
        self._ngpu = ngpu
        if self._ngpu > 0:
             assert ngpu == n_processes
             wait([self._client.submit(lambda wid: vipy.globals.gpuindex( k ), wid, workers=wid) for (k,wid) in enumerate(self._client.scheduler_info()['workers'].keys())])

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self
    
    def __len__(self):
        return len(self._objlist)

    def __repr__(self):
        return str('<vipy.batch: type=%s, len=%d, procs=%d, gpu=%d>' % (str(self._batchtype), len(self), self.n_processes(), self.ngpu()))

    def ngpu(self):
        return self._ngpu

    def info(self):
        return self._client.scheduler_info()

    def shutdown(self):
        vipy.globals.dask().shutdown()

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
                warnings.warn('[vipy.batch]: batch cannot be restarted after killing with ctrl-c - You must create a new Batch()')
                vipy.globals.dask().shutdown()
                self._client = None
                return None  
            except:
                # warnings.warn('[vipy.batch]: batch cannot be restarted after exception - Recreate Batch()')                
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

    def result(self):
        return self.batch()
    
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
    
    def torchmap(self, f, net):
        """Apply lambda function f prepare data, and pass resulting tensor to data parallel to torch network"""
        assert self.__dict__['_client'] is not None, "Batch() must be reconsructed after shutdown"        

        c = self.__dict__['_client']       
        deviceid = 'cuda' if torch.cuda.is_available and self.ngpu() > 0 else 'cpu'
        device = torch.device(deviceid)

        modeldist = torch.nn.DataParallel(net)
        modeldist = modeldist.to(device)
        
        return [modeldist(t.to(device)) for t in as_completed([c.submit(f, im) for im in self._objlist])]
        
    def chunkmap(self, f, obj, batchsize):
        c = self.__dict__['_client']
        objdist = c.scatter(obj)        
        return self.batch([c.submit(f, objdist, imb) for imb in chunklistbysize(self._objlist, batchsize)])                

    def scattermap(self, f, obj):
        """Scatter obj to all workers, and apply lambda function f(obj, im) to each element in batch
        
           Usage: 
         
           >>> Batch(mylist, ngpu=8).scattermap(lambda net, im: net(im), net).result()
        
           This will scatter the large object net to all workers, and pin it to a specific GPU.  Within the net object, you can call 
           vipy.global.gpuindex() to retrieve your assigned GPU index, which can be used by torch.cuda.device().  Then, the net
           object processes each element in the batch using net according to the lambda, and returns the results.  This function 
           includes ngpu processes, and assumes there are ngpu available on the target machine.  Each net is replicated in a different
           process, so it is the callers responsibility for getting vipy.global.gpuindex() from within the process and setting 
           net to take advantage of this GPU rather than using the defaeult cuda:0.  

        """
        c = self.__dict__['_client']
        objdist = c.scatter(obj)        
        return self.batch([c.submit(f, objdist, im) for im in self._objlist])

