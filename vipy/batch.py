import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir, chunklistbysize
from itertools import repeat
import dill
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

    >>> b.result()  # retrieve results after a sequence of map or filter chains
    
    """    
             
    def __init__(self, objlist, n_processes=None, dashboard=False, ngpu=None, strict=True, as_completed=False):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        assert isinstance(objlist, list), "Input must be a list"
        self._batchtype = type(objlist[0])        
        assert all([isinstance(im, self._batchtype) for im in objlist]), "Invalid input - Must be homogeneous list of the same type"                
        self._objlist = objlist

        if n_processes is not None and ngpu is not None:
            assert n_processes == ngpu, "Number of processes must be equal to the number of GPUs"
        elif n_processes is not None:
            n_processes = ngpu if (ngpu is not None) and isinstance(ngpu, int) and ngpu > 0 else n_processes
            n_processes = vipy.globals.max_workers() if n_processes is None else n_processes      
            if vipy.globals.dask() is None or n_processes != vipy.globals.dask().num_processes():
                assert n_processes is not None, "set vipy.globals.max_workers() or n_processes kwarg"
                vipy.globals.dask(num_processes=n_processes, dashboard=dashboard)
        elif ngpu is not None:
            n_processes = ngpu
            if vipy.globals.dask() is None or n_processes != vipy.globals.dask().num_processes():
                vipy.globals.dask(num_processes=n_processes, dashboard=dashboard)
        else:
            assert vipy.globals.dask() is not None

        self._client = vipy.globals.dask().client()  # shutdown using vipy.globals.dask().shutdown(), or let python garbage collect it
    
        self._ngpu = ngpu
        if ngpu is not None:
            assert isinstance(ngpu, int) and ngpu > 0, "Number of GPUs must be >= 0 not '%s'" % (str(ngpu))
            wait([self._client.submit(vipy.globals.gpuindex, k, workers=wid) for (k, wid) in enumerate(self._client.scheduler_info()['workers'].keys())])

        self._strict = strict
        self._as_completed = as_completed

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self
    
    def __len__(self):
        return len(self._objlist)

    def __repr__(self):        
        return str('<vipy.batch: type=%s, len=%d, dask=%s%s>' % (str(self._batchtype), len(self), str(self._client), (', ngpu=%d' % self._ngpu) if self._ngpu is not None else ''))

    def ngpu(self):
        return self._ngpu

    def info(self):
        return self._client.scheduler_info()

    def shutdown(self):
        vipy.globals.dask().shutdown()

    def n_processes(self):
        return len(self.info()['workers'])

    def _wait(self, futures):
        assert islist(futures) and all([hasattr(f, 'result') for f in futures])
        try:
            f_as_completed = lambda f: as_completed(f) if self._as_completed else f
            f_wait = lambda f: wait(f) if not self._as_completed else f

            results = []            
            f_wait(futures)
            for f in f_as_completed(futures):  
                    try:
                        results.append(f.result())  # not order preserving
                    except:
                        if self._strict:
                            raise
                        else:
                            print('[vipy.batch]: future %s failed with error "%s"' % (str(f), str(f.exception())))
                            results.append(None)
            return results
        except KeyboardInterrupt:
            # warnings.warn('[vipy.batch]: batch cannot be restarted after killing with ctrl-c - You must create a new Batch()')
            vipy.globals.dask().shutdown()
            self._client = None
            return None  
        except:
            # warnings.warn('[vipy.batch]: batch cannot be restarted after exception - Recreate Batch()')                
            raise

        #return [f.result() for f in futures]

    def result(self):
        """Return the result of the batch processing"""
        return self._objlist

    def __iter__(self):
        for im in self._objlist:
            yield im
            
    def product(self, f_lambda, args):
        """Cartesian product of args and batch.  Use this with extreme caution, as the memory requirements may be high."""
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"                        
        c = self.__dict__['_client']
        objlist = c.scatter(self._objlist)        
        self._objlist = self._wait([c.submit(f_lambda, im, *a) for im in objlist for a in args])
        return self

    def map(self, f_lambda, args=None):
        """Run the lambda function on each of the elements of the batch and return the batch object.
        
        If args is provided, then this is a unique argument for the lambda function for each of the elements in the batch, or is broadcastable.
        
        >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)]   
        >>> imb = vipy.image.Batch(iml, n_processes=4) 
        >>> imb.map(lambda im,f: im.saveas(f), args=[('/tmp/out%d.jpg'%k,) for k in range(0,1000)])  
        >>> imb.map(lambda im: im.rgb())  # this is equivalent to imb.rgb()

        The lambda function f_lambda must not include closures.  If it does, construct the batch with tuples (obj,prms).

        """
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"                
        c = self.__dict__['_client']        
        if args is not None:
            if len(self._objlist) > 1:
                assert islist(args) and len(list(args)) == len(self._objlist), "args must be a list of arguments of length %d, one for each element in batch" % len(self._objlist)
                objlist = c.scatter(self._objlist)
                self._objlist = self._wait([c.submit(f_lambda, im, *a) for (im, a) in zip(objlist, args)])
            else:
                assert islist(args), "args must be a list"
                obj = c.scatter(self._objlist[0], broadcast=True)
                self._objlist = self._wait([self.__dict__['_client'].submit(f_lambda, obj, *a) for a in args])
        else:
            self._objlist = self._wait(self.__dict__['_client'].map(f_lambda, self._objlist))
        return self

    def filter(self, f_lambda):
        """Run the lambda function on each of the elements of the batch and filter based on the provided lambda keeping those elements that return true 
        """
        assert self.__dict__['_client'] is not None, "Batch() must be reconstructed after shutdown"        
        c = self.__dict__['_client']
        objlist = self._objlist  # original list
        is_filtered = self._wait(self.__dict__['_client'].map(f_lambda, c.scatter(self._objlist)))  # distributed filter (replaces self._objlist)
        self._objlist = [obj for (f, obj) in zip(is_filtered, objlist) if f is True]  # keep only elements that filter true
        return self
        
    def torch(self):
        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        return torch.cat(self.map(lambda im: im.torch()).result())

    def numpy(self):
        """Convert the batch of N HxWxC images to a NxCxHxW torch tensor"""
        return np.stack(self.map(lambda im: im.numpy()).result())
    
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
        objdist = c.scatter(obj, broadcast=True)        
        self._objlist = self._wait([c.submit(f, objdist, imb) for imb in chunklistbysize(self._objlist, batchsize)])
        return self

    def scattermap(self, f, obj):
        """Scatter obj to all workers, and apply lambda function f(obj, im) to each element in batch
        
           Usage: 
         
           >>> Batch(mylist, ngpu=8).scattermap(lambda net, im: net(im), net).result()
        
           This will scatter the large object net to all workers, and pin it to a specific GPU.  Within the net object, you can call 
           vipy.global.gpuindex() to retrieve your assigned GPU index, which can be used by torch.cuda.device().  Then, the net
           object processes each element in the batch using net according to the lambda, and returns the results.  This function 
           includes ngpu processes, and assumes there are ngpu available on the target machine.  Each net is replicated in a different
           process, so it is the callers responsibility for getting vipy.global.gpuindex() from within the process and setting 
           net to take advantage of this GPU rather than using the default cuda:0.  

        """
        c = self.__dict__['_client']
        objdist = c.scatter(obj, broadcast=True)        
        self._objlist = self._wait([c.submit(f, objdist, im) for im in self._objlist])
        return self

