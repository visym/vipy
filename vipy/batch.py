import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir, chunklistbysize, listpkl, filetail, filebase, tempdir
from itertools import repeat
import dill
try_import('dask', 'dask distributed torch')
from dask.distributed import as_completed, wait
try_import('torch', 'torch');  import torch
import numpy as np
import tempfile
import warnings
import vipy.globals
import hashlib
import uuid
import shutil


class Checkpoint(object):
    """Batch checkpoints for long running jobs"""
    def __init__(self, checkpointdir=None):
        if checkpointdir is not None:
            self._checkpointdir = checkpointdir
        elif vipy.globals.cache() is not None:
            self._checkpointdir = os.path.join(vipy.globals.cache(), 'batch')
        else:
            self._checkpointdir = os.path.join(tempdir(), 'batch')

    def checkpoint(self, archiveid=None):
        """Return the last checkpointed result.  Useful for recovering from dask crashes for long jobs."""
        pklfiles = self._list_checkpoints(archiveid)
        if len(pklfiles) > 0:
            print('[vipy.batch]: loading %d checkpoints %s' % (len(pklfiles), str(pklfiles)))
            return [v for f in pklfiles for v in vipy.util.load(f)]
        else:
            return None

    def last_archive(self):
        archivelist = self._list_archives()
        return archivelist[0] if len(archivelist) > 0 else None

    def _checkpointid(self):
        assert self._checkpointdir is not None
        hash_object = hashlib.md5(str(self._checkpointdir).encode())
        return str(hash_object.hexdigest())

    def _list_checkpoints(self, archiveid=None):
        cpdir = os.path.join(self._checkpointdir, archiveid) if (archiveid is not None and self._checkpointdir is not None) else self._checkpointdir
        return sorted([f for f in listpkl(cpdir) if self._checkpointid() in f], key=lambda f: int(filebase(f).split('_')[1])) if cpdir is not None and os.path.isdir(cpdir) else []

    def _list_archives(self):
        return [filetail(d) for d in vipy.util.dirlist_sorted_bycreation(self._checkpointdir)] if self._checkpointdir is not None else []

    def _flush_checkpoint(self):
        for f in self._list_checkpoints():
            if self._checkpointid() in f:
                os.remove(f)
        return self

    def _archive_checkpoint(self):
        archivedir = os.path.join(self._checkpointdir, str(uuid.uuid4().hex))
        for f in self._list_checkpoints():
            if self._checkpointid() in f:
                f_new = os.path.join(remkdir(archivedir), filetail(f))
                print('[vipy.batch]: archiving checkpoint %s -> %s' % (f, f_new))
                shutil.copyfile(f, f_new)
        return self


class Batch(Checkpoint):
    """vipy.batch.Batch class

    This class provides a representation of a set of vipy objects.  All of the object types must be the same.  If so, then an operation on the batch is performed on each of the elements in the batch in parallel.

    Examples:

    >>> b = vipy.batch.Batch([Image(filename='img_%06d.png' % k) for k in range(0,100)])
    >>> b.map(lambda im: im.bgr())  
    >>> b.map(lambda im: np.sum(im.array())) 
    >>> b.map(lambda im, f: im.saveas(f), args=['out%d.jpg' % k for k in range(0,100)])
    
    >>> v = vipy.video.RandomSceneActivity()
    >>> b = vipy.batch.Batch(v, n_processes=16)
    >>> b.map(lambda v,k: v[k], args=[(k,) for k in range(0, len(v))])  # paralle interpolation

    >>> d = vipy.dataset.kinetics.Kinetics700('/path/to/kinetics').download().trainset()
    >>> b = vipy.batch.Batch(d, n_processes=32)
    >>> b.map(lambda v: v.download().save())  # will download and clip dataset in parallel

    >>> b.result()  # retrieve results after a sequence of map or filter chains

    Parameters:
      -strict=False: if distributed processing fails, return None for that element and print the exception rather than raise
      -as_completed=True:  Return the objects to the scheduler as they complete, this can introduce instabilities for large complex objects, use with caution

    """    
             
    def __init__(self, objlist, n_processes=None, dashboard=False, ngpu=None, strict=True, as_completed=False, checkpoint=False, checkpointdir=None, checkpointfrac=0.1):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        assert isinstance(objlist, list), "Input must be a list"
        self._batchtype = type(objlist[0]) if len(objlist)>0 else type(None)
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
            assert vipy.globals.dask() is not None, "Create a global dask scheduler using vipy.globals.dask(num_processes=n) for a given number of workers, or create Batch() with n_processes>0"            

        assert checkpointfrac > 0.0 and checkpointfrac <= 1.0, "Invalid checkpoint fraction"
        self._checkpointsize = max(1, int(len(objlist) * checkpointfrac))
        super().__init__(checkpointdir)
        self._checkpoint = checkpoint
        if checkpoint:
            as_completed = True  # force self._as_completed=True

        self._client = vipy.globals.dask().client()  # shutdown using vipy.globals.dask().shutdown(), or let python garbage collect it
    
        self._ngpu = ngpu
        if ngpu is not None:
            assert isinstance(ngpu, int) and ngpu > 0, "Number of GPUs must be >= 0 not '%s'" % (str(ngpu))
            wait([self._client.submit(vipy.globals.gpuindex, k, workers=wid) for (k, wid) in enumerate(self._client.scheduler_info()['workers'].keys())])

        self._strict = strict
        self._as_completed = as_completed  # this may introduce instabilities for large complex objects, use with caution


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


    def _batch_wait(self, futures):
        self._archive_checkpoint()._flush_checkpoint()
        k_checkpoint = 0
        try:
            results = []            
            for (k, batch) in enumerate(as_completed(futures, with_results=True, raise_errors=False).batches()):
                for (future, result) in batch:
                    if future.status != 'error':
                        results.append(result)  # not order preserving
                    else:
                        if self._strict:
                            typ, exc, tb = result
                            raise exc.with_traceback(tb)
                        else:
                            print('[vipy.batch]: future %s failed with error "%s" - SKIPPING' % (str(future), str(result)))
                        results.append(None)
                    k_checkpoint = k_checkpoint + 1

                # Save intermediate results
                if self._checkpoint and (k_checkpoint > self._checkpointsize):
                    pklfile = os.path.join(remkdir(self._checkpointdir), '%s_%d.pkl' % (self._checkpointid(), k))
                    print('[vipy.batch]: saving checkpoint %s ' % pklfile)
                    vipy.util.save(results[-k_checkpoint:], pklfile)
                    k_checkpoint = 0

            return results

        except KeyboardInterrupt:
            # warnings.warn('[vipy.batch]: batch cannot be restarted after killing with ctrl-c - You must create a new Batch()')
            #vipy.globals.dask().shutdown()
            #self._client = None
            return results
        except:
            # warnings.warn('[vipy.batch]: batch cannot be restarted after exception - Recreate Batch()')                
            raise


    
    def _wait(self, futures):
        assert islist(futures) and all([hasattr(f, 'result') for f in futures])
        if self._as_completed:
            return self._batch_wait(futures)
        
        try:
            results = []            
            wait(futures)
            for f in futures:  
                try:
                    results.append(f.result()) 
                except:
                    if self._strict:
                        raise
                    try:
                        print('[vipy.batch]: future %s failed with error "%s" for batch "%s"' % (str(f), str(f.exception()), str(self)))
                    except:
                        print('[vipy.batch]: future failed')
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


