import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir, chunklistbysize, listpkl, filetail, filebase, tempdir
from itertools import repeat
import dill
dill.extend(False)  # https://github.com/uqfoundation/dill/issues/383
try_import('dask', 'dask distributed')
from dask.distributed import as_completed, wait
try_import('torch', 'torch');  import torch
import dask
import dask.config
from dask.distributed import Client
from dask.distributed import as_completed, wait
from dask.distributed import get_worker         
import numpy as np
import tempfile
import warnings
import vipy.globals
import hashlib
import uuid
import shutil
dill.extend(True)  # https://github.com/uqfoundation/dill/issues/383


class Dask(object):
    """Dask distributed client"""

    def __init__(self, num_processes=None, dashboard=False, verbose=False, address=None, num_gpus=None):
        assert address is not None or num_processes is not None or num_gpus is not None, "Invalid input"

        if num_gpus is not None:
            assert num_processes is None, "Cannot specify both num_gpus and num_processes"
            num_processes = num_gpus   # coercing

        self._num_processes = num_processes

        os.environ['DASK_LOGGING__DISTRIBUTED'] = 'warning' if not verbose else 'info'
        env = {'VIPY_BACKEND':'Agg',  # headless 
               'PYTHONOPATH':os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '',
               'PATH':os.environ['PATH'] if 'PATH' in os.environ else '',
               'DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT':"30s",
               'DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT':"30s"}

        if 'VIPY_CACHE' in os.environ:
            env.update({'VIPY_CACHE':os.environ['VIPY_CACHE']})
        if 'VIPY_AWS_ACCESS_KEY_ID' in os.environ:
            env.update({'VIPY_AWS_ACCESS_KEY_ID':os.environ['VIPY_AWS_ACCESS_KEY_ID']})            
        if  'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ:
            env.update({'VIPY_AWS_SECRET_ACCESS_KEY':os.environ['VIPY_AWS_SECRET_ACCESS_KEY']})        
                    
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.CONNECT'.lower():'30s'})
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.TCP'.lower():'30s'})
        dask.config.set({'DISTRIBUTED.DEPLOY.LOST_WORKER_TIMEOUT'.lower():'30s'})
        dask.config.refresh()

        if address is not None:
            # Distributed scheduler
            self._client = Client(name='vipy', address=address)

            # Update key environment variables on remote workers using out of band function (yuck)
            # Make sure that any environment variables are accessible on all machines!  (e.g. VIPY_CACHE)
            # If not, then you need to unset these variables from os.environ prior to calling Dask()
            def _f_setenv_remote(localenv):
                import os; os.environ.update(localenv)

            localenv = {k:v for (k,v) in os.environ.items() if k.startswith('VIPY_')}
            localenv.update( {'VIPY_BACKEND':'Agg'} )
            self._client.run(lambda env=localenv: _f_setenv_remote(env))

        else:
            # Local scheduler
            self._client = Client(name='vipy',
                                  address=address,  # to connect to distributed scheduler HOSTNAME:PORT
                                  scheduler_port=0,   # random
                                  dashboard_address=None if not dashboard else ':0',  # random port
                                  processes=True, 
                                  threads_per_worker=1,
                                  n_workers=num_processes, 
                                  env=env,
                                  direct_to_workers=True, 
                                  #memory_limit='auto',
                                  #silence_logs=20 if verbose else 40, 
                                  local_directory=tempfile.mkdtemp())

        self._num_gpus = num_gpus
        if self._num_gpus is not None:
            assert isinstance(self._num_gpus, int) and self._num_gpus > 0, "Number of GPUs must be >= 0 not '%s'" % (str(self._num_gpus))
            assert self._num_gpus == self._num_processes
            wait([self._client.submit(vipy.globals.gpuindex, k, workers=wid) for (k, wid) in enumerate(self._client.scheduler_info()['workers'].keys())])


    def __repr__(self):
        if self._num_processes is not None or self._num_gpus is not None:
            # Local 
            return str('<vipy.globals.dask: %s%s>' % ('gpus=%d' % self.num_gpus() if self.num_gpus() is not None else 'processes=%d' % self.num_processes(), ', dashboard="%s"' % str(self._client.dashboard_link) if self.has_dashboard() else ''))
        elif self._client is not None:
            # Distributed
            return str('<vipy.globals.dask: %s>' % (str(self._client)))
        else:
            return str('<vipy.globals.dask: shutdown')

    def num_gpus(self):
        return self._num_gpus

    def has_dashboard(self):
        return len(self._client.dashboard_link) > 0 if self._client is not None else False

    def dashboard(self):        
        webbrowser.open(self._client.dashboard_link) if len(self._client.dashboard_link)>0 else None
    
    def num_processes(self):
        return len(self._client.nthreads()) if self._client is not None else 0

    def shutdown(self):
        self._client.close()
        self._num_processes = None
        self._num_gpus = None
        vipy.globals.GLOBAL['DASK_CLIENT'] = None        
        return self

    def client(self):
        return self._client


class Batch():
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

    >>> d = vipy.data.kinetics.Kinetics700('/path/to/kinetics').download().trainset()
    >>> b = vipy.batch.Batch(d, n_processes=32)
    >>> b.map(lambda v: v.download().save())  # will download and clip dataset in parallel

    >>> b.result()  # retrieve results after a sequence of map or filter chains
    >>> list(b)     # equivalent to b.result()

    Args:
        strict: [bool] if distributed processing fails, return None for that element and print the exception rather than raise
        as_completed: [bool] Return the objects to the scheduler as they complete, this can introduce instabilities for large complex objects, use with caution
        ordered: [bool]: If True, then preserve the order of objects in objlist in distributed processing

    """    
             
    def __init__(self, objlist, strict=False, as_completed=False, warnme=False, minscatter=None, ordered=False):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        assert isinstance(objlist, list), "Input must be a list"
        self._batchtype = type(objlist[0]) if len(objlist)>0 else type(None)
        assert all([isinstance(im, self._batchtype) for im in objlist]), "Invalid input - Must be homogeneous list of the same type"                

        # Move this into map and disable using localmap
        if vipy.globals.dask() is None and warnme:
            print('[vipy.batch.Batch]: vipy.batch.Batch() is not set to use parallelism.  This is set using:\n    >>> with vipy.globals.parallel(n) for multi-processing with n processes\n    >>> vipy.globals.parallel(pct=0.8) for multiprocessing that uses a percentage of the current system resources\n    >>> vipy.globals.dask(address="SCHEDULER:PORT") which connects to a Dask distributed scheduler.\n    >>> vipy.globals.noparallel() to completely disable all parallelism.')

        # FIXME: this needs to go into Dask()
        #self._ngpu = ngpu
        #if ngpu is not None:
        #    assert isinstance(ngpu, int) and ngpu > 0, "Number of GPUs must be >= 0 not '%s'" % (str(ngpu))
        #    wait([self._client.submit(vipy.globals.gpuindex, k, workers=wid) for (k, wid) in enumerate(self._client.scheduler_info()['workers'].keys())])

        self._strict = strict
        self._as_completed = as_completed  # this may introduce instabilities for large complex objects, use with caution
        self._minscatter = minscatter
        self._ordered = ordered
        self._objlist = [(k,o) for (k,o) in enumerate(objlist)] if ordered else objlist

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self
    
    def __len__(self):
        return len(self._objlist)

    def __repr__(self):        
        return str('<vipy.batch: type=%s, len=%d%s>' % (str(self._batchtype), len(self), (', client=%s' % str(vipy.globals.dask())) if vipy.globals.dask() is not None else ''))

    def _client(self):
        return vipy.globals.dask()._client if vipy.globals.dask() is not None else None

    def _batch_wait(self, futures):
        try:
            results = []            
            for (k, batch) in enumerate(as_completed(futures, with_results=True, raise_errors=False).batches()):
                for (future, result) in batch:
                    if future.status != 'error':
                        results.append(result)  # not order preserving, will restore order in result()
                    else:
                        if self._strict:
                            typ, exc, tb = result
                            raise exc.with_traceback(tb)
                        else:
                            print('[vipy.batch]: future %s failed with error "%s" - SKIPPING' % (str(future), str(result)))
                        results.append(None)
                    del future, result  # distributed memory cleanup

                # Distributed memory cleanup
                del batch
  
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
            #vipy.globals.dask().shutdown()
            #self._client = None
            return None  
        except:
            # warnings.warn('[vipy.batch]: batch cannot be restarted after exception - Recreate Batch()')                
            raise

    def result(self):
        """Return the result of the batch processing, ordered"""
        if self._ordered:
            objlist = {int(v[0]):v[1] for v in self._objlist if v is not None}
            return [objlist[k] if k in objlist else None for k in range(len(self._objlist))]  # restore order
        else:
            return self._objlist

    def __iter__(self):
        for x in self.result():
            yield x
            
    def map(self, f_lambda, args=None):
        """Run the lambda function on each of the elements of the batch and return the batch object.
        
        >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)]   
        >>> imb = vipy.image.Batch(iml) 
        >>> imb.map(lambda im: im.rgb())  

        The lambda function f_lambda should not include closures.  If it does, construct the lambda with default parameter capture:

        >>> f = lambda x, prm1=42: x+prm1

        instead of:

        >>> prm1 = 42
        >>> f = lambda x: x+prm1

        """
        c = self._client()

        if c is None:
            f_lambda_ordered = (lambda x,f=f_lambda: (x[0], f(x[1]))) if self._ordered else f_lambda
            self._objlist = [f_lambda_ordered(o) for o in self._objlist]  # no parallelism
        else:
            f_lambda_ordered = (lambda x,f=f_lambda: (x[0], f(x[1]))) if self._ordered else f_lambda
            objlist = c.scatter(self._objlist) if (self._minscatter is not None and len(self._objlist) >= self._minscatter) else self._objlist
            self._objlist = self._wait(c.map(f_lambda_ordered, objlist))
        return self

    def filter(self, f_lambda):
        """Run the lambda function on each of the elements of the batch and filter based on the provided lambda keeping those elements that return true 
        """
        c = self._client()
        f_lambda_ordered = (lambda x,f=f_lambda: (x[0], f(x[1]))) if self._ordered else f_lambda

        if c is None:
            self._objlist = [o for o in self._objlist if f_lambda_ordered(o)[1]]  # no parallelism
        else:
            objlist = self._objlist  # original list
            is_filtered = self._wait(c.map(f_lambda_ordered, c.scatter(self._objlist)))  # distributed filter (replaces self._objlist)
            self._objlist = [obj for (f, obj) in zip(is_filtered, objlist) if f[1] is True]  # keep only elements that filter true
        return self
        
    def scattermap(self, f_lambda, obj):
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
        c = self._client()
        f_lambda_ordered = (lambda net,x,f=f_lambda: (x[0], f(net,x[1]))) if self._ordered else (lambda net,x,f=f_lambda: f(net, x))

        if c is None:
            self._objlist = [f_lambda_ordered(obj, o) for o in self._objlist]  # no parallelism
        else:
            objdist = c.scatter(obj, broadcast=True)        
            objlist = c.scatter(self._objlist) if (self._minscatter is not None and len(self._objlist) >= self._minscatter) else self._objlist
            self._objlist = self._wait([c.submit(f_lambda_ordered, objdist, im) for im in objlist])
        return self


