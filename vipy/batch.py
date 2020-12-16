import os
import sys
from vipy.util import try_import, islist, tolist, tempdir, remkdir, chunklistbysize, listpkl, filetail, filebase, tempdir
from itertools import repeat
import dill
dill.extend(False)  # https://github.com/uqfoundation/dill/issues/383
try_import('dask', 'dask distributed torch')
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
    def __init__(self, num_processes=None, dashboard=False, verbose=False, address=None, num_gpus=None):
        assert address is not None or num_processes is not None or num_gpus is not None, "Invalid input"

        if num_gpus is not None:
            assert num_processes is None, "Cannot specify both num_gpus and num_processes"
            num_processes = num_gpus   # coercing

        self._num_processes = num_processes

        if address is not None:
            # Distributed scheduler
            self._client = Client(name='vipy', address=address)
        else:
            # Local scheduler
            self._client = Client(name='vipy',
                                  address=address,  # to connect to distributed scheduler HOSTNAME:PORT
                                  scheduler_port=0,   # random
                                  dashboard_address=None if not dashboard else ':0',  # random port
                                  processes=True, 
                                  threads_per_worker=1,
                                  n_workers=num_processes, 
                                  env={'VIPY_BACKEND':'Agg',  # headless 
                                       'PYTHONOPATH':os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '',
                                       'PATH':os.environ['PATH'] if 'PATH' in os.environ else '',
                                       #'DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP':"30s",
                                       #'DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT':"30s",
                                       #'DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING':'false',
                                       #'DASK_DISTRIBUTED__ADMIN__TICK__LIMIT':"30s",
                                       #'DASK_DISTRIBUTED__ADMIN__TICK__INTERVAL':"2s",
                                       #'DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT':"30s"
                                   },
                                  direct_to_workers=True,
                                  silence_logs=20 if verbose else 40,  # logging.WARN=30 or logging.ERROR=40 or logging.INFO=20
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
             
    def __init__(self, objlist, strict=True, as_completed=False, checkpoint=False, checkpointdir=None, checkpointfrac=0.1, warnme=True, minscatter=None):
        """Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call
        """
        assert isinstance(objlist, list), "Input must be a list"
        self._batchtype = type(objlist[0]) if len(objlist)>0 else type(None)
        assert all([isinstance(im, self._batchtype) for im in objlist]), "Invalid input - Must be homogeneous list of the same type"                
        self._objlist = [(k,o) for (k,o) in enumerate(objlist)]  # ordered

        assert checkpointfrac > 0.0 and checkpointfrac <= 1.0, "Invalid checkpoint fraction"
        self._checkpointsize = max(1, int(len(objlist) * checkpointfrac))
        super().__init__(checkpointdir)
        self._checkpoint = checkpoint
        if checkpoint:
            as_completed = True  # force self._as_completed=True

        if vipy.globals.dask() is None and warnme:
            print('[vipy.batch.Batch]: vipy.batch.Batch() is not set to use parallelism.  This is set using:\n    >>> vipy.globals.parallel(n) for multi-processing with n processes\n    >>> vipy.globals.parallel(pct=0.8) for multiprocessing that uses a percentage of the current system resources\n    >>> vipy.globals.dask(address="SCHEDULER:PORT") which connects to a Dask distributed scheduler.\n    >>> vipy.globals.noparallel() to completely disable all parallelism.')

        # FIXME: this needs to go into Dask()
        #self._ngpu = ngpu
        #if ngpu is not None:
        #    assert isinstance(ngpu, int) and ngpu > 0, "Number of GPUs must be >= 0 not '%s'" % (str(ngpu))
        #    wait([self._client.submit(vipy.globals.gpuindex, k, workers=wid) for (k, wid) in enumerate(self._client.scheduler_info()['workers'].keys())])

        self._strict = strict
        self._as_completed = as_completed  # this may introduce instabilities for large complex objects, use with caution
        self._minscatter = minscatter

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self
    
    def __len__(self):
        return len(self._objlist)

    def __repr__(self):        
        return str('<vipy.batch: type=%s, len=%d, client=%s>' % (str(self._batchtype), len(self), str(vipy.globals.dask())))

    def _client(self):
        return vipy.globals.dask()._client if vipy.globals.dask() is not None else None

    def _batch_wait(self, futures):
        self._archive_checkpoint()._flush_checkpoint()
        k_checkpoint = 0
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
                    k_checkpoint = k_checkpoint + 1
                    
                    del future, result  # distributed memory cleanup

                # Save intermediate results
                if self._checkpoint and (k_checkpoint > self._checkpointsize):
                    pklfile = os.path.join(remkdir(self._checkpointdir), '%s_%d.pkl' % (self._checkpointid(), k))
                    print('[vipy.batch]: saving checkpoint %s ' % pklfile)
                    vipy.util.save(results[-k_checkpoint:], pklfile)
                    k_checkpoint = 0

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

    def restore(self):
        return self.checkpoint() if self._checkpoint else None

    def result(self):
        """Return the result of the batch processing, ordered"""
        objlist = {int(v[0]):v[1] for v in self._objlist if v is not None}
        return [objlist[k] if k in objlist else None for k in range(len(self._objlist))]  # restore order

    def __iter__(self):
        for x in self.result():
            yield x
            
    def map(self, f_lambda, args=None):
        """Run the lambda function on each of the elements of the batch and return the batch object.
        
        >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)]   
        >>> imb = vipy.image.Batch(iml) 
        >>> imb.map(lambda im: im.rgb())  

        The lambda function f_lambda must not include closures.  If it does, construct the batch with tuples (obj,prms) or with default parameter capture:
        >>> f = lambda x, prm1=1, prm2=2: x+prm1+prm2

        """
        c = self._client()

        if c is None:
            f_lambda_ordered = lambda x,f=f_lambda: (x[0], f(x[1]))                         
            self._objlist = [f_lambda_ordered(o) for o in self._objlist]  # no parallelism
        else:
            f_lambda_ordered = lambda x,f=f_lambda: (x[0], f(x[1]))            
            objlist = c.scatter(self._objlist) if (self._minscatter is not None and len(self._objlist) > self._minscatter) else self._objlist
            self._objlist = self._wait(c.map(f_lambda_ordered, objlist))
        return self

    def filter(self, f_lambda):
        """Run the lambda function on each of the elements of the batch and filter based on the provided lambda keeping those elements that return true 
        """
        c = self._client()
        f_lambda_ordered = lambda x,f=f_lambda: (x[0], f(x[1])) 

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
        f_lambda_ordered = lambda net,x,f=f_lambda: (x[0], f(net,x[1])) 

        if c is None:
            self._objlist = [f_lambda_ordered(obj, o) for o in self._objlist]  # no parallelism
        else:
            objdist = c.scatter(obj, broadcast=True)        
            objlist = c.scatter(self._objlist) if len(self._objlist) > self._minscatter else self._objlist
            self._objlist = self._wait([c.submit(f_lambda_ordered, objdist, im) for im in objlist])
        return self


