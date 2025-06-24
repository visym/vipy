import os
import webbrowser
import tempfile
import builtins
import logging 
import concurrent.futures 


# Global mutable dictionary
GLOBAL = {'DASK_CLIENT': None,   # Global Dask() client for distributed processing
          'CONCURRENT_FUTURES':None,  # global futures client 
          'CACHE':os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None,   # Cache directory for vipy.video and vipy.image donwloads
          'LOGGER':logging.getLogger('vipy'),     # The global logger
          'DEBUG':False, # globally enable debugging flags
          'GUI':{'escape':False},
          'AWS':{'AWS_ACCESS_KEY_ID':os.environ['VIPY_AWS_ACCESS_KEY_ID'] if 'VIPY_AWS_ACCESS_KEY_ID' in os.environ else None,
                 'AWS_SECRET_ACCESS_KEY':os.environ['VIPY_AWS_SECRET_ACCESS_KEY'] if 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ else None,
                 'AWS_SESSION_TOKEN':os.environ['VIPY_AWS_SESSION_TOKEN'] if 'VIPY_AWS_SESSION_TOKEN' in os.environ else None},
          'LATEX':os.environ['VIPY_LATEX'] if 'VIPY_LATEX' in os.environ else None}

log = GLOBAL['LOGGER']

def logger():
    return GLOBALS['LOGGER']


class Dask(object):
    """Dask distributed client"""
    
    def __init__(self, num_workers=None, threaded=True, dashboard=False, verbose=False, address=None):
        from vipy.util import try_import
        try_import('dask', 'dask distributed'); import dask, dask.distributed;
    
        assert address is not None or num_workers is not None, "Invalid input"

        self._num_workers = num_workers
        self._has_dashboard = dashboard
        
        # Dask configuration: https://docs.dask.org/en/latest/configuration.html
        # - when using vipy.dataset.Dataset minibatch iterator, large minibatches can result in a warning about large graphs
        # - The end user can set these environemnt variables, and will only be overwritten with defaults here if not provided
        if 'DASK_LOGGING__DISTRIBUTED' not in os.environ:
            os.environ['DASK_LOGGING__DISTRIBUTED'] = 'warning' if not verbose else 'info'
        if 'DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT' not in os.environ:
            os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT'] = "30s"
        if 'DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP' not in os.environ:
            os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP'] = "30s"
        if 'DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT' not in os.environ:
            os.environ['DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT'] = "30s"
        if 'DASK_DISTRIBUTED__COMM__RETRY__COUNT' not in os.environ:
            os.environ['DASK_DISTRIBUTED__COMM__RETRY__COUNT'] = "10"        
        if 'DASK_ADMIN_LARGE_GRAPH_WARNING_THREHSOLD' not in os.environ:
            os.environ['DASK_ADMIN_LARGE_GRAPH_WARNING_THREHSOLD'] = "50MB"        

        dask.config.refresh()
        
        dask.config.set({'DISTRIBUTED.COMM.RETRY.COUNT'.lower():int(os.environ['DASK_DISTRIBUTED__COMM__RETRY__COUNT'])})
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.CONNECT'.lower():os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT']})
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.TCP'.lower():os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP']})
        dask.config.set({'DISTRIBUTED.DEPLOY.LOST_WORKER_TIMEOUT'.lower():os.environ['DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT']})        
        dask.config.set({"distributed.admin.large-graph-warning-threshold": os.environ['DASK_ADMIN_LARGE_GRAPH_WARNING_THREHSOLD']})
        
        
        # Worker env
        env = {'VIPY_BACKEND':'Agg',  # headless in workers
               'PYTHONOPATH':os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '',
               'PATH':os.environ['PATH'] if 'PATH' in os.environ else ''}

        if 'VIPY_CACHE' in os.environ:
            env.update({'VIPY_CACHE':os.environ['VIPY_CACHE']})
        if 'VIPY_AWS_ACCESS_KEY_ID' in os.environ:
            env.update({'VIPY_AWS_ACCESS_KEY_ID':os.environ['VIPY_AWS_ACCESS_KEY_ID']})            
        if 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ:
            env.update({'VIPY_AWS_SECRET_ACCESS_KEY':os.environ['VIPY_AWS_SECRET_ACCESS_KEY']})        
                    
        for (k,v) in os.environ.items():
            if k.startswith('DASK_'):
                env[k] = v
    
        if address is not None:
            # Distributed scheduler
            self._client = dask.distributed.Client(name='vipy', address=address)

            # Update key environment variables on remote workers using out of band function (yuck)
            # Make sure that any environment variables are accessible on all machines!  (e.g. VIPY_CACHE)
            # If not, then you need to unset these variables from os.environ prior to calling Dask()
            def _f_setenv_remote(localenv):
                import os; os.environ.update(localenv)

            localenv = {k:v for (k,v) in os.environ.items() if k.startswith('VIPY_')}
            localenv.update( {'VIPY_BACKEND':'Agg'} )
            self._client.run(lambda env=localenv: _f_setenv_remote(env))

        else:
            kwargs = {'name':'vipy',
                      'address':address,  # to connect to distributed scheduler HOSTNAME:PORT
                      'scheduler_port':0,   # random
                      'dashboard_address':None if not dashboard else ':0',  # random port
                      'processes':not threaded,
                      'threads_per_worker':1,
                      'n_workers':num_workers,
                      'local_directory':tempfile.mkdtemp()}
            kwargs.update({'env':env, 'direct_to_workers':True} if not threaded else {})
            
            # Local scheduler
            self._client = dask.distributed.Client(**kwargs)
            

    def __repr__(self):
        if self._num_workers is not None:
            # Local 
            return str('<vipy.globals.Dask: %s%s>' % ('workers=%d' % self.num_workers(), ', dashboard=%s' % str(self._client.dashboard_link) if self._has_dashboard else ''))
        elif self._client is not None:
            # Distributed
            return str('<vipy.globals.Dask: %s>' % (str(self._client)))
        else:
            return str('<vipy.globals.Dask: shutdown')

    def num_workers(self):
        return self._num_workers

    def shutdown(self):
        self._client.close()
        self._num_workers = None
        GLOBAL['DASK_CLIENT'] = None        
        return self

    def client(self):
        return self._client



def cache(cachedir=None):
    """The cache is the location that URLs are downloaded to on your system.  This can be set here, or with the environment variable VIPY_CACHE

    >>> vipy.globals.cache('/path/to/.vipy')
    >>> cachedir = vipy.globals.cache()

    Args:
        cachedir:  the location to store cached files when downloaded.  Can also be set using the VIPY_CACHE environment variable.  if none, return the current cachedir
    
    Returns:
        The current cachedir if cachedir=None else None
    
    """
    if cachedir is not None:
        from vipy.util import remkdir        
        os.environ['VIPY_CACHE'] = remkdir(cachedir)
        GLOBAL['CACHE'] = cachedir
    return os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None
    

def _user_hit_escape(b=None):
    """Did the user hit the escape key?  Useful for matplotlib GUI to stop displaying video"""
    if b is None:
        if GLOBAL['GUI']['escape']:
            GLOBAL['GUI']['escape'] = False  # toggle it
            return True
        else:
            return False
    else:
        # Set in vipy.gui.using_matplotlib.escape_to_exit()
        assert isinstance(b, bool)
        GLOBAL['GUI']['escape'] = b  

        
def cf(num_workers=None, threaded=True):
    if num_workers is not None:
        if GLOBAL['CONCURRENT_FUTURES']:
            GLOBAL['CONCURRENT_FUTURES'].shutdown()
            
        GLOBAL['CONCURRENT_FUTURES'] = (concurrent.futures.ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='vipy') if threaded else
                                        concurrent.futures.ProcessPoolExecutor(max_workers=num_workers))
    return GLOBAL['CONCURRENT_FUTURES']
                                                 
    
def dask(num_workers=None, dashboard=False, address=None, pct=None, threaded=True):
    """Return the current Dask client, can be accessed globally for parallel processing.
    
    Args:
        pct: float in [0,1] the percentage of the current machine to use
        address:  the dask scheduler of the form 'HOSTNAME:PORT'
        num_workers:  the number of prpcesses to use on the current machine
        dashboard: [bool] whether to inialize the dask client with a web dashboard
        threaded: [bool] if true, create threaded workers intead of processes

    Returns:
        The `vipy.batch.Dask` object pointing to the Dask Distrbuted object
    """
    if pct is not None:
        assert pct > 0 and pct <= 1
        num_workers = int(pct*os.cpu_count())
    if address is not None or num_workers is not None:
        if GLOBAL['DASK_CLIENT']:
            GLOBAL['DASK_CLIENT'].shutdown()
        GLOBAL['DASK_CLIENT'] = Dask(num_workers, threaded=threaded, dashboard=dashboard, verbose=False, address=address)        
    return GLOBAL['DASK_CLIENT']


def parallel(workers=None, pct=None, threaded=True):
    """Enable parallel processing with n>=1 processes or a percentage of system core (pct in [0,1])  .

    This can be be used as a context manager
    
    >>> with vipy.globals.parallel(n=4):
    >>>     vipy.batch.Batch(...)

    or using the global variables:

    >>> vipy.globals.parallel(n=4):
    >>> vipy.batch.Batch(...)
    >>> vipy.globals.noparallel()
    
    To check the current parallelism level:
    
    >>> num_workers = vipy.globals.parallel().num_workers()

    To run with a dask scheduler:
    
    >>> with vipy.globals.parallel(scheduler='10.0.1.1:8585')
    >>>    vipy.batch.Batch(...)

    Args:
        workers: [int] number of parallel workers
        pct: [float] the percentage [0,1] of system cores to dedicate to parallel processing
        threaded [bool]: if false, use processes (not recommended, since vipy parallel processing usually releases the GIL)
    """

    class Parallel():
        def __init__(self, workers):
            self._workers = workers
            self._threaded = threaded
            self.start()
            
        def __enter__(self):
            pass
            
        def __exit__(self, *args):            
            self.shutdown() 
            
        def __repr__(self):
            return '<vipy.globals.parallel: workers=%d, cf=%s>' % (self.num_workers(), GLOBAL['CONCURRENT_FUTURES'] if GLOBAL['CONCURRENT_FUTURES'] else 'stopped')

        def start(self):
            if not GLOBAL['CONCURRENT_FUTURES'] and self._workers>0:                
                cf(num_workers=self._workers, threaded=self._threaded)
            GLOBAL['LOGGER'].info('Parallel executor initialized %s' % self)
            return self
        
        def shutdown(self):
            if GLOBAL['CONCURRENT_FUTURES']:
                GLOBAL['LOGGER'].info('Parallel executor shutdown %s' % self)                            
                GLOBAL['CONCURRENT_FUTURES'].shutdown(wait=True)
            GLOBAL['CONCURRENT_FUTURES'] = None
        
        def num_workers(self):
            return self._workers

    return Parallel(workers if not pct else int(pct*os.cpu_count()))

def multithreading(n=None, pct=None):
    """Context manager for concurrent futures multithreaded executor, use with `vipy.dataset.Dataset`"""
    return parallel(workers=n, pct=pct, threaded=True)

def multiprocessing(n=None, pct=None):
    """Context manager for concurrent futures multiprocessing executor, use with `vipy.dataset.Dataset`"""    
    return parallel(workers=n, pct=pct, threaded=False)

def noparallel():
    """Disable all parallel processing"""
    if GLOBAL['DASK_CLIENT'] is not None:
        GLOBAL['DASK_CLIENT'].shutdown()
        del GLOBAL['DASK_CLIENT']
        GLOBAL['LOGGER'].info('Parallel executor shutdown')
    if GLOBAL['CONCURRENT_FUTURES']:
        GLOBAL['CONCURRENT_FUTURES'].shutdown()
        
    GLOBAL['CONCURRENT_FUTURES'] = None
    GLOBAL['DASK_CLIENT'] = None 
    
    
def shutdown():
    """Alias for `vipy.globals.noparallel`"""    
    return noparallel()

