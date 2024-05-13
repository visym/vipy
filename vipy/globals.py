import os
import webbrowser
import tempfile
import vipy.math
import builtins
import logging 


# Global mutable dictionary
GLOBAL = {'DASK_CLIENT': None,   # Global Dask() client for distributed processing
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
    
    def __init__(self, num_processes=None, dashboard=False, verbose=False, address=None):
        from vipy.util import try_import
        try_import('dask', 'dask distributed'); import dask, dask.distributed;
    
        assert address is not None or num_processes is not None, "Invalid input"

        self._num_processes = num_processes

        # Dask configuration: https://docs.dask.org/en/latest/configuration.html
        os.environ['DASK_LOGGING__DISTRIBUTED'] = 'warning' if not verbose else 'info'
        os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT'] = "30s"
        os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP'] = "30s"
        os.environ['DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT'] = "30s"
        os.environ['DASK_DISTRIBUTED__COMM__RETRY__COUNT'] = "10"        

        dask.config.set({'DISTRIBUTED.COMM.RETRY.COUNT'.lower():os.environ['DASK_DISTRIBUTED__COMM__RETRY__COUNT']})
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.CONNECT'.lower():os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT']})
        dask.config.set({'DISTRIBUTED.COMM.TIMEOUTS.TCP'.lower():os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP']})
        dask.config.set({'DISTRIBUTED.DEPLOY.LOST_WORKER_TIMEOUT'.lower():os.environ['DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT']})
        dask.config.refresh()

        # Worker env
        env = {'VIPY_BACKEND':'Agg',  # headless in workers
               'PYTHONOPATH':os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '',
               'PATH':os.environ['PATH'] if 'PATH' in os.environ else ''}

        if 'VIPY_CACHE' in os.environ:
            env.update({'VIPY_CACHE':os.environ['VIPY_CACHE']})
        if 'VIPY_AWS_ACCESS_KEY_ID' in os.environ:
            env.update({'VIPY_AWS_ACCESS_KEY_ID':os.environ['VIPY_AWS_ACCESS_KEY_ID']})            
        if  'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ:
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
            # Local scheduler
            self._client = dask.distributed.Client(name='vipy',
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


    def __repr__(self):
        if self._num_processes is not None:
            # Local 
            return str('<vipy.globals.dask: %s%s>' % ('processes=%d' % self.num_processes(), ', dashboard="%s"' % str(self._client.dashboard_link) if self.has_dashboard() else ''))
        elif self._client is not None:
            # Distributed
            return str('<vipy.globals.dask: %s>' % (str(self._client)))
        else:
            return str('<vipy.globals.dask: shutdown')

    def has_dashboard(self):
        return len(self._client.dashboard_link) > 0 if self._client is not None else False

    def dashboard(self):
        """Open a web dashboard for dask client.  As of 2024, this appears to be broken returning 404"""
        webbrowser.open(self._client.dashboard_link) if len(self._client.dashboard_link)>0 else None
    
    def num_processes(self):
        return len(self._client.nthreads()) if self._client is not None else 0

    def shutdown(self):
        self._client.close()
        self._num_processes = None
        GLOBAL['DASK_CLIENT'] = None        
        return self

    def client(self):
        return self._client



#def logging(enable=None, format=None):
#    """Single entry point for enabling/disabling logging vs. printing
#       
#       All vipy functions overload "from vipy.globals import print" for simplified readability of code.
#       This global function redirects print or warn to using the standard logging module.
#       If format is provided, this will create a basicConfig handler, but this should be configured by the end-user.    
#    """
#    if enable is not None:
#        assert isinstance(enable, bool)
#        GLOBAL['LOGGING'] = enable
#        if format is not None:
#            python_logging.basicConfig(level=python_logging.INFO, format=format)
#        GLOBAL['LOGGER'] = python_logging.getLogger('vipy')
#        GLOBAL['LOGGER'].propagate = True if enable else False
#        
#    return GLOBAL['LOGGING']


#def warn(s):
#    if GLOBAL['VERBOSE']:
#        warnings.warn(s) if (not GLOBAL['LOGGING'] or GLOBAL['LOGGER'] is None) else GLOBAL['LOGGER'].warn(s)

        
#def print(s, end='\n'):
#    """Main entry point for all print statements in the vipy package. All vipy code calls this to print helpful messages.
#
#    .. notes::
#        -Printing can be disabled by calling vipy.globals.silent()
#        -Printing can be redirected to logging by calling vipy.globals.logging(True)
#        -All print() statements in vipy.* are overloaded to call vipy.globals.print() so that it can be redirected to logging
#        -System print is flushed for buffered stdout (e.g. tee logging)
#    """
#    if GLOBAL['VERBOSE']:
#        builtins.print(s, end=end, flush=True) if (not GLOBAL['LOGGING'] or GLOBAL['LOGGER'] is None) else GLOBAL['LOGGER'].info(s)


#def verbose():
#    """The global verbosity level, only really used right now for FFMPEG messages"""
#    GLOBAL['VERBOSE'] = True

#def isverbose():
#    return GLOBAL['VERBOSE']

#def silent():
#    """Silence the global verbosity level, only really used right now for FFMPEG messages"""
#    GLOBAL['VERBOSE'] = False    

#def issilent():
#    """Is the global verbosity silent?"""
#    return GLOBAL['VERBOSE'] == False 

#def verbosity(v):
#    """Set the global verbosity level [0,1,2]=debug, warn, info"""
#    assert v in [0,1,2]    # debug, warn, info
#    GLOBAL['VERBOSITY'] = v

#def debug():
#    verbose()
#    verbosity(0)

#def isdebug():
#    return GLOBAL['VERBOSE'] and GLOBAL['VERBOSITY'] == 0


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
            

def dask(num_processes=None, dashboard=False, address=None, pct=None):
    """Return the current Dask client, can be accessed globally for parallel processing.
    
    Args:
        pct: float in [0,1] the percentage of the current machine to use
        address:  the dask scheduler of the form 'HOSTNAME:PORT'
        num_processes:  the number of prpcesses to use on the current machine
        dashboard: [bool] whether to inialize the dask client with a web dashboard

    Returns:
        The `vipy.batch.Dask` object pointing to the Dask Distrbuted object
    """
    if pct is not None:
        assert pct > 0 and pct <= 1
        import multiprocessing
        num_processes = vipy.math.poweroftwo(pct*multiprocessing.cpu_count())        
    if (address is not None or (num_processes is not None and (GLOBAL['DASK_CLIENT'] is None or GLOBAL['DASK_CLIENT'].num_processes() != num_processes))):
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard, verbose=True, address=address)        
    return GLOBAL['DASK_CLIENT']


def parallel(n=None, pct=None, scheduler=None):
    """Enable parallel processing with n>=1 processes or a percentage of system core (pct in [0,1]) or a dask scheduler .

    This can be be used as a context manager
    
    >>> with vipy.globals.parallel(n=4):
    >>>     vipy.batch.Batch(...)

    or using the global variables:

    >>> vipy.globals.parallel(n=4):
    >>> vipy.batch.Batch(...)
    >>> vipy.globals.noparallel()
    
    To check the current parallelism level:
    
    >>> num_processes = vipy.globals.parallel()

    To run with a dask scheduler:
    
    >>> with vipy.globals.parallel(scheduler='10.0.1.1:8585')
    >>>    vipy.batch.Batch(...)

    Args:
        n: [int] number of parallel processes
        pct: [float] the percentage [0,1] of system cores to dedicate to parallel processing
        scheduler: [str]  the dask scheduler of the form 'HOSTNAME:PORT' like '128.0.0.1:8785'.  See <https://docs.dask.org/en/latest/install.html>
    """

    class Parallel():
        def __init__(self, n=None, pct=None, scheduler=None):
            assert n is not None or pct is not None or scheduler is not None
            assert sum([x is not None for x in (n, pct, scheduler)]) == 1, "Exactly one"
            assert n is None or (isinstance(n, int) and n>=1)
            assert pct is None or (pct > 0 and pct <= 1)
            dask(num_processes=n, pct=pct, address=scheduler)
            self._n = n
            self._pct = pct
            self._scheduler = scheduler
            
        def __enter__(self):
            pass

        def __exit__(self, *args):
            if self._scheduler is None:
                noparallel()

    if n is None and pct is None and scheduler is None:
        return GLOBAL['DASK_CLIENT'].num_processes() if  GLOBAL['DASK_CLIENT'] is not None else 0
    else:
        assert n is not None or pct is not None or scheduler is not None
        assert sum([x is not None for x in (n, pct, scheduler)]) == 1, "Exactly one"
        return Parallel(n=n, pct=pct, scheduler=scheduler) 



def noparallel():
    """Disable all parallel processing"""
    if GLOBAL['DASK_CLIENT'] is not None:
        GLOBAL['DASK_CLIENT'].shutdown()
        del GLOBAL['DASK_CLIENT']
    GLOBAL['DASK_CLIENT'] = None 

def nodask():
    """Alias for `vipy.globals.noparallel`"""
    return noparallel()
