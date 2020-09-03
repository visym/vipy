import os
import webbrowser
import dill
import tempfile
import vipy.math
from vipy.util import remkdir
import builtins
import logging as python_logging
import warnings


# Global mutable dictionary
GLOBAL = {'VERBOSE': True,       # If False, will silence everything, equivalent to calling vipy.globals.silent()
          'VERBOSITY': 2,        # 0=debug, 1=warn, 2=info, only if VERBOSE=True
          'DASK_CLIENT': None,   # Global Dask() client for distributed processing
          'DASK_MAX_WORKERS':1,  # Maximum number of processes when creating Dask() client
          'CACHE':None,          # Cache directory for vipy.video and vipy.image donwloads
          'GPU':None,            # GPU index assigned to this process
          'LOGGING':False,       # If True, use python logging (handler provided by end-user) intead of print 
          'LOGGER':None}         # The global logger used by vipy.globals.print() and vipy.globals.warn() if LOGGING=True


def logging(enable=None, format=None):
    """Single entry point for enabling/disabling logging vs. printing
       
       All vipy functions overload "from vipy.globals import print" for simplified readability of code.
       This global function redirects print or warn to using the standard logging module.
       If format is provided, this will create a basicConfig handler, but this should be configured by the end-user.    
    """
    if enable is not None:
        assert isinstance(enable, bool)
        GLOBAL['LOGGING'] = enable
        if format is not None:
            python_logging.basicConfig(level=python_logging.INFO, format=format)
        GLOBAL['LOGGER'] = python_logging.getLogger('vipy')
        GLOBAL['LOGGER'].propagate = True if enable else False
        
    return GLOBAL['LOGGING']


def warn(s):
    if GLOBAL['VERBOSE']:
        warnings.warn(s) if (not GLOBAL['LOGGING'] or GLOBAL['LOGGER'] is None) else GLOBAL['LOGGER'].warn(s)

        
def print(s, end='\n'):
    """Main entry point for all print statements in the vipy package. All vipy code calls this to print helpful messages.
      
       -Printing can be disabled by calling vipy.globals.silent()
       -Printing can be redirected to logging by calling vipy.globals.logging(True)
       -All print() statements in vipy.* are overloaded to call vipy.globals.print() so that it can be redirected to logging

    """
    if GLOBAL['VERBOSE']:
        builtins.print(s, end=end) if (not GLOBAL['LOGGING'] or GLOBAL['LOGGER'] is None) else GLOBAL['LOGGER'].info(s)


def verbose():
    """The global verbosity level, only really used right now for FFMPEG messages"""
    GLOBAL['VERBOSE'] = True

def isverbose():
    return GLOBAL['VERBOSE']

def silent():
    GLOBAL['VERBOSE'] = False    

def issilent():
    return GLOBAL['VERBOSE'] == False 

def verbosity(v):
    assert v in [0,1,2]    # debug, warn, info
    GLOBAL['VERBOSITY'] = v

def debug():
    verbose()
    verbosity(0)

def isdebug():
    return GLOBAL['VERBOSE'] and GLOBAL['VERBOSITY'] == 0


def cache(cachedir=None):
    """The cache is the location that URLs are downloaded to on your system.  This can be set here, or with the environment variable VIPY_CACHE"""
    if cachedir is not None:
        os.environ['VIPY_CACHE'] = remkdir(cachedir)
    return os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None
    

class Dask(object):
    def __init__(self, num_processes, dashboard=False, verbose=False):
        assert isinstance(num_processes, int) and num_processes >=1, "num_processes must be >= 1"

        from vipy.util import try_import
        try_import('dask', 'dask distributed')
        import dask
        import dask.config
        dask.config.set(distributed__comm__timeouts__tcp="60s")
        dask.config.set(distributed__comm__timeouts__connect="60s")        
        from dask.distributed import Client
        from dask.distributed import as_completed, wait
        from dask.distributed import get_worker         

        self._num_processes = num_processes
        self._client = Client(name='vipy', 
                              scheduler_port=0,   # random
                              dashboard_address=None if not dashboard else ':0',  # random port
                              processes=True, 
                              threads_per_worker=1, 
                              n_workers=num_processes, 
                              env={'VIPY_BACKEND':'Agg',
                                   'PYTHONOPATH':os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '',
                                   'PATH':os.environ['PATH'] if 'PATH' in os.environ else ''},
                              direct_to_workers=True,
                              silence_logs=(False if isdebug() else 30) if not verbose else 40,  # logging.WARN or logging.ERROR or logging.INFO
                              local_directory=tempfile.mkdtemp())

    def __repr__(self):
        return str('<vipy.globals.dask: num_processes=%d%s>' % (self._num_processes, '' if self._num_processes==0 or len(self._client.dashboard_link)==0 else ', dashboard="%s"' % str(self._client.dashboard_link)))

    def dashboard(self):        
        webbrowser.open(self._client.dashboard_link) if len(self._client.dashboard_link)>0 else None
    
    def num_processes(self):
        return self._num_processes

    def shutdown(self):
        self._client.close()
        self._num_processes = 0
        GLOBAL['DASK_CLIENT'] = None
        return self

    def client(self):
        return self._client


def cpuonly():
    GLOBAL['GPU'] = None


def gpuindex(gpu=None):
    if gpu == 'cpu':
        cpuonly()
    elif gpu is not None:
        GLOBAL['GPU'] = gpu
    return GLOBAL['GPU']


def dask(num_processes=None, dashboard=False):
    """Return the local Dask client, can be accessed globally for parallel processing"""
    if (num_processes is not None and (GLOBAL['DASK_CLIENT'] is None or GLOBAL['DASK_CLIENT'].num_processes() != num_processes)):
        if GLOBAL['DASK_CLIENT'] is not None:
            GLOBAL['DASK_CLIENT'].shutdown()
        assert num_processes >= 1, "num_processes>=1"
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard, verbose=isverbose())        
    return GLOBAL['DASK_CLIENT']


def max_workers(n=None, pct=None):
    """Set the maximum number of workers as the largest power of two <= pct% of the number of CPUs on the current system, or the provided number.  This will be used as the default when creating a dask client."""
    if n is not None:
        assert isinstance(n, int)
        GLOBAL['DASK_MAX_WORKERS'] = n
    elif pct is not None:
        import multiprocessing
        GLOBAL['DASK_MAX_WORKERS'] = vipy.math.poweroftwo(pct*multiprocessing.cpu_count())
    return GLOBAL['DASK_MAX_WORKERS'] 
