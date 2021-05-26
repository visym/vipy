import os
import webbrowser
import dill
import tempfile
import vipy.math
from vipy.util import remkdir, tempdir
import builtins
import logging as python_logging
import warnings


# Global mutable dictionary
GLOBAL = {'VERBOSE': True,       # If False, will silence everything, equivalent to calling vipy.globals.silent()
          'VERBOSITY': 2,        # 0=debug, 1=warn, 2=info, only if VERBOSE=True
          'DASK_CLIENT': None,   # Global Dask() client for distributed processing
          'CACHE':os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None,   # Cache directory for vipy.video and vipy.image donwloads
          'GPU':None,            # GPU index assigned to this process
          'LOGGING':False,       # If True, use python logging (handler provided by end-user) intead of print 
          'LOGGER':None,         # The global logger used by vipy.globals.print() and vipy.globals.warn() if LOGGING=True
          'GUI':{'escape':False},
          'AWS':{'AWS_ACCESS_KEY_ID':os.environ['VIPY_AWS_SECRET_ACCESS_KEY'] if 'VIPY_AWS_ACCESS_KEY_ID' in os.environ else None,
                 'AWS_SECRET_ACCESS_KEY':os.environ['VIPY_AWS_SECRET_ACCESS_KEY'] if 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ else None,
                 'AWS_SESSION_TOKEN':os.environ['VIPY_AWS_SESSION_TOKEN'] if 'VIPY_AWS_SESSION_TOKEN' in os.environ else None},
          'LATEX':os.environ['VIPY_LATEX'] if 'VIPY_LATEX' in os.environ else None}


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

    .. notes::
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
    """Silence the global verbosity level, only really used right now for FFMPEG messages"""
    GLOBAL['VERBOSE'] = False    

def issilent():
    """Is the global verbosity silent?"""
    return GLOBAL['VERBOSE'] == False 

def verbosity(v):
    """Set the global verbosity level [0,1,2]=debug, warn, info"""
    assert v in [0,1,2]    # debug, warn, info
    GLOBAL['VERBOSITY'] = v

def debug():
    verbose()
    verbosity(0)

def isdebug():
    return GLOBAL['VERBOSE'] and GLOBAL['VERBOSITY'] == 0


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
            
def cpuonly():
    GLOBAL['GPU'] = None


def gpuindex(gpu=None):
    if gpu == 'cpu':
        cpuonly()
    elif gpu is not None:
        GLOBAL['GPU'] = gpu
    return GLOBAL['GPU']


def dask(num_processes=None, num_gpus=None, dashboard=False, address=None, pct=None):
    """Return the current Dask client, can be accessed globally for parallel processing.
    
    Args:
        pct: float in [0,1] the percentage of the current machine to use
        address:  the dask scheduler of the form 'HOSTNAME:PORT'
        num_processes:  the number of prpcesses to use on the current machine
        num_gpus:  the number of GPUs to use on the current machine
        dashboard: [bool] whether to inialize the dask client with a web dashboard

    Returns:
        The `vipy.batch.Dask` object pointing to the Dask Distrbuted object
    """
    from vipy.batch import Dask
    if pct is not None:
        assert pct > 0 and pct <= 1
        import multiprocessing
        num_processes = vipy.math.poweroftwo(pct*multiprocessing.cpu_count())        
    if (address is not None or (num_processes is not None and (GLOBAL['DASK_CLIENT'] is None or GLOBAL['DASK_CLIENT'].num_processes() != num_processes)) or num_gpus is not None):
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard, verbose=isverbose(), address=address, num_gpus=num_gpus)        
    return GLOBAL['DASK_CLIENT']


def parallel(n=None, pct=None, scheduler=None):
    """Enable parallel processing with n>=1 processes or a percentage of system core (pct \in [0,1]) or a dask scheduler .

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
            
        def __enter__(self):
            pass

        def __exit__(self, *args):
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
    GLOBAL['DASK_CLIENT'] = None 

def nodask():
    """Alias for `vipy.globals.noparallel`"""
    return noparallel()
