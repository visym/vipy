import os
import webbrowser
import tempfile
import vipy.math
from vipy.util import remkdir


# Global mutable dictionary
GLOBAL = {'VERBOSE': False, 
          'DASK_CLIENT': None,
          'DASK_MAX_WORKERS':1,
          'CACHE':None,
          'GPU':None}


def cache(cachedir=None):
    """The cache is the location that URLs are downloaded to on your system.  This can be set here, or with the environment variable VIPY_CACHE"""
    if cachedir is not None:
        os.environ['VIPY_CACHE'] = remkdir(cachedir)
    return os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None
    

def verbose(b=None):
    """The global verbosity level, only really used right now for FFMPEG messages"""
    if b is not None:
        GLOBAL['VERBOSE'] = b
    return GLOBAL['VERBOSE']


class Dask(object):
    def __init__(self, num_processes, dashboard=False):
        assert isinstance(num_processes, int) and num_processes >=1, "num_processes must be >= 1"

        from vipy.util import try_import
        try_import('dask', 'dask distributed')
        import dask
        from dask.distributed import Client
        from dask.distributed import as_completed, wait
        from dask.config import set as dask_config_set
        from dask.distributed import get_worker         

        dask_config_set({"distributed.comm.timeouts.tcp": "50s"})
        dask_config_set({"distributed.comm.timeouts.connect": "10s"})        
        self._num_processes = num_processes
        self._client = Client(name='vipy', 
                              scheduler_port=0, 
                              dashboard_address=None if not dashboard else ':0',  # random port
                              processes=True, 
                              threads_per_worker=1, 
                              n_workers=num_processes, 
                              env={'VIPY_BACKEND':'Agg', 'PYTHONOPATH':os.environ['PYTHONPATH'], 'PATH':os.environ['PATH']},
                              direct_to_workers=True,
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
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard)        
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
