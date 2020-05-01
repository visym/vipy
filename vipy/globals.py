import os
import webbrowser
import tempfile
import vipy.math


GLOBAL = {'VERBOSE': False, 
          'DASK_CLIENT': None,
          'CACHE':None}


def cache(cachedir=None):
    if cachedir is not None:
        os.environ['VIPY_CACHE'] = cachedir
    return os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None
    

def verbose(b=None):
    if b is not None:
        GLOBAL['VERBOSE'] = b
    return GLOBAL['VERBOSE']


class Dask(object):
    def __init__(self, num_processes, dashboard=False):
        assert isinstance(num_processes, int) and num_processes >=2, "num_processes must be >= 2"

        from vipy.util import try_import
        try_import('dask', 'dask distributed')
        import dask
        from dask.distributed import Client
        from dask.distributed import as_completed, wait
        from dask.config import set as dask_config_set
        
        dask_config_set({"distributed.comm.timeouts.tcp": "50s"})
        dask_config_set({"distributed.comm.timeouts.connect": "10s"})        
        self._num_processes = num_processes
        self._client = Client(name='vipy', 
                              scheduler_port=0, 
                              dashboard_address=None if not dashboard else ':0',  # random port
                              processes=True, 
                              threads_per_worker=1, 
                              n_workers=num_processes, 
                              env={'VIPY_BACKEND':'Agg'},
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

    
def dask(num_processes=None, dashboard=False):
    if GLOBAL['DASK_CLIENT'] is None and num_processes is not None:
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard)        
    elif GLOBAL['DASK_CLIENT'] is not None and num_processes is not None and GLOBAL['DASK_CLIENT'].num_processes() != num_processes:
        GLOBAL['DASK_CLIENT'].shutdown()
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard)        
    return GLOBAL['DASK_CLIENT']


def num_workers(n=None):
    if n is not None:
        return dask(num_processes=n)
    return 1 if dask() is None else dask().num_processes()

def max_workers(pct=0.5):
    """Set the maximum number of workers as the largest power of two <= 90% of the number of CPUs on the current system"""
    import multiprocessing
    return dask(num_processes=vipy.math.poweroftwo(pct*multiprocessing.cpu_count()))
