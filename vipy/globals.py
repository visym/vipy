import os
import webbrowser
import tempfile

from vipy.util import try_import
try_import('dask', 'dask distributed')
from dask.distributed import Client
from dask.distributed import as_completed, wait


GLOBAL = {'VERBOSE': True, 
          'DASK_CLIENT': None}


def cache(cachedir=None):
    if cachedir is not None:
        os.environ['VIPY_CACHE'] = cachedir
    return os.environ['VIPY_CACHE'] if 'VIPY_CACHE' in os.environ else None
    

class Dask(object):
    def __init__(self, num_processes, dashboard=False):
        assert isinstance(num_processes, int) and num_processes >=2, "num_processes must be >= 2"

        self._num_processes = num_processes
        self._client = Client(name='vipy', 
                             scheduler_port=0, 
                             dashboard_address=None if not dashboard else ':0', 
                             processes=True, 
                             threads_per_worker=1, 
                             n_workers=num_processes, 
                             env={'VIPY_BACKEND':'Agg'},
                             direct_to_workers=True,
                             local_directory=tempfile.mkdtemp())

        self._dashboard = 'http://localhost:8787/status' if dashboard else None 

    def __repr__(self):
        return str('<vipy.globals.dask: num_processes=%d%s>' % (self._num_processes, ', dashboard="%s"' % str(self._dashboard) if self._dashboard is not None else ''))

    def dashboard(self):
        return webbrowser.open(self._dashboard) if self._dashboard is not None else None
    
    def num_processes(self):
        return self._num_processes

    def shutdown(self):
        self._client.shutdown()
        GLOBAL['DASK_CLIENT'] = None

    def client(self):
        return self._client


def dask(num_processes=None, dashboard=False):
    if GLOBAL['DASK_CLIENT'] is None and num_processes is not None:
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard)        
    elif GLOBAL['DASK_CLIENT'] is not None and num_processes is not None and GLOBAL['DASK_CLIENT'].num_processes() != num_processes:
        GLOBAL['DASK_CLIENT'].shutdown()
        GLOBAL['DASK_CLIENT'] = Dask(num_processes, dashboard=dashboard)        
    return GLOBAL['DASK_CLIENT']
