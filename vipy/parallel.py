import vipy
import concurrent.futures as cf
import collections
import itertools
import threading
from queue import Queue


def map(f, ingen, reducer=None):
    """Apply the function f to each element in the iterator, returning the results unordered.  Function cannot be anonymous if cf executor is multiprocess """
    assert vipy.globals.cf() is not None, "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.map(...)' "    
    assert callable(f)    
    results = vipy.globals.cf().map(f, ingen)  # not order preserving
    return reducer(results) if reducer else results

def identity(x):
    return x


def iter(ingen, mapper=identity, bufsize=1024, progress=False):
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): result = [x for x in vipy.parallel.iter(...)]' "    
    assert callable(mapper)    

    e = vipy.globals.cf()
    q = Queue()
    sem = threading.BoundedSemaphore(bufsize)
    lock = threading.Lock()
    
    if progress:
        vipy.util.try_import('tqdm','tqdm'); from tqdm import tqdm;
        ingen = tqdm(ingen, total=len(ingen) if hasattr(ingen, '__len__') else None)
    
    # Producer worker: this is useful for filling the pipeline while waiting on GPU I/O    
    def _producer():
        futures = set()  
        for i in ingen:
            sem.acquire()  # block when buffer is full
            f = e.submit(mapper, i)
            def _callback(fut, sem=sem, q=q):
                q.put(fut.result())      
                sem.release()
                with lock:
                    futures.discard(fut)                
            f.add_done_callback(_callback)
            with lock:
                futures.add(f)
        with lock:
            pending = list(futures)
        cf.wait(pending, return_when=cf.ALL_COMPLETED)
        q.put(None)
        
    threading.Thread(target=_producer, daemon=True).start()

    # Consumer loop: yield loaded elements from the producers
    while True:        
        res = q.get()
        if res is None:
            break
        yield res
    

def ordered_map(f, ingen):
    """Apply the function f to each element in the iterator, returning tuples (order, result) results sorted by order"""
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.ordered_map(...)' "    
    assert callable(f), "mapper required"    
    return sorted(vipy.globals.cf().map(f, enumerate(ingen)), key=lambda x: x[0])  # expensive



    
    
