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


def iter(ingen, mapper=identity, bufsize=1024):
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): result = [x for x in vipy.parallel.iter(...)]' "    
    assert callable(mapper)    

    e = vipy.globals.cf()
    q = Queue()

    # Producer thread: this is useful for filling the pipeline while waiting on GPU I/O    
    def _producer():
        futures = set()
        for i in ingen:
            futures.add(e.submit(mapper, i))
            if len(futures) >= bufsize:
                done, futures = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for d in done:
                    q.put(d.result())
        for f in futures:
            q.put(f.result())
        q.put(None)                
    threading.Thread(target=_producer, daemon=True).start()

    # Consumer loop
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



    
    
