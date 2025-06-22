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


def iter(ingen, mapper=identity, bufsize=1024, progress=False, accepter=None):
    """Return an iterator on the input generator that will apply the mapper to each object and yield only those elements where the accepter is true

    ```python
    with vipy.globals.parallel(8):
        for im in vipy.parallel.iter(vipy.dataset.registry('yfcc100m_url:train'), mapper=lambda im: im.try_download(), accepter=lambda im: im.is_downloaded()):
            print(im)
    ```

    Most common use cases will use `vipy.dataset.Dataset.minibatch` or `vipy.dataset.Dataset.__parallel_iter__` instead of using this iterator directly

    Args:
        ingen [generator]:
        mapper [callable]: A function applied to each element before yielding
        bufsize [int]: The maximum size of the parallel queue used by producers.  
        accepter [callable]: A function which returns true or false, such that the iterator only yields elements for which the accepter returns true

    Returns:
        An iterator that yields mapped and accepted elements from ingen, whre mapping is performed in parallel by vipy.parallel.cf() executor

    """

    # local iterator fallback
    if vipy.globals.cf() is None:
        for x in ingen:  # iterable access (faster)
            x = mapper(x) if mapper is not None else x
            if accepter is None or accepter(x):
                yield x                

    else:    
        assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): result = [x for x in vipy.parallel.iter(...)]' "    
        assert callable(mapper)
        assert accepter is None or callable(accepter)
        assert bufsize>0
        
        e = vipy.globals.cf()
        q = Queue()
        sem = threading.BoundedSemaphore(bufsize)
    
        if progress:
            vipy.util.try_import('tqdm','tqdm'); from tqdm import tqdm;
            ingen = tqdm(ingen, total=len(ingen) if hasattr(ingen, '__len__') else None)
    
        # Producer worker: this is useful for filling the pipeline while waiting on GPU I/O    
        def _producer():
            for i in ingen:
                sem.acquire()  # block when buffer is full
                f = e.submit(mapper, i)
                def _callback(fut, sem=sem, q=q):
                    q.put(fut.result())      
                    sem.release()
                f.add_done_callback(_callback)
            for k in range(bufsize):
                sem.acquire()  # wait until all callbacks have fired
            q.put(None)
        
        threading.Thread(target=_producer, daemon=True).start()

        # Consumer loop: yield loaded elements from the producers
        while True:        
            res = q.get()
            if res is None:
                break
            if accepter is None or accepter(res):
                yield res
    

def ordered_map(f, ingen):
    """Apply the function f to each element in the iterator, returning tuples (order, result) results sorted by order"""
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.ordered_map(...)' "    
    assert callable(f), "mapper required"    
    return sorted(vipy.globals.cf().map(f, enumerate(ingen)), key=lambda x: x[0])  # expensive



    
    
