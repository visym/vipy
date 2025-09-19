import vipy
import concurrent.futures as cf
import collections
import itertools
import threading
from queue import Queue
import functools


def identity(x):
    """identity function to avoid lambda x: x callables in parallel processing"""
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
        bufsize [int]: The maximum size of the parallel queue used by producers.  This defines the maximum number of tasks in-flight, avoids submitting every element in a long iterator
        accepter [callable]: A function which returns true or false, such that the iterator only yields elements for which the accepter returns true
        progress [bool|int]: If True, show progress with a tqdm style progress bar, if integer, use this number as the progress bar total
    
    Returns:
        An iterator that yields mapped and accepted elements from ingen, whre mapping is performed in parallel by vipy.parallel.cf() executor

    """
    if progress:
        vipy.util.try_import('tqdm','tqdm'); from tqdm import tqdm;
        ingen = tqdm(ingen, total=len(ingen) if hasattr(ingen, '__len__') else (progress if isinstance(progress, int) else None))

    # local iterator fallback
    if vipy.globals.cf() is None:
        for x in ingen:  # iterable access (faster)
            y = mapper(x) if mapper is not None else x
            if accepter is None or accepter(y):
                yield y

    else:
        mapper = identity if mapper is None else mapper
        
        assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): result = [x for x in vipy.parallel.iter(...)]' "
        assert callable(mapper)
        assert accepter is None or callable(accepter)
        assert bufsize>0
        
        e = vipy.globals.cf()        
        q = Queue()
        sem = threading.BoundedSemaphore(bufsize)
    
        # Producer worker: this is useful for filling the pipeline up to a max depth for long iterators while waiting on GPU I/O    
        def _producer():
            for i in ingen:
                sem.acquire()  # block when buffer is full
                f = e.submit(mapper, i)
                def _callback(fut, sem=sem, q=q, i=i):
                    q.put((i,fut.result()))      
                    sem.release()
                f.add_done_callback(_callback)
            for k in range(bufsize):
                sem.acquire()  # wait until all callbacks have fired
            q.put((None,None))
        
        threading.Thread(target=_producer, daemon=True).start()

        # Consumer loop: yield loaded elements from the producers
        while True:        
            (x,y) = q.get()
            if (x,y) == (None,None):
                break
            if accepter is None or accepter(y):
                yield y
    

def map(f, ingen, **kwargs):
    """Apply the function f with the provided kwargs to each element in the iterator, returning the results unordered.  Function cannot be anonymous if cf executor is multiprocess """
    assert vipy.globals.cf() is not None, "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.map(...)' "    
    assert callable(f)    
    return vipy.globals.cf().map(functools.partial(f, **kwargs), ingen)  # order preserving
                

