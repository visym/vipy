import vipy
import concurrent.futures as cf


def map(f, initer, reducer=None):
    """Apply the function f to each element in the iterator, returning the results unordered.  If strict=True, require executor"""
    assert vipy.globals.cf() is not None, "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.map(...)' "    
    assert callable(f)    
    results = vipy.globals.cf().map(f, initer)  # not order preserving
    return reducer(results) if reducer else results

def identity(x):
    return x

def iter(f, initer, batchsize=1024, progress=False):
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): result = [x for x in vipy.parallel.iter(...)]' "    
    assert f is None or callable(f)    

    batches = vipy.util.chunkgenbysize(initer, batchsize)
    if progress:
        vipy.util.try_import('tqdm', 'tqdm'); from tqdm import tqdm;
        total = (len(initer)//batchsize) if hasattr(initer, '__len__') else None
        batches = tqdm(batches, total=total)
    
    e = vipy.globals.cf()        
    for batch in batches:
        for b in cf.as_completed((e.submit(f if f is not None else identity, b) for b in batch)):
            yield b.result()  # submit batches for very long input iterators
    

def ordered_map(f, initer):
    """Apply the function f to each element in the iterator, returning tuples (order, result) results sorted by order"""
    assert vipy.globals.cf(), "vipy.globals.cf() executor required - Try 'with vipy.globals.parallel(n=4): vipy.parallel.ordered_map(...)' "    
    assert callable(f), "mapper required"    
    return sorted(vipy.globals.cf().map(f, enumerate(initer)), key=lambda x: x[0])  # expensive


