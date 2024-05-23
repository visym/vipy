import vipy
import vipy.batch
import concurrent.futures as cf


def executor(workers=None):
    return vipy.globals.cf(workers, threaded=True)


def localmap(f, initer):
    for x in initer:
        yield f(x)

        
def map(f, initer, strict=False, reducer=None):
    """Apply the function f to each element in the iterator, returning the results unordered.  If strict=True, require executor"""
    assert not strict or executor(), "vipy.parallel.executor() required"    
    assert callable(f)    
    results = executor().map(f, initer) if strict and executor() else localmap(f, initer)   # not order preserving, with fallback
    return reducer(results) if reducer else results


def streaming_map(f, initer, batchsize=1024):
    assert executor(), "vipy.parallel.executor() required"    
    assert callable(f)    

    e = executor()    
    for batch in vipy.util.chunkgenbysize(initer, batchsize):
        for b in cf.as_completed((e.submit(f,b) for b in batch)):
            yield b.result()  # submit batches for ver ong input iterators
    

def ordered_map(f, initer):
    """Apply the function f to each element in the iterator, returning tuples (order, result) results sorted by order"""
    assert executor(), "vipy.parallel.executor() required"    
    assert callable(f), "mapper required"    
    return sorted(executor().map(f, enumerate(initer)), key=lambda x: x[0])  # expensive


