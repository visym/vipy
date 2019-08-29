import os
import platform
import argparse
import ctypes
import tempfile
from strpy.bobo.util import quietprint, remkdir, timestamp, setverbosity, islinux, ismacosx, linuxversion

SPARK_CONTEXT = None  # Module level variable for current spark context

TIMESTAMP = None # Module level timestamp for unique result directories

def binarypath():
    return os.path.join(release(), 'bin', janusPlatform())

def libfile(libname):
    if janusPlatform() == 'macos10.9':
        filename = os.path.join(release(), 'lib', 'macos10.9', 'lib%s.dylib' % libname)
    elif janusPlatform() in [ 'centos6', 'centos7' ]:
        filename = os.path.join(release(), 'lib', janusPlatform(), 'lib%s.so' % libname)
    else:
        raise ValueError('unsupported os')
    return filename


def num_cores():
    from pyspark import SparkConf            
    sparkConf = SparkConf()
    p = sparkConf.get('spark.default.parallelism')  # defaults to number of cores on platform
    return int(p)  

def name():
    return sc.appName  # FIXME: sc should be a package variable, anyone who needs uses bobo.app.sparkContext
    
def root():
    rootdir = os.environ.get('JANUS_ROOT')
    if rootdir is None:
        raise ValueError('JANUS_ROOT is not set!')
    else:
        return rootdir

def vendor():
    return os.path.join(root(), 'vendor')

def models():
    modeldir = os.environ.get('JANUS_MODELS')
    return modeldir if modeldir is not None else os.path.join(root(), 'models')

def path():
    return os.environ.get('PATH')

def janusCache():
    """WARNING: this is deprecated"""
    #return os.environ.get('JANUS_CACHE')
    return tempfile.gettempdir()


def results():
    return os.environ.get('JANUS_RESULTS')

def resultsUnique(prefix=None):
    results_dir = os.environ.get('JANUS_RESULTS')
    prefix = '' if prefix is None else prefix
    global TIMESTAMP
    if TIMESTAMP is None:
        TIMESTAMP = timestamp()
    results_dir = os.path.join(results_dir, prefix + TIMESTAMP)
    remkdir(results_dir)
    return results_dir

def logs():
    return os.environ.get('JANUS_LOGS')

def release():
    return os.environ.get('JANUS_RELEASE')

def bobo():
    return os.path.join(release(), 'bobo')

def pythonPath():
    return os.environ.get('PYTHONPATH')

def janusPlatform():
    remote = os.environ.get('JANUS_PLATFORM')
    if remote is None:
        raise ValueError('JANUS_PLATFORM is not set!')
    else:
        return remote

def janusWorker():
    worker = os.environ.get('JANUS_WORKER')
    if worker is None:
        return 'False'
    else:
        return worker

def getCentosVersion():
    ver = platform.linux_distribtution()[1]
    return 'centos' + ver[0]

def isWorker():
    worker = janusWorker()
    return (worker.lower() == 'true')

def datadir(indir=None):
    if indir is None:
        return os.environ.get('JANUS_DATA')
    else:
        setData(indir)
    
def boboCache():
    cachedir = os.environ.get('JANUS_DATA')
    if cachedir is None:
        cachedir = os.environ.get('BOBO_CACHE')
        if cachedir is None:        
            raise ValueError('JANUS_DATA is not set!')
        else:
            return cachedir
    else:
        return cachedir

def setData(indir):
    os.environ['JANUS_DATA'] = indir
    
def componentpath():
    return os.path.join(release(),'bin',janusPlatform())

def frameworkpath():
    return os.environ.get('DYLD_FALLBACK_FRAMEWORK_PATH')

def librarypath():
    if janusPlatform() == 'macos10.9':
        return 'DYLD_FALLBACK_LIBRARY_PATH',os.environ.get('JANUS_LD_LIBRARY_PATH')
    elif janusPlatform() in [ 'centos6', 'centos7' ]:
        return 'LD_LIBRARY_PATH',os.environ.get('JANUS_LD_LIBRARY_PATH')
    else:
        raise ValueError('unsupported os')

def loadlibrary(libname):
    class BoboLib():
        def __init__(self, libname):
            if isWorker():
                if janusPlatform() == 'macos10.9':
                    self.libfile = os.path.join(release(), 'lib', 'macos10.9', 'lib%s.dylib' % libname)
                    self.lib = ctypes.cdll.LoadLibrary(self.libfile)
                elif janusPlatform() in [ 'centos6', 'centos7' ]:
                    self.libfile = os.path.join(release(), 'lib', janusPlatform(), 'lib%s.so' % libname)
                    self.lib = ctypes.cdll.LoadLibrary(self.libfile)
                else:
                    raise ValueError('unsupported os')
            else:
                if ismacosx():
                    self.libfile = os.path.join(release(), 'lib', 'macos10.9', 'lib%s.dylib' % libname)
                    self.lib = ctypes.cdll.LoadLibrary(self.libfile)
                elif islinux():
                    self.libfile = os.path.join(release(), 'lib', 'centos%s'%linuxversion()[0], 'lib%s.so' % libname)
                    self.lib = ctypes.cdll.LoadLibrary(self.libfile)
                else:
                    raise ValueError('unsupported os')
            self.lib.initialize()
            pass
            
        def __repr__(self):
            return str('<bobo.app.library: "%s">' % (self.libfile))
            
        def __del__(self):
            #self.lib.release()  # free all allocated resources when python garbage collects this library
            pass
            
    return BoboLib(libname)

    
def init(appName='Bobo Application', verbosity=1, sparkProperties=None):
    quietprint("[bobo.app.init]: Initializing spark platform for application '%s' " % appName, 1)

    # Platform Verbosity
    setverbosity(verbosity)
    quietprint("[bobo.app.init]: Setting verbosity level = %d " % verbosity, 2);
    
    # Spark context already created?
    global SPARK_CONTEXT  # module level variable
    if SPARK_CONTEXT is not None:
        quietprint("[bobo.app.init]: Shutting down spark platform for application '%s' " % SPARK_CONTEXT.appName, 2)        
        SPARK_CONTEXT.stop()
            
    # Results directory
    quietprint("[bobo.app.init]: Janus root = '%s'" % root(), 2)        
    quietprint("[bobo.app.init]: Creating results directory '%s'" % results(), 2)    
    remkdir(results())
    
    # Spark Context!
    from pyspark import SparkContext, SparkConf        

    # Make sure wokrers can find platform 
    (ldvar, ldpath) = librarypath()
        
    python_path = pythonPath()
    environ = {
        'JANUS_ROOT' : root(),
        'JANUS_LOGS' : logs(),
        'JANUS_RESULTS' : results(),
#        'JANUS_CACHE' : janusCache(),
        'JANUS_RELEASE' : release(),
        'PATH' : path(),
        'PYTHONPATH' : python_path,
        ldvar : ldpath,
        'JANUS_LD_LIBRARY_PATH':ldpath,
        'JANUS_DATA' : boboCache(),
        'BOBO_CACHE' : boboCache(),            
        'JANUS_PLATFORM' : janusPlatform(),
        'JANUS_WORKER' : 'True',
        'JANUS_VERBOSITY' : str(verbosity)
    }
    if janusPlatform() == 'macos10.9' and frameworkpath() is not None:
        environ['DYLD_FALLBACK_FRAMEWORK_PATH'] = frameworkpath()

    sparkConf = SparkConf()
    for k,v in environ.iteritems():
        quietprint("[bobo.app.init]: Adding environment %s=%s"%(k,v), 2)
        sparkConf.setExecutorEnv(k,v)

    if sparkProperties is not None:
        [ sparkConf.set(k,v) for (k,v) in sparkProperties.iteritems() ]

    SPARK_CONTEXT = SparkContext(conf=sparkConf, appName=appName)

    # Return sparkContext object for creating RDDs
    quietprint("[bobo.app.init]: Executing application", 2)
    return SPARK_CONTEXT

