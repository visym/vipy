import sys
from vipy.dataset.kinetics import Kinetics700
from vipy.batch import Batch
from vipy.util import print_and_return


def run(outdir):
    d = Kinetics700(outdir)
    if not d.isdownloaded():
        d.download()
    Batch(d.valset() + d.testset() + d.trainset()).map(lambda v: v.download(ignoreErrors=True).save(ignoreErrors=True) if not v.hasfilename() else print_and_return(v))

    
if __name__ == '__main__':
    assert len(sys.argv) == 2, "python download_kinetics.py /path/to/kinetics"
    run(sys.argv[1])

