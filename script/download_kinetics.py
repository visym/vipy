import sys
from vipy.dataset.kinetics import Kinetics700
from vipy.batch import Batch
from vipy.util import print_and_return


def run(outdir, n_processes):
    d = Kinetics700(outdir)
    if not d.isdownloaded():
        d.download()
    Batch(d.valset() + d.testset() + d.trainset(), n_processes=n_processes).map(lambda v: v.download(ignoreErrors=True).save(ignoreErrors=True) if not v.hasfilename() else print_and_return(v)).result()

    
if __name__ == '__main__':
    """Please review the youtube terms of service to confirm your compliance before using: https://www.youtube.com/static?template=terms    
       As of Fall 2019, Youtube has instituted rate limiting, which greatly restricts the ability to use the Kinetics dataset for research.
    """
    assert len(sys.argv) >= 2 and len(sys.argv) <= 3, "python download_kinetics.py $/path/to/kinetics $num_processes (e.g. 'python download_kinetics.py /my/kinetics 32', or 'python download_kinetics.py /my/kinetics')"
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv)==3 else 2)

    
