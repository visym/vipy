import vipy
import vipy.data
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--outdir", help="Output directory to contain downloaded datasets (default=os.environ['VIPY_CACHE'])")
    parser.add_argument("--parallel", help="Number of parallel downloads (default=4)", default=4, type=int)
    parser.add_argument("--redownload", help="Force redownload datasets", default=False, action='store_true')    
    args = parser.parse_args()
        
    classic = [vipy.data.caltech101.Caltech101,
               vipy.data.caltech256.Caltech256,
               vipy.data.cifar.CIFAR10,
               vipy.data.cifar.CIFAR100,
               vipy.data.coco.Detection_TrainVal_2014,
               vipy.data.coil100.COIL100,
               vipy.data.ethzshapes.ETHZShapes,
               vipy.data.fddb.FDDB,
               vipy.data.food101.Food101,
               vipy.data.imagenet.Imagenet2012,
               vipy.data.inaturalist.iNaturalist2021,
               vipy.data.lfw.LFW,
               vipy.data.mit67.MIT67,
               vipy.data.mnist.MNIST,
               vipy.data.mnist.EMNIST,
               vipy.data.objectnet.Objectnet,
               vipy.data.oxford_flowers_102.Flowers102,
               vipy.data.places.Places356,
               vipy.data.visualgenome.VisualGenome]
    
    if args.outdir:
        os.environ['VIPY_CACHE'] = vipy.util.remkdir(args.outdir)
        
    with vipy.globals.parallel(args.parallel):
        vipy.dataset.Dataset(classic).map(lambda d: d(redownload=args.redownload), oneway=True)

    
    

                
