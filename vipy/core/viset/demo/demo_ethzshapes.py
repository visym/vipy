from viset.library.ethzshapes import ETHZShapeClasses
from viset.dataset import Viset
from viset.show import imshow, imbbox

def main():
    dbfile = ETHZShapeClasses().export(verbose=True)
    db = Viset(dbfile, verbose=True)
    for (im,annotation) in db.annotation.detection:
        print 'Image=' + str(annotation['idx_image']) + ' Category=' + annotation['category']
        imbbox(im,annotation['bbox_xmin'],annotation['bbox_xmax'],annotation['bbox_ymin'],annotation['bbox_ymax'], bboxcaption=annotation['category'])

if __name__ == '__main__':
    main()
        
