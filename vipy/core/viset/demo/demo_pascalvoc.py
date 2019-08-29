from viset.library.pascalvoc import PascalVOC2012
from viset.dataset import Viset
from viset.show import imshow

def main():
    dbfile = PascalVOC2012().export(verbose=True)
    db = Viset(dbfile, verbose=True)
    for (im,annotation) in db.annotation.categorization(step=10):
        print 'Image=' + str(annotation['idx_image']) + ' Category=' + annotation['category']
        imshow(im, title=annotation['category'])
        
if __name__ == '__main__':
    main()

