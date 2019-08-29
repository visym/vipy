from viset.library.labelme import LabelMe3
from viset.dataset import Viset
from viset.show import imshow

def main():
    dbfile = LabelMe3().export(verbose=True)
    db = Viset(dbfile, verbose=True)
    for (im) in db.image:
        #print 'Image=' + str(annotation['id_img']) + ' Category=' + annotation['category']
        if im is not None:
            imshow(im)

if __name__ == '__main__':
    main()

