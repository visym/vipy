from viset.show import imshow
from viset.library import caltech256
from viset.stream import Recognition

def main():
    for (im,annotation) in Recognition(caltech256.export()):
        imshow(im.get(), title=annotation)

if __name__ == '__main__':
    main()

