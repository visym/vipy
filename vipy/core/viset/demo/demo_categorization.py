import sys
from viset.show import imshow
from viset.library import caltech101
from viset.stream import Recognition

def main():
    filename = sys.argv[1]
    for (im,annotation) in Recognition(filename):
        imshow(im.get(), title=annotation)
        
if __name__ == '__main__':
  main()

