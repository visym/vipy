from viset.library import caltech101
from bobo.image import ImageCategoryStream

def main():
    for imcategory in ImageCategoryStream(caltech101.export(), cache=caltech101.cache()):
        imcategory.show()

if __name__ == '__main__':
    main()

  
