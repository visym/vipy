from viset.library import imagenet

def main():
    csvfile = '/Volumes/JEBYRNE/visym/cache/imagenet_fall2011/imagenet_fall2011.csv'
    #imagenet.download(imagenet.export())
    imagenet.download(csvfile)

if __name__ == '__main__':
    main()

  
