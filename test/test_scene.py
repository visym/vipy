import sys
sys.path.append('..')
from vipy.scene import Scene
from vipy.object import Detection

def run():
    im = Scene()
    print('Empty Constructor: PASSED')

    im = Scene(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').read()
    print('URL: PASSED')

    im = im.objects([Detection('obj1',50,100,300,300), Detection('obj2',600,600,400,400)])
    im.show(outfile='test_scene.jpg')
    
if __name__ == "__main__":
    run()



