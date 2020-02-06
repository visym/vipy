import numpy as np
from vipy.geometry import BoundingBox
from vipy.image import ImageDetection

def run():
    try:
        bb = BoundingBox()
        raise
    except:
        print('Empty constructor: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=10, height=10.2)
    print('(x,y,w,h) constructor: PASSED')        

    bb = BoundingBox(xmin='0', ymin='0.5', width='1E2', height='10.2')
    print('(x,y,w,h) string constructor: PASSED')        

    bb = BoundingBox(xmin='0', ymin='0.5', xmax='1E2', ymax='1000.2')
    print('(xmin,ymin,xmax,ymax) constructor: PASSED')        

    bb = BoundingBox(centroid=('0',0), width=10, height=10)
    print('(centroid, width, height) constructor: PASSED')        

    try:
        bb = BoundingBox(mask=np.zeros( (10,10) ))        
        raise
    except:
        print('Degenerate mask constructor: PASSED')        
        
    bb = BoundingBox(xmin=0, ymin=0, width=-100, height=0)
    if not bb.isdegenerate():
        raise
    print('Invalid box: PASSED')


    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=50, width=100, height=100)
    bb1.intersection(bb2)
    assert bb1.to_xywh() == [50.0, 50.0, 50.0, 50.0]
    print('Box.intersection: PASSED')

    bb2 = BoundingBox(xmin=200, ymin=200, width=100, height=100)    
    try:
        bb1.intersection(bb2)
        raise
    except:
        bb1.intersection(bb2, strict=False)
        print('Box.intersection degeneracy: PASSED')        
        
    
    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb.translate(10,20)
    assert bb.to_xywh() == [10.0, 20.0, 100.0, 100.0]
    print('Box.translate: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb.setheight(20)
    bb.setwidth(40)
    assert bb.to_xywh() == [30.0, 40.0, 40.0, 20]
    print('Box.setheight: PASSED')
    print('Box.setwidth: PASSED')    

    bb = BoundingBox(centroid=(10,10), width=1, height=1)
    assert bb.centroid() == [10,10]
    print('Box.centroid: PASSED')
    
    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=60, width=100, height=100)
    bb1.dx(bb2)
    bb1.dy(bb2)
    assert bb1.dx(bb2) == 50 and bb1.dy(bb2) == 60
    print('Box.dx: PASSED')
    print('Box.dy: PASSED')    

    assert(bb1.dist(bb2) == np.sqrt(50*50 + 60*60))
    print('Box.dist: PASSED')        

    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=50, width=50, height=50)
    assert bb1.iou(bb2) == ((50.0*50.0)/(100.0*100.0))
    print('Box.iou: PASSED')

    assert bb1.inside((0,25)) and not bb1.inside( (-10,0) ) and bb1.inside( (99,99) )
    print('Box.inside: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate(2.0)
    assert bb.width() == 200 and bb.height() == 200 and bb.centroid() == [50,50]
    print('Box.dilate: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate_height(3.0)
    assert bb.width() == 100 and bb.height() == 300 and bb.centroid() == [50,50]
    print('Box.dilate_height: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate_width(1.1)
    assert bb.width() == 110 and bb.height() == 100 and list(np.round(bb.centroid())) == [50,50]
    print('Box.dilate_width: PASSED')

    bb = BoundingBox(xmin=1, ymin=1, width=1, height=1).top(1.0).bottom(1.0).left(1.0).right(1.0)
    assert bb.to_xywh() == [0,0,3,3]
    print('Box.top: PASSED')
    print('Box.bottom: PASSED')
    print('Box.left: PASSED')
    print('Box.right: PASSED')    
    
    bb = BoundingBox(xmin=1, ymin=1, width=1, height=1).rescale(2.5)
    assert bb.to_xywh() == [2.5,2.5,2.5,2.5]
    print('Box.rescale: PASSED')    
 
    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).maxsquare()
    assert bb.to_xywh() == [5,20,40,40]
    print('Box.maxsquare: PASSED')    

    bb = BoundingBox(xmin=0, ymin=0, width=0, height=0).convexhull(np.array([ [0,0],[100,100],[50,50] ]))
    assert bb.to_xywh() == [0,0,100,100]
    print('Box.convexhull: PASSED')    

    img = ImageDetection(array=np.zeros( (128,256) )).boundingbox(xmin=10,ymin=20,width=30,height=40).mask()
    im = ImageDetection(array=np.float32(img), colorspace='float').boundingbox(xmin=10,ymin=20,width=30,height=40)
    assert np.sum(img) == np.sum(im.crop().array())
    
    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).rot90cw(128, 256)
    im = ImageDetection(array=np.float32(np.rot90(img,3)), bbox=bb, colorspace='float')
    assert np.sum(img) == np.sum(im.crop().array())

    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).rot90ccw(128, 256)
    im = ImageDetection(array=np.rot90(img,1), bbox=bb)
    assert np.sum(img) == np.sum(im.crop().array())
    print('Box.rot90: PASSED')
    
    
if __name__ == "__main__":
    run()



