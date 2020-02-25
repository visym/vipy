import numpy as np
from vipy.geometry import BoundingBox
from vipy.image import ImageDetection
import vipy.geometry
import vipy.linalg
from vipy.util import Failed


def test_geometry():
    C = vipy.linalg.random_positive_semidefinite_matrix(2)
    assert vipy.geometry.covariance_to_ellipse(C).shape == (3,)
    print('[test_geometry.cov_to_ellipse]: passed')

    p = np.random.rand(2)
    assert np.all(vipy.geometry.dehomogenize(vipy.geometry.homogenize(p)).flatten() == p.flatten())
    p = np.random.rand(2,4)
    assert np.all(vipy.geometry.dehomogenize(vipy.geometry.homogenize(p)).flatten() == p.flatten())
    try:
        p = np.random.rand(4,2)
        vipy.geometry.dehomogenize(p)
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        p = np.random.rand(4,3)
        vipy.geometry.homogenize(p)
        raise Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_geometry.homogeneize]: passed')
    print('[test_geometry.dehomogeneize]: passed')

    vipy.geometry.apply_homography(np.random.rand(3,3), np.random.rand(2,1))
    try:
        vipy.geometry.apply_homography(np.random.rand(3,3), np.random.rand(1,3))
        raise Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_geometry.apply_homography]: passed')

    assert vipy.geometry.random_affine_transform().shape == (3,3)
    print('[test_geometry.random_affine_transform]: passed')
    assert vipy.geometry.similarity_transform_2x3((1,2), 3, 4).shape == (2,3)
    print('[test_geometry.similarity_transform_2x3]: passed')
    assert vipy.geometry.similarity_transform((1,2), 3, 4).shape == (3,3)
    print('[test_geometry.similarity_transform]: passed')
    assert vipy.geometry.sqdist(np.random.rand(10,4), np.random.rand(20,4)).shape == (10,20)
    print('[test_geometry.sqdist]: passed')
    assert np.allclose(np.linalg.norm(vipy.geometry.normalize(np.random.rand(4))), 1.0)
    print('[test_geometry.normalize]: passed')


def test_boundingbox():
    # Constructors
    try:
        bb = BoundingBox()
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        bb = BoundingBox(xmin=0)
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        bb = BoundingBox(xmin=0, ymin=0, xcentroid=0, ycentroid=0)
        raise Failed()
    except Failed:
        raise
    except:
        pass
    try:
        bb = BoundingBox(xmin=0, width=0, xcentroid=0, ycentroid=0)
        raise Failed()
    except Failed:
        raise
    except:
        pass
    print('[test_geometry.boundingbox]: Degenerate constructors: PASSED')

    str(BoundingBox(xmin=0, ymin=0, width=10, height=10.2))
    print('[test_geometry.boundingbox]: __str__ PASSED')

    BoundingBox(xmin=0, ymin=0, width=10, height=10.2).__repr__()
    print('[test_geometry.boundingbox]: __repr__ PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=10, height=10.2)
    print('[test_geometry.boundingbox]: (x,y,w,h) constructor: PASSED')

    bb = BoundingBox(xmin='0', ymin='0.5', width='1E2', height='10.2')
    print('[test_geometry.boundingbox]: (x,y,w,h) string constructor: PASSED')

    bb = BoundingBox(xmin='0', ymin='0.5', xmax='1E2', ymax='1000.2')
    print('[test_geometry.boundingbox]: (xmin,ymin,xmax,ymax) constructor: PASSED')

    bb = BoundingBox(centroid=('0',0), width=10, height=10)
    print('[test_geometry.boundingbox]: (centroid, width, height) constructor: PASSED')

    assert BoundingBox(centroid=(0,0), width=10, height=20) == BoundingBox(xmin=-5, ymin=-10, width=10, height=20)
    assert BoundingBox(xmin=10, ymin=20, xmax=30, ymax=40) != BoundingBox(xmin=-5, ymin=-10, width=10, height=20)
    print('[test_geometry.boundingbox]: equivalence PASSED')

    try:
        bb = BoundingBox(mask=np.zeros((10,10)))
        raise Failed()
    except Failed:
        raise
    except:
        print('[test_geometry.boundingbox]: Degenerate mask constructor: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=-100, height=0)
    assert bb.isdegenerate()
    print('[test_geometry.boundingbox]: Invalid box: PASSED')

    # Corners
    bb = BoundingBox(xmin=10, ymin=20, width=100, height=200)
    assert bb.xmin() == 10
    assert bb.ymin() == 20
    assert bb.width() == 100
    assert bb.height() == 200
    assert bb.xmax() == 10 + 100
    assert bb.ymax() == 20 + 200
    assert bb.upperleft() == (10,20)
    assert bb.upperright() == (10 + 100, 20)
    assert bb.bottomleft() == (10, 20 + 200)
    assert bb.bottomright() == (10 + 100, 20 + 200)
    print('[test_geometry.boundingbox.corners]: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb.setheight(20)
    bb.setwidth(40)
    assert bb.to_xywh() == (30.0, 40.0, 40.0, 20)
    print('[test_geometry.boundingbox.setheight]: PASSED')
    print('[test_geometry.boundingbox.setwidth]: PASSED')

    bb = BoundingBox(centroid=(10,10), width=1, height=1)
    assert bb.centroid() == (10,10)
    print('[test_geometry.boundingbox.centroid]: PASSED')

    # Intersection
    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=50, width=100, height=100)
    bb1.intersection(bb2)
    assert bb1.to_xywh() == (50.0, 50.0, 50.0, 50.0)
    assert bb1.xywh() == (50.0, 50.0, 50.0, 50.0)
    print('[test_geometry.boundingbox.intersection]: PASSED')
    print('[test_geometry.boundingbox.xwyh]: PASSED')

    bb2 = BoundingBox(xmin=200, ymin=200, width=100, height=100)
    try:
        bb1.intersection(bb2)
        raise Failed()
    except Failed:
        raise
    except:
        bb1.intersection(bb2, strict=False)
        print('[test_geometry.boundingbox]: intersection degeneracy: PASSED')

    # Translation
    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb.translate(10,20)
    assert bb.to_xywh() == (10.0, 20.0, 100.0, 100.0)
    print('[test_geometry.boundingbox]: translate: PASSED')
    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=60, width=100, height=100)
    bb1.dx(bb2)
    bb1.dy(bb2)
    assert bb1.dx(bb2) == 50 and bb1.dy(bb2) == 60
    print('[test_geometry.boundingbox.dx]: PASSED')
    print('[test_geometry.boundingbox.dy]: PASSED')

    # Dist
    assert(bb1.dist(bb2) == np.sqrt(50 * 50 + 60 * 60))
    print('[test_geometry.boundingbox.dist]: PASSED')

    # IoU
    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    bb2 = BoundingBox(xmin=50, ymin=50, width=50, height=50)
    assert bb1.iou(bb2) == ((50.0 * 50.0) / (100.0 * 100.0))
    assert bb1.intersection_over_union(bb2) == ((50.0 * 50.0) / (100.0 * 100.0))
    print('[test_geometry.boundingbox.iou]: PASSED')

    # Union
    bb1 = BoundingBox(xmin=1, ymin=2, width=3, height=1)
    bb2 = BoundingBox(xmin=0, ymin=0, width=2, height=5)
    assert bb1.clone().union(bb2).xywh() == (0,0,4,5)    
    bb3 = BoundingBox(xmin=0, ymin=0, width=2, height=6)    
    assert bb1.clone().union([bb2,bb3]).xywh() == (0,0,4,6)
    assert bb1.clone().union([]).xywh() == bb1.xywh()
    print('[test_geometry.boundingbox.union]: PASSED')
    
    # Inside
    bb1 = BoundingBox(xmin=0, ymin=0, width=100, height=100)
    assert bb1.inside((0,25)) and not bb1.inside((-10,0)) and bb1.inside((99,99))
    print('[test_geometry.boundingbox.inside]: PASSED')

    # Typecast
    assert isinstance(bb1.int().xmin(), int)
    print('[test_geometry.boundingbox.int]: PASSED')

    # Dilation
    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate(2.0)
    assert bb.width() == 200 and bb.height() == 200 and bb.centroid() == (50,50)
    print('[test_geometry.boundingbox.dilate]: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate_height(3.0)
    assert bb.width() == 100 and bb.height() == 300 and bb.centroid() == (50,50)
    print('[test_geometry.boundingbox.dilate_height]: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=100, height=100).dilate_width(1.1)
    assert bb.width() == 110 and bb.height() == 100 and list(np.round(bb.centroid())) == [50,50]
    print('[test_geometry.boundingbox.dilate_width]: PASSED')

    bb = BoundingBox(xmin=1, ymin=1, width=1, height=1).top(1.0).bottom(1.0).left(1.0).right(1.0)
    assert bb.to_xywh() == (0,0,3,3)
    print('[test_geometry.boundingbox.top]: PASSED')
    print('[test_geometry.boundingbox.bottom]: PASSED')
    print('[test_geometry.boundingbox.left]: PASSED')
    print('[test_geometry.boundingbox.right]: PASSED')

    # Transformations
    bb = BoundingBox(xmin=1, ymin=1, width=1, height=1).rescale(2.5)
    assert bb.to_xywh() == (2.5,2.5,2.5,2.5)
    print('[test_geometry.boundingbox.rescale]: PASSED')

    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).maxsquare()
    assert bb.to_xywh() == (5,20,40,40)
    print('[test_geometry.boundingbox.maxsquare]: PASSED')

    bb = BoundingBox(xmin=0, ymin=0, width=0, height=0).convexhull(np.array([[0,0],[100,100],[50,50]]))
    assert bb.to_xywh() == (0,0,100,100)
    print('[test_geometry.boundingbox.convexhull]: PASSED')

    img = ImageDetection(array=np.zeros((128,256), dtype=np.float32)).boundingbox(xmin=10,ymin=20,width=30,height=40).rectangular_mask()
    im = ImageDetection(array=np.float32(img), colorspace='float').boundingbox(xmin=10,ymin=20,width=30,height=40)
    assert np.sum(img) == np.sum(im.crop().array())
    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).rot90cw(128, 256)
    im = ImageDetection(array=np.float32(np.rot90(img,3)), bbox=bb, colorspace='float')
    assert np.sum(img) == np.sum(im.crop().array())
    print('[test_geometry.boundingbox.rot90cw]: PASSED')

    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40).rot90ccw(128, 256)
    im = ImageDetection(array=np.rot90(img,1), bbox=bb)
    assert np.sum(img) == np.sum(im.crop().array())
    print('[test_geometry.boundingbox.rot90ccw]: PASSED')

    bb = BoundingBox(xmin=10, ymin=20, width=30, height=40)
    img = ImageDetection(array=np.zeros((128,256), dtype=np.float32)).boundingbox(bbox=bb).rectangular_mask()
    bb = bb.fliplr(img)
    im = ImageDetection(array=np.fliplr(img), bbox=bb)
    assert np.sum(img) == np.sum(im.crop().array())
    print('[test_geometry.boundingbox.fliplr]: PASSED')

    assert BoundingBox(xmin=-10, ymin=-10, width=30, height=40).hasoverlap(np.zeros((128,256), dtype=np.float32))
    assert not BoundingBox(xmin=1000, ymin=1000, width=30, height=40).hasoverlap(np.zeros((128,256), dtype=np.float32))
    print('[test_geometry.boundingbox.hasoverlap]: PASSED')

    assert BoundingBox(xmin=-10, ymin=-10, width=30, height=40).imclip(np.zeros((128,256), dtype=np.float32)) == BoundingBox(xmin=0, ymin=0, width=20, height=30)
    assert BoundingBox(xmin=-10, ymin=-10, width=30, height=40).imclipshape(128,256) == BoundingBox(xmin=0, ymin=0, width=20, height=30)
    try:
        BoundingBox(xmin=-100, ymin=-100, width=30, height=40).imclip(np.zeros((128,256), dtype=np.float32))
        Failed()
    except Failed:
        raise
    except:
        print('[test_geometry.boundingbox.imclip]: PASSED')

    assert BoundingBox(xmin=-20, ymin=-10, width=30, height=40).aspectratio() == 30.0 / 40.0
    print('[test_geometry.boundingbox.aspectratio]: PASSED')

    assert BoundingBox(xmin=-20, ymin=-10, width=30, height=40).shape() == (40, 30)
    print('[test_geometry.boundingbox.shape]: PASSED')

    assert BoundingBox(xmin=-20, ymin=-10, width=30, height=40).mindimension() == (30)
    assert BoundingBox(xmin=-20, ymin=-10, width=30, height=40).mindim() == (30)
    print('[test_geometry.boundingbox.mindimension]: PASSED')
    print('[test_geometry.boundingbox.mindim]: PASSED')

    BoundingBox(xmin=-20, ymin=-10, width=30, height=40).dict()
    print('[test_geometry.boundingbox]: dict PASSED')
    
    
def test_ellipse():
    e = BoundingBox(xmin=-20, ymin=-10, width=30, height=40).ellipse()

    assert e.inside((10,10))
    assert not e.inside((100,200))
    print('[test_geometry.ellipse.inside]: PASSED')

    assert e.angle() == 0
    assert e.axes() == (30 / 2.0, 40 / 2.0)
    print('[test_geometry.ellipse.dimensions]: PASSED')

    img = e.mask()
    assert img.dtype == bool
    print('[test_geometry.ellipse.mask]: PASSED')


if __name__ == "__main__":
    test_geometry()
    test_boundingbox()
    test_ellipse()
