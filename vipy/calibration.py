import numpy as np
import vipy.image


def checkerboard(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with black and white colors with black in upper left and bottom right, return np.array() float32 in [0,1]"""
    img = None
    for i in range(0,nrows):
        row = np.hstack([float((j + i) % 2) * np.ones((dx,dy)) for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def red_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with red colors, return Image()"""
    red = checkerboard(dx, dy, nrows, ncols)
    return vipy.image.Image(array=np.uint8(255*np.dstack( (red, np.zeros_like(red), np.zeros_like(red)))))


def blue_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with red colors, return Image()"""
    blue = checkerboard(dx, dy, nrows, ncols)
    return vipy.image.Image(array=np.uint8(255*np.dstack( (np.zeros_like(blue), np.zeros_like(blue), blue))))


def color_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with random colors, return Image()"""
    return vipy.image.Image(array=np.uint8(255*np.random.rand(nrows, ncols, 3)), colorspace='rgb').resize(dx*nrows, dy*ncols, interp='nearest')


def color_checkerboard(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with random colors", return np.array"""
    return color_checkerboard_image(dx, dy, nrows, ncols).array()


def testimage():
    """Return an Image() object of a superb owl from wikipedia"""
    return vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/1/11/Horned1b.jpg').minsquare().mindim(512)

def owl():
    """Return an Image() object of a superb owl from wikipedia"""
    return vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/1/11/Horned1b.jpg').minsquare().mindim(512)

def randomimage():
    return vipy.image.Image(array=np.uint8(255*np.random.rand(512,512,3)), colorspace='rgb')

def testimg():
    """Return a numpy array for a superb owl"""
    return testimage().array()


def tile(T, nrows=16, ncols=16):
    """Create a 2D tile pattern with texture T repeated (M,N) times"""
    img = None
    for i in range(0,nrows):
        row = np.hstack([T for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def greenblock(dx, dy):
    """Return an (dx, dy, 3) float64 RGB channel image with green channel=1.0"""        
    img = np.zeros((dx,dy,3))
    img[:,:,1] = 1.0
    return img


def redblock(dx, dy):
    """Return an (dx, dy, 3) float64 RGB channel image with red channel=1.0"""    
    img = np.zeros((dx,dy,3))
    img[:,:,0] = 1.0
    return img


def blueblock(dx, dy):
    """Return an (dx, dy, 3) float64 RGB channel image with blue channel=1.0"""
    img = np.zeros((dx,dy,3))
    img[:,:,2] = 1.0
    return img


def bayer(dx, dy, M=16, N=16):
    """Return an (M,N) tiled texture pattern of [blue, green, blue, green; green red green red; blue green blue green, green red green red] such that each subblock element is (dx,dy) and the total repeated subblock size is (4*dx, 4*dy)"""
    T = np.vstack([np.hstack([blueblock(dx,dy), greenblock(dx,dy), blueblock(dx,dy), greenblock(dx,dy)]),
                   np.hstack([greenblock(dx,dy), redblock(dx,dy), greenblock(dx,dy), redblock(dx,dy)]),
                   np.hstack([blueblock(dx,dy), greenblock(dx,dy), blueblock(dx,dy), greenblock(dx,dy)]),
                   np.hstack([greenblock(dx,dy), redblock(dx,dy), greenblock(dx,dy), redblock(dx,dy)])])
    return tile(T, M, N)


def bayer_image(dx, dy, M=16, N=16):
    """Return bayer pattern as Image()"""
    return vipy.image.Image(array=np.uint8(255.0*bayer(dx, dy, M, N)), colorspace='rgb')


def dots(dx=16, dy=16, nrows=8, ncols=8):
    """Create a sequence of dots (e.g. single pixels on black background) with strides (dx, dy) and image of size (dx*ncols,dy*nrows)"""

    imdot = np.zeros((dx,dy))
    imdot[int(np.floor(dx / 2.0)), int(np.floor(dy / 2.0))] = 1.0
    img = None
    for i in range(0,nrows):
        row = np.hstack([imdot for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def vertical_gradient(nrows, ncols):
    """Create a 2D linear ramp image """
    return np.outer([(255.0 * (x / float(nrows))) for x in range(0,nrows)], np.ones((1,ncols))).astype(np.uint8)

def centersquare(height=512, width=512, squaresize=256):
    img = np.zeros( (height, width) )
    (x,y,s) = (int(height//2), int(width//2), int(squaresize//2))
    img[x-s:x+s, y-s:y+s] = 1.0
    return img

def centersquare_image(height=512, width=512, squaresize=256):    
    return vipy.image.Image(array=np.uint8(255*centersquare(height, width, squaresize)), colorspace='lum')
