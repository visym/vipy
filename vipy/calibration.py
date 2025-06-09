import numpy as np
import vipy.image


def checkerboard(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with black and white colors with black in upper left and bottom right.

    Returns:
        2D numpy array np.array() float32 in [0,1]
    """
    img = None
    for i in range(0,nrows):
        row = np.hstack([float((j + i) % 2) * np.ones((dx,dy)) for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def red_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with red colors.

    Returns:
        vipy.image.Image
    """
    red = checkerboard(dx, dy, nrows, ncols)
    return vipy.image.Image(array=np.uint8(255*np.dstack( (red, np.zeros_like(red), np.zeros_like(red)))))


def blue_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with blue colors.
    
    Returns:
        vipy.image.Image
    """
    blue = checkerboard(dx, dy, nrows, ncols)
    return vipy.image.Image(array=np.uint8(255*np.dstack( (np.zeros_like(blue), np.zeros_like(blue), blue))))


def color_checkerboard_image(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with random colors
    
    Returns:
        vipy.image.Image
    """
    return vipy.image.Image(array=np.uint8(255*np.random.rand(nrows, ncols, 3)), colorspace='rgb').resize(dx*nrows, dy*ncols, interp='nearest')


def color_checkerboard(dx=16, dy=16, nrows=8, ncols=8):
    """Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx*ncols,dy*nrows) with random colors.
    
    Returns:
        3D numpy array with three channels, uint8
    """
    return color_checkerboard_image(dx, dy, nrows, ncols).array()


def testimage():
    """Return a `vipy.image.Image` object of a superb owl from wikipedia"""
    return vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/1/11/Horned1b.jpg').minsquare().mindim(512)

def owl():
    """Return a `vipy.image.Image` object of a superb owl from wikipedia"""
    return vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/1/11/Horned1b.jpg').minsquare().mindim(512)

def randomimage(m=512, n=512):
    """Return a uniform random RGB image as uint8 numpy array of size (m,n,3)"""
    assert m > 0 and n > 0
    return vipy.image.Image(array=np.uint8(255*np.random.rand(m,n,3)), colorspace='rgb')

def testimg():
    """Return a numpy array for `vipy.calibration.testimage` of a superb owl"""
    return testimage().array()


def tile(T, nrows=16, ncols=16):
    """Create a 2D tile pattern with texture T repeated (nrows, ncols) times.
    
    Returns:
        float32 numpy array of size (T.shape[0]*nrows, T.shape[1]*ncols)
    """
    img = None
    for i in range(0,nrows):
        row = np.hstack([T for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def greenblock(dx, dy):
    """Return an (dx, dy, 3) numpy array float64 RGB channel image with green channel=1.0"""        
    img = np.zeros((dx,dy,3))
    img[:,:,1] = 1.0
    return img


def redblock(dx, dy):
    """Return an (dx, dy, 3) numpy array float64 RGB channel image with red channel=1.0"""    
    img = np.zeros((dx,dy,3))
    img[:,:,0] = 1.0
    return img


def blueblock(dx, dy):
    """Return an (dx, dy, 3) numpy array float64 RGB channel image with blue channel=1.0"""
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
    """Return `vipy.calibration.bayer` as `vipy.image.Image`"""
    return vipy.image.Image(array=np.uint8(255.0*bayer(dx, dy, M, N)), colorspace='rgb')


def dots(dx=16, dy=16, nrows=8, ncols=8):
    """Create a sequence of dots (e.g. single pixels on black background) separated by strides (dx, dy) with image of size (dx*ncols,dy*nrows)
    
    Returns:
        float32 numpy array in range [0,1]
    """

    imdot = np.zeros((dx,dy))
    imdot[int(np.floor(dx / 2.0)), int(np.floor(dy / 2.0))] = 1.0
    img = None
    for i in range(0,nrows):
        row = np.hstack([imdot for j in range(0, ncols)])
        img = np.vstack((img, row)) if img is not None else row
    return img.astype(np.float32)


def vertical_gradient(nrows, ncols):
    """Create 2D linear ramp image with the ramp increasing from top to bottom.

    Returns:
        uint8 numpy array of size (nrows, ncols) with veritical gradient increasing over rows
    """
    return np.outer([(255.0 * (x / float(nrows))) for x in range(0,nrows)], np.ones((1,ncols))).astype(np.uint8)

def centersquare(height=512, width=512, squaresize=256, channels=1):
    """Create a white square on a black background of an image of shape (width, height).

     Returns:
         numpy array of appropriate channels of float64 in [0,1]
    """
    img = np.zeros( (height, width, channels) )
    (x,y,s) = (int(height//2), int(width//2), int(squaresize//2))
    img[x-s:x+s, y-s:y+s] = 1.0
    return img

def centersquare_border(height=512, width=512, squaresize=None, channels=1, asimage=False):
    """Create a white square with black interior on a black background of an image of shape (width, height).  The border of the square has width of a single pixel

     Returns:
         numpy array of appropriate channels of float32 in [0,1]
         asimage=True: returns a vipy.image.Image object 
    """
    img = np.zeros( (height, width, channels), dtype=np.float32)
    (x,y,s) = (int(height//2), int(width//2), int(squaresize//2) if squaresize is not None else int(min(height, width)//4))
    img[x-s, y-s:y+s+1] = 1.0
    img[x+s, y-s:y+s+1] = 1.0    
    img[x-s:x+s+1, y-s] = 1.0
    img[x-s:x+s+1, y+s] = 1.0    
    return img if not asimage else vipy.image.Image(array=np.uint8(img*255), colorspace='lum')

def centersquare_image(height=512, width=512, squaresize=256):    
    """Returns `vipy.image.Image` for `vipy.calibration.centersquare` numpy array"""
    return vipy.image.Image(array=np.uint8(255*centersquare(height, width, squaresize, channels=1)), colorspace='lum')

def imcentersquare(height=512, width=512, squaresize=256):
    """alias for `vipy.calibration.centersquare_image`"""
    return centersquare_image(height, width, squaresize)

def circle(x, y, r, width, height, channels=1):
    """Create a white circle on a black background centered at (x,y) with radius r pixels, of shape (width, height).  
    
    Returns:
        numpy array of appropriate channels of float32 in [0,1]
    """
    img = np.zeros( (height, width, channels) if channels!=1 else (height,width), dtype=np.float32 )
    (X,Y) = np.meshgrid(range(width), range(height))
    img[np.sqrt((X-x)**2 + (Y-y)**2) < r] = 1.0
    return img

def imcircle(x, y, r, width, height, channels=1):
    """Create a white circle on a black background centered at (x,y) with radius r pixels, of shape (width, height).  
    
    Returns:
        `vipy.image.Image` object with array defined by `vipy.calibration.circle`
    """    
    return vipy.image.Image(array=circle(x, y, r, width, height, channels))


def imstep(width, height):
    """Create a left black/right white step image of size (width, height)
    
    Returns:
        `vipy.image.Image` object
    """    
    img = np.zeros( (height, width, 1), dtype=np.float32 )
    img[:, -width//2:] = 1
    return vipy.image.Image(array=img)
    
