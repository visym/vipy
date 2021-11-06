import os
import numpy as np
import shutil
from vipy.globals import print
from vipy.util import remkdir, imlist, filetail, istuple, islist, isnumpy, filebase, temphtml, isurl, fileext, tolist, iswebp, isimage, chunklistbysize, ishtml
from vipy.image import Image
from vipy.show import savefig
from collections import defaultdict
from datetime import datetime
import time
import PIL
import vipy.video
import webbrowser
import pathlib
import html


def hoverpixel(urls, outfile=None, pixelsize=32, sortby='color', loupe=True, hoversize=512, ultext=None, display=False, ultextcolor='white', ultextsize='large'):
    """Generate a standalone hoverpixel animation.

    Args:
        urls: a list of urls to publicly accessible images or webp files
        outfile: an html output file, if None a temp file will be used 
        pixelsize: the size of the elements in the montage
        sortby: if 'color' then sort the images rowwise by increasing hue
        loupe: if true, then the magnifier should be a circle
        hoversize: the diameter of the magnifier
        ultext: string to include overlay on the upper left.  include <br> tags for line breaks in string, and standard html text string escaping (e.g. &lt, &gt for <>).  if None, nothing is displayed
        display: if true, then open the html file in the defult browser when done
        ultextcolor: an html color string (e.g. "white", "black") for the upper left text color
        ultextsize: an html font-size string (e.g. "large", "x-large") for the upper left text
    
    Returns:
        A standalone html file that renders each url in montage such that hovering over elements in the montage will show the url in a magnifier.  Try to put chrome in fullscreen for best experience.

    """

    assert outfile is None or ishtml(outfile)
    assert all([isurl(url) and (iswebp(url) or isimage(url)) for url in urls])
    vidlist = [vipy.video.Video(url=url).frame(0).url(url) for url in urls if iswebp(url) and vipy.video.Video(url=url).canload()]
    imlist  = [vipy.image.Image(url=url) for url in urls if isimage(url)]
    imlist = imlist + vidlist
    if sortby == 'color':
        imlist = sorted(imlist, key=lambda im: float(im.clone().resize(16,16,interp='nearest').hsv().channel(0).mean()))  # will load images
        urls = [im.url() for im in imlist]
        
    # Create montage image
    im = montage(imlist, pixelsize, pixelsize, aspectratio=2560/1440, border=1)
    urlarray = chunklistbysize(urls, im.width()//(pixelsize+1))  # including border
    
    # Create HTML visualization that
    filename = outfile if outfile is not None else temphtml()
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.hoverpixel (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))
    f.write('<html>\n')
    f.write('<meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
    style = ('<style>',
             '* {box-sizing: border-box;}',
             '.img-hoverpixel-container {',
             '  position:relative;',
             '}',
             '.img-hoverpixel-glass {',
             '  position: absolute;',
             '  border: 1px solid #999;',
             '  border-radius: 50%;' if loupe else '  border-radius: 0%;',
             '  box-shadow: 2px 2px 6px black;',
             '  background: rgba(0, 0, 0, 0.25);',
             '  cursor: crosshair;',
             '  width: %dpx;' % hoversize,
             '  height: %spx;' % hoversize,
             '}',
             '.upper-left-text {',
             '  position: absolute;',
             '  top: 16px;',
             '  left: 16px;',
             '  color: %s;' % ultextcolor,
             '  font-size: %s;' % ultextsize,
             '  font-family: Arial, Helvetica, sans-serif',
             '}',
             '</style>')    
    f.write('\n'.join(style))

    script = ('<script>',
              'let pixelsize = %d;' % (pixelsize+1),  # include montage border 
              'let urls = %s;' % str(urlarray),  # public URLs for each montage element
              'function hoverpixel(imgID, zoom) {',
              '  var img, glass, w, h, bw;',
              '  img = document.getElementById(imgID);',
              '  glass = document.createElement("DIV");',
              '  glass.setAttribute("class", "img-hoverpixel-glass");',
              '  img.parentElement.insertBefore(glass, img);',
              '  glass.style.backgroundRepeat = "no-repeat";',
              '  bw = 3;',
              '  w = glass.offsetWidth / 2;',              
              '  h = glass.offsetHeight / 2;',
              '  glass.addEventListener("mousemove", moveMagnifier);',
              '  img.addEventListener("mousemove", moveMagnifier);',
              '  glass.addEventListener("touchmove", moveMagnifier);',
              '  img.addEventListener("touchmove", moveMagnifier);',                  
              '  function getCursorPosition(e) {',
              '    var a, x = 0, y = 0;',
              '    e = e || window.event;',
              '    a = img.getBoundingClientRect();',
              '    x = e.pageX - a.left;',
              '    y = e.pageY - a.top;',
              '    x = x - window.pageXOffset;',
              '    y = y - window.pageYOffset;',
              '    return {x : x, y : y};',
              '  }',
              '  function moveMagnifier(e) {',
              '    var pos, x, y;',
              '    e.preventDefault();',
              '    pos = getCursorPosition(e);',
              '    x = pos.x;',
              '    y = pos.y;',
              '    i = Math.floor(pos.x / pixelsize);',  
              '    j = Math.floor(pos.y / pixelsize);'
              '    if (x > img.width) {x = img.width;}',  # truncate to image boundary
              '    if (x < 0) {x = 0;}',
              '    if (y > img.height) {y = img.height;}',
              '    if (y < 0) {y = 0;}',
              '    glass.style.left = (x - w) + "px";',
              '    glass.style.top = (y - h) + "px";',
              '    glass.style.backgroundPosition = "-" + ((0 * zoom) - w + bw) + "px -" + ((0 * zoom) - h + bw) + "px";',                            
              '    glass.style.backgroundImage = "url(\'" + urls[j][i] + "\')";',                           
              '    glass.style.backgroundSize = %d + "px " + %d + "px";' % (hoversize, hoversize),              
              '    glass.style.transform = "scale(" + 1.0/window.devicePixelRatio + "," + 1.0/window.devicePixelRatio + ")";',
              '    glass.style.webkitTransform = "scale(" + 1.0/window.devicePixelRatio + "," + 1.0/window.devicePixelRatio + ")";',
              '    glass.style.mozTransform = "scale(" + 1.0/window.devicePixelRatio + "," + 1.0/window.devicePixelRatio + ")";',                            
              #'    console.log(window.devicePixelRatio);',
              '  }',
              '}',
              '</script>')

    f.write('\n'.join(script))    
    f.write('<body style="background-color:black;>\n')
    f.write('<div class="img-hoverpixel-container">\n')
    f.write(im.html(id="img-hoverpixel", attributes={'width':im.width(), 'height':im.height()}))   # base-64 encoded image with img tag
    f.write('<div class="upper-left-text">%s</div>\n' % ultext if ultext is not None else '')
    f.write('</div>\n')
    f.write('<script>\n')
    f.write('hoverpixel("img-hoverpixel", 1);\n')
    f.write('</script>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()

    if display:
        print('[vipy.visualize.hoverpixel]: Opening "%s" in default browser' % filename)
        webbrowser.open('file://%s' % filename)

    return filename


def mosaic(videos):
    """Create a mosaic iterator from an iterable of videos.
    
    A mosaic is a tiling of videos into a grid such that each grid element is one video.  This function returns an iterator that iterates frames in the video mosaic.
    This mosaic generation can also be performed using ffmpeg, but here we use python iterators to zip a set of videos into a spatial mosaic.  

    >>> for im in vipy.visualize.mosaic( (vipy.video.RandomScene(64,64,32), vipy.video.RandomScene(64,64,32)) )
    >>>     im.show()
    
    Args:
        videos [iterable of `vipy.video.Video`]
    
    Returns:
        A generator which yields frames of the mosaic.  All videos are at their native frame rates, and all videos are anisotropically resized to the (height, width) of the first video

    .. note:: This is the streaming version of `vipy.visualize.video_montage` which requires all videos to be loadable
.
    """
    assert (isinstance(videos, list) and all([isinstance(v, vipy.video.Video) for v in videos])) or isinstance(videos, tuple)
    (H,W) = videos[0].shape()
    for frames in zip(*videos):
        yield vipy.visualize.montage(frames, H, W)


def montage(imlist, imgheight, imgwidth, gridrows=None, gridcols=None, aspectratio=1, crop=False, skip=True, border=1, border_bgr=(128,128,128), do_flush=False, verbose=False):
    """Create a montage image from the of provided list of vipy.image.Image objects.

    Args:
        imlist: [list, tuple] iterable of vipy.image.Image objects which is used to montage rowwise
        imgheight: [int] The height of each individual image in the grid
        imgwidth: [int] the width of each individual image in the grid
        gridrows: [int]  The number of images per row, and number of images per column.  This defines the montage shape.
        gridcols: [int]  The number of images per row, and number of images per column.  This defines the montage shape.
        aspectratio: [float].  This is an optional parameter which defines the shape of the montage as (gridcols/gridrows) without specifying the gridrows, gridcols input
        crop: [bool]  If true, the vipy.image.Image objects should call crop(), which will trigger a load
        skip: [bool]  Whether images should be skipped on failure to load(), useful for lazy downloading
        border: [int]  a border of size in pixels surrounding each image in the grid
        border_bgr [tuple (r,g,b)]:  the border color in a bgr color tuple (b, g, r) in [0,255], uint8
        do_flush: [bool]  flush the loaded images as garbage collection for large montages
        verbose: [bool]  display optional verbose messages

    Returns:
        Return a vipy.image.Image montage which is of size (gridrows*(imgheight + 2*border), gridcols*(imgwidth+2*border))
    
    """

    (n,m) = (imgheight, imgwidth)
    (rows,cols) = (gridrows, gridcols)
    n_imgs = len(imlist)
    M = int(np.ceil(np.sqrt(n_imgs)))
    N = int(np.ceil(n_imgs/M))
    if aspectratio != 1 and aspectratio is not None:
        x = int(round((aspectratio * N - M) / (1 + aspectratio)))
        N = N - x
        M = M + x
    elif rows is not None and cols is not None:
        N = rows
        M = cols
    size = (M * m + ((M + 1) * border), N * n + ((N + 1) * border))
    bc = border_bgr
    img_montage = np.array(PIL.Image.new(mode='RGB', size=size, color=bc))
    k = 0
    for j in range(N):
        for i in range(M):
            if k >= n_imgs:
                break
            sliceM, sliceN = i * (m + border) + border, j * (n + border) + border
            try:
                if crop:
                    if imlist[k].bbox.valid() is False:
                        print('[vipy.visualize.montage] invalid bounding box "%s" ' % str(imlist[k].bbox))
                        if skip is False:
                            print('[vipy.visualize.montage] using original image')
                            im = imlist[k].rgb().resize(n,m).array()
                        else:
                            raise
                    else:
                        im = imlist[k].rgb().crop().resize(height=n, width=m).array()
                else:
                    im = imlist[k].rgb().resize(height=n, width=m).array()

                img_montage[sliceN:sliceN + n, sliceM:sliceM + m] = im

            except KeyboardInterrupt:
                raise
            except Exception as exception:
                print('[vipy.visualize.montage][%d/%d]: skipping "%s"' % (k+1, len(imlist), str(imlist[k])))
                if skip:
                    print('[vipy.visualize.montage][%d/%d]: "%s"' % (k+1, len(imlist), str(exception)))
                else:
                    raise

            if do_flush:
                imlist[k].clone(flush=True)  # clear memory
            if verbose and ((k % 100) == 0):
                print('[vipy.visualize.montage][%d/%d] processing...' % (k, n_imgs))

            k += 1

    if k == 0:
        print('[vipy.visualize.montage] Warning: No images were processed')

    return Image(array=img_montage, colorspace=imlist[0].colorspace())


def videomontage(vidlist, imgheight, imgwidth, gridrows=None, gridcols=None, aspectratio=1, crop=False, skip=True, border=1, border_bgr=(128,128,128), do_flush=False, verbose=True, framerate=30.0, max_duration=None):
    """Generate a video montage for the provided videos by creating a image montage for every frame.  

    Args:
        `vipy.visualize.montage`:  See the args   
        framerate: [float] the framerate of the montage video.  All of the input videos are resampled to this common frame rate
        max_duration: [float] If not None, the maximum diuration of any element in the montage before it cycles

    Returns:
        An video file in outfile that shows each video tiled into a montage.  <Like https://www.youtube.com/watch?v=HjNa7_T-Xkc>

    .. warning:: 
        - This loads every video into memory, so be careful with large montages!
        - If max_duration is not set, then this can result in loading very long video elements in the montage, which will make for long videos
    """
    assert len(vidlist) > 0, "Invalid input"
    assert max_duration is None or max_duration > 0
    assert framerate > 0

    if verbose:
        print('[vipy.visualize.videomontage]: Loading %d videos' % len(vidlist))

    vidlist = [v.framerate(framerate) for v in vidlist]  # resample to a common framerate, this must occur prior to load
    vidlist = [v.load() for v in vidlist]  # triggers load, make sure that the vidlist videos have a reasonably small frames
    max_length = max([len(v) for v in vidlist]) if max_duration is None else int(round(max_duration * framerate))   
    
    if verbose:
        print('[vipy.visualize.videomontage]: Maximum video length (frames) = %d' % (max_length))

    # FIXME: use stream here:
    # with Video(outfile).stream(write=True) as s:
    #     s.write(montage(...))
    
    montagelist = [montage([v[k % len(v)].mindim(max(imgheight, imgwidth)).centercrop(imgheight, imgwidth) for v in vidlist], imgheight, imgwidth, gridrows, gridcols, aspectratio, crop, skip, border, border_bgr, do_flush, verbose=False)
                   for k in range(0, max_length)]
    return vipy.video.Video(array=np.stack([im.array() for im in montagelist]), colorspace='rgb', framerate=framerate)


def urls(urllist, title='URL Visualization', imagewidth=1024, outfile=None, display=False):
    """Given a list of public image URLs, create a stand-alone HTML page to show them all.
    
    Args:
        urllist: [list] A list of urls to display
        title: [str] The title of the html file
        imagewidth: [int] The size of the images in the page
        outfile: [str] The path to the output html file
        display: [bool] open the html file in the default system viewer when complete
    
    """
    urllist = tolist(urllist)
    assert all([isurl(url) for url in urllist])

    # Create summary page to show precomputed images
    k_divid = 0    
    filename = outfile if outfile is not None else temphtml()
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.urls (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))    
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">Title: %s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of URLs: %d<br>\n' % len(urllist))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid)
    k_divid = k_divid + 1

    # Generate images and html
    for url in urllist:
        f.write('<p>\n</p>\n')
        f.write('URL: <a href="%s">%s</a>\n' % (url, url))
        f.write('<br>\n')        
        f.write('<img src="%s" alt="image" width=%d loading="lazy">\n' % (url, imagewidth))        
        f.write('<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')

    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()

    # Display?
    if display:
        url = pathlib.Path(filename).as_uri()
        print('[vipy.visualize.urls]: Opening "%s" in default browser' % url)
        webbrowser.open(url)
        
    return filename

    
    
def tohtml(imlist, imdict=None, title='Image Visualization', mindim=1024, outfile=None, display=False):
    """Given a list of vipy.image.Image objects, show the images along with the dictionary contents of imdict (one per image) in a single standalone HTML file
    
    Args:
        imlist: [list `vipy.image.Image`] 
        imdict: [list of dict] An optional list of dictionaries, such that each dictionary is visualized per image
        title: [str] The title of the html file
        imagewidth: [int] The size of the images in the page
        outfile: [str] The path to the output html file
        display: [bool] open the html file in the default system viewer when complete

    Returns:
        An html file in outfile that contains all the images as a standalone embedded file (no links or external files).
    """

    assert all([isinstance(im, vipy.image.Image) for im in imlist])
    assert imdict is None or (len(imdict) == len(imlist) and isinstance(imdict[0], dict)), "imdict must be one dictionary per image"
        
    # Create summary page to show precomputed images
    k_divid = 0    
    filename = outfile if outfile is not None else temphtml()
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.tohtml (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))    
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">%s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of Images: %d<br>\n' % len(imlist))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid)
    k_divid = k_divid + 1

    # Generate images and html
    for (k,im) in enumerate(imlist):
        # Write out associated dictionary (if provided)
        f.write('<p>\n</p>\n')
        if imdict is not None:
            for (k,v) in imdict[k].items():
                f.write('<b>%s</b>: %s<br>\n' % (html.escape(str(k)), html.escape(str(v))))
        f.write('<br>\n')

        # Write image as base64 encoded string
        im = im.load().mindim(mindim)
        f.write(im.html())   # base-64 encoded image with img tag
        f.write('<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')

    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()

    # Display?
    if display:
        url = pathlib.Path(filename).as_uri()
        print('[vipy.visualize.tohtml]: Opening "%s" in default browser' % url)
        webbrowser.open(url)
        
    return filename


def videolist(vidlist, viddict=None, title='Video Visualization', outfile=None, display=False):
    """Create a standalone HTML file that will visualize the set of videos in vidlist using HTML5 video player"""
    assert all([isinstance(v, vipy.video.Video) for v in vidlist])
        
    # Create summary page to show downloaded videos
    filename = outfile if outfile is not None else temphtml()
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.videolist (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))    
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2048px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">%s</h1><br>\n' % title)
    f.write('Summary HTML generated on %s<br>\n' % time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
    f.write('Number of videos: %d<br>\n' % len(vidlist))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % 0)

    # Generate images and html
    for (k,v) in enumerate(vidlist):
        f.write('<p> <video width="%d" heigh="%d" controls="true"> <source src="%s" type="video/%s"></source>  You need an HTML5 capable browser. </video> </p>\n' % (v.downloadif().width(), v.height(), v.filename(), fileext(v.filename(), withdot=False)))
        f.write('<p>%s</p>\n' % html.escape(str(v)))
        f.write('<hr>\n')        

    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()

    # Display?
    if display:
        url = pathlib.Path(filename).as_uri()
        print('[vipy.visualize.videolist]: Opening "%s" in default browser' % url)
        webbrowser.open(url)
        
    return filename

    
    
def imagelist(list_of_image_files, outdir, title='Image Visualization', imagewidth=64):
    """Given a list of image filenames wth absolute paths, copy to outdir, and create an index.html file that visualizes each.    
    """

    # FIXME: should this just call tohtml?

    k_divid = 0

    # Create summary page to show precomputed images
    outdir = remkdir(outdir)
    filename = os.path.join(remkdir(outdir), 'index.html')
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.imagelist (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))    
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">Title: %s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of Images: %d<br>\n' % len(list_of_image_files))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid)
    k_divid = k_divid + 1

    # Generate images and html
    assert all([os.path.exists(f) and vipy.util.isimg(f) for f in list_of_image_files])
    for (k, imsrc) in enumerate(list_of_image_files):
        shutil.copyfile(imsrc, os.path.join(outdir, filetail(imsrc)))
        imdst = filetail(imsrc)
        f.write('<p>\n</p>\n')
        f.write('<b>Filename: %s</b><br>\n' % imdst)
        f.write('<br>\n')
        f.write('<img src="%s" alt="image" width=%d loading="lazy">\n' % (imdst, imagewidth))
        f.write('<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')

    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()
    return filename


def imagetuplelist(list_of_tuples_of_image_files, outdir, title='Image Visualization', imagewidth=64):
    """Imagelist but put tuples on same row"""
    k_divid = 0

    # Create summary page to show precomputed images
    outdir = remkdir(outdir)
    filename = os.path.join(remkdir(outdir), 'index.html')
    f = open(filename,'w')
    f.write('<!DOCTYPE html>\n')
    f.write('<!--\n    Visym Labs\n    vipy.visualize.imagetuplelist (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))    
    f.write('<html>\n')
    f.write('<body>\n')
    f.write('<div id="container" style="width:2400px">\n')
    f.write('<div id="header">\n')
    f.write('<h1 style="margin-bottom:0;">Title: %s</h1><br>\n' % title)
    localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    f.write('Summary HTML generated on %s<br>\n' % localtime)
    f.write('Number of Tuples: %d<br>\n' % len(list_of_tuples_of_image_files))
    f.write('</div>\n')
    f.write('<br>\n')
    f.write('<hr>\n')
    f.write('<div id="%04d" style="float:left;">\n' % k_divid)
    k_divid = k_divid + 1

    # Generate images and html
    for (k, imsrclist) in enumerate(list_of_tuples_of_image_files):
        f.write('<p>\n</p>\n')
        for imsrc in imsrclist:
            shutil.copyfile(imsrc, os.path.join(outdir, filetail(imsrc)))
            imdst = filetail(imsrc)
            f.write('<b>Filename: %s</b><br>\n' % imdst)
        f.write('<p>\n</p>\n')
        f.write('<br>\n')
        for imsrc in imsrclist:
            imdst = filetail(imsrc)
            f.write('<img src="%s" alt="image" width=%d loading="lazy">' % (imdst, imagewidth))
        f.write('\n<p>\n</p>\n')
        f.write('<hr>\n')
        f.write('<p>\n</p>\n')

    f.write('</div>\n')
    f.write('</body>\n')
    f.write('</html>\n')
    f.close()
    return filename
