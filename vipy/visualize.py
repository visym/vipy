import os
import vipy
import numpy as np
import shutil
from vipy.globals import log
from vipy.util import remkdir, imlist, filetail, filebase, temphtml, isurl, fileext, tolist, iswebp, isimage, chunklistbysize, ishtml, escape_string_for_innerHTML
from vipy.image import Image
from vipy.show import savefig
from collections import defaultdict
from datetime import datetime
from vipy.metrics import ascii_bar_chart
from pathlib import Path
import time
import PIL.Image
import vipy.video
import webbrowser
import html
import urllib
import warnings
import hashlib
import matplotlib
import json
import html


def scene_explorer(im, outfile=None, width=None, title='Scene Explorer', previewurl=None, keypoint_alpha=0.7, popup_alpha=0.8, embed=True, open_in_browser=False, keypoint_radius=None, max_keypoint_radius=0.1,
                   tag_formatter=lambda im: "<strong>Image Tags</strong><br><br>%s" % ('<br>'.join(im.tags()) if not im.has_soft_tags() else ascii_bar_chart(im.soft_tags(), 64)),
                   attribute_formatter=lambda im: "<strong>Image Attributes</strong><br><br>Object counts: %s<br><br>Image attributes: %s" % (json.dumps(vipy.dataset.Dataset(im.objects()).frequency(lambda im: im.category()), indent=2),
                                                                                                                                              json.dumps(im.clone().flush().clear().json(encode=False), indent=2)),
                   description_formatter=lambda im: "<strong>Image Description</strong><br><br>%s" % 'None',
                   category_to_color={}):                   
    
    """Generate a standalone scene_explorer visualization.

    A scene_explorer visualization is a standalone HTML file that shows a single `vipy.image.Scene` object with an interactive search and visualization of all objects, attributes, captions and tags

    Args:
        im: a `vipy.image.Scene` object for visualization
        outfile: an html output file, if None a temp file with an html extension will be used 
        title [str]: the title of the html file
        previewurl [str]: A url containing an image thumbnail of what the html file contains.  Useful for previewing the file contents if shared on social media
        keypoint_alpha [float]: the keypoint transparency in the range (0,1)
        popup_alpha [float]: the popup window transparency in the range (0,1)
        embed [bool]: If true, embed the image as base64 encoded image.  If false, use the provided url in the image
        open_in_browser [bool]: If true, open the html file in the default browser on the current system
        tag_formatter [lambda]:  A lambda function with a single argument equal to the input image that returns a HTML string for display in the tag popup.  This can be any valid html.
        attribute_formatter [lambda]:  A lambda function with a single argument equal to the input image that returns a HTML string for display in the attribute popup.  This can be any valid html
        description_formatter [lambda]:  A lambda function with a single argument equal to the input image that returns a HTML string for display in the description popup.  This can be any valid html    
        width [float]: A fraction of the browser width that this image should take up (default=1)
        keypoint_radius [float]: All keypoints should have the same radius for display, specified as a fraction of the image width for each keypoint radius (default=None)
        max_keypoint_radius [float]: All keypoints should have a maximum radius for display, specified as a fraction of the image width for each keypoint (default=0.1). 
        category_to_color [dict]: A dict mapping the category to a keypoint color as a hex string (e.g. {'person':'#FF0000'} for red)
    
    Returns:
        outfile

    Current support for `vipy.object.Keypoint2d` only, `vipy.object.Detection` are ignored
    
    """
    assert isinstance(im, vipy.image.Scene)
    assert outfile is None or ishtml(outfile)
    assert previewurl is None or isurl(previewurl)
    assert keypoint_alpha >=0 and keypoint_alpha <=1
    assert popup_alpha >=0 and popup_alpha <=1
    assert embed or im.has_url()
    assert width is None or (width>=0 and width<=1)
    assert keypoint_radius is None or (keypoint_radius>=0 and keypoint_radius<=1)    
    if not all([isinstance(o, vipy.object.Keypoint2d) for o in im.objects()]):
        log.warning('Scene explorer supports vipy.object.Keypoint2d only - vipy.object.Detection will be converted to keypoints')

    imc = im.clone()
    img = ('data:image/jpeg;charset=utf-8;base64,%s' % imc.load().base64().decode('ascii')) if embed else imc.flush().url()
    colors = [matplotlib.colors.to_hex(c) for c in vipy.show.colorlist()]
    keypoints = [vipy.object.Keypoint2d.cast(o) for o in imc.objects()]
    if keypoint_radius:
        keypoints = [o.set_radius(keypoint_radius*imc.width()) for o in keypoints]
    if max_keypoint_radius != 1:
        keypoints = [o.set_radius(min(o.r, max_keypoint_radius*imc.width())) for o in keypoints]
    d_category_to_color = {o.category():colors[int(hashlib.sha1(o.category().split(' ')[-1].encode('utf-8')).hexdigest(), 16) % len(colors)] for o in keypoints} | category_to_color   # consistent color mapping by category suffix (space separated)
    
    html = Path(Path(__file__).parent.resolve() / 'visualize_scene_explorer.html').read_text()
    keywords = {'${TITLE}': title,
                '${META_OG_IMAGE}': ('<meta property="og:image" content="%s">' % previewurl) if previewurl is not None else '',  # useful to show as a preview of the HTML contents for shared links
                '${IMG_CONTAINER_WIDTH}':'100%' if width is None else str(int(100*width))+'%',
                '${IMG_CONTAINER_HEIGHT}':'100%',                
                '${IMG_WIDTH}':str(imc.width()),
                '${IMG_HEIGHT}':str(imc.height()),
                '${IMG}':img,
                '${IMG_ATTRIBUTES}':escape_string_for_innerHTML(attribute_formatter(im)),
                '${IMG_DESCRIPTION}': escape_string_for_innerHTML(description_formatter(im)),
                '${IMG_TAGS}': escape_string_for_innerHTML(tag_formatter(im)),
                '${SEARCHBOX_WIDTH}':str(1024 // 4),
                '${POPUP_ALPHA}': str(popup_alpha),
                '${CLASSLIST}':str(sorted(set([o.category() for o in keypoints]))),  # assumes that shared class prefix encodes grouping
                '${KP_X}':str([o.x for o in keypoints]),
                '${KP_Y}':str([o.y for o in keypoints]),
                '${KP_R}':str([o.r for o in keypoints]),
                '${KP_ALPHA_HEX}':str(format(int(keypoint_alpha*255), '02x')),  # keypoint face alpha = [0,255] -> [00, FF]
                '${KP_CLASSLIST}':str([o.category() for o in keypoints]),
                '${KP_ATTRIBUTELIST}':str([o.json() for o in keypoints]),
                '${KP_COLORLIST}':str([d_category_to_color[o.category()] for o in keypoints])}    
    for (k,v) in keywords.items():
        html = html.replace(k,v)

    filename = outfile if outfile is not None else temphtml()        
    with open(filename, 'w') as f:
        f.write(html)

    if open_in_browser:
        os.system('open %s' % filename)
        
    return filename
    

def hoverpixel(urls, outfile=None, pixelsize=32, sortby='color', loupe=True, hoversize=512, ultext=None, display=False, ultextcolor='white', ultextsize='large', aspectratio=2560/1440, ultextoffset=(16,16)):
    """Generate a standalone hoverpixel visualization.

    A hoverpixel visualization is an HTML file that shows a montage such that each pixel in the montage is a video or image.  
    When the user hovers the mouse over a pixel in the montage, a high resolution magnified popup for the corresponding video or image is displayed.  
    This is a way to visualize and explore large datasets, showing the entire dataset in a glance, but allowing the user to zoom into specific sections.
    Each of the magnified media elements must be publicly accessible as a URL.  If the URL is a webp animation, then the magnified media will show 
    the animation (if the browser supports it).  

    Args:
        urls: a list of urls to publicly accessible images or webp files.  These are the URLs that are used to display inside the magnified popup.
        outfile: an html output file, if None a temp file will be used 
        pixelsize [int]: the square size of the elements in the montage
        aspectratio [float]: The ratio of columns/rows in the pixel montage. Set to 1 for square.
        sortby: if 'color' then sort the images rowwise by increasing hue, if None, then use provided ordering of URLs
        loupe: if true, then the magnifier should be a circle
        hoversize: the diameter of the magnifier
        ultext: string to include overlay on the upper left.  include <br> tags for line breaks in string, and standard html text string escaping (e.g. &lt, &gt for <>).  if None, nothing is displayed
        display: if true, then open the html file in the defult browser when done
        ultextcolor: an html color string (e.g. "white", "black") for the upper left text color
        ultextsize: an html font-size string (e.g. "large", "x-large") for the upper left text
        ultextoffset [tuple): An offset in (x=pixels, y=pixels) of the origin of the upper left text

    Returns:
        A standalone html file that renders each url in montage such that hovering over elements in the montage will show the url in a magnifier.  

    .. note:: 
        - Use fullscreen browsing for best experience.
        - Try to zoom out in your browser to see the full montage.
        - Wrap this HTML output file in an iframe for integration into your website: <iframe src="https://path/to/my/outfile.html" style="width: 1024px; height: 768px; border: 0px;" allowfullscreen></iframe>

    """

    assert outfile is None or ishtml(outfile)
    assert all([isurl(url) and (iswebp(url) or isimage(url)) for url in urls])
    vidlist = [vipy.video.Video(url=url).frame(0).resize(pixelsize, pixelsize).url(url) for url in urls if iswebp(url) and vipy.video.Video(url=url).canload()]  # small frame=0 stored in memory
    imlist  = [vipy.image.Image(url=url) for url in urls if isimage(url)]  # not loaded
    imlist = imlist + vidlist
    if sortby == 'color':
        imlist = sorted(imlist, key=lambda im: float(im.clone().resize(16,16,interp='nearest').hsv().channel(0).mean()))  # will load images (but not store)
        urls = [im.url() for im in imlist]
        
    # Create montage image
    im = montage(imlist, pixelsize, pixelsize, aspectratio=aspectratio, border=1, do_flush=True, verbose=True)
    urlarray = chunklistbysize(urls, im.width()//(pixelsize+1))  # including border

    # Create HTML+javascript visualization file
    filename = outfile if outfile is not None else temphtml()
    with open(filename, 'w') as f:
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
                 '  top: %dpx;' % ultextoffset[0],
                 '  left: %dpx;' % ultextoffset[1],
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
                  '  }',
                  '}',
                  '</script>')
        
        f.write('\n'.join(script))    
        f.write('<body style="background-color:black;">\n')
        f.write('<div class="img-hoverpixel-container">\n')
        f.write(im.html(id="img-hoverpixel", alt='hoverpixel', attributes={'hoverpixelwidth':im.width(), 'hoverpixelheight':im.height(), 'width':im.width(), 'height':im.height()}))   # base-64 encoded image with img tag, extra attributes for hoverpixel_selector() 
        f.write('<div class="upper-left-text">%s</div>\n' % ultext if ultext is not None else '')
        f.write('</div>\n')
        f.write('<script>\n')
        f.write('hoverpixel("img-hoverpixel", 1);\n')
        f.write('</script>\n')
        f.write('</body>\n')
        f.write('</html>\n')

    if display:
        log.info('[vipy.visualize.hoverpixel]: Opening "%s" in default browser' % filename)
        webbrowser.open('file://%s' % filename)

    return filename


def hoverpixel_selector(htmllist, legendlist, outfile=None, display=False, offset=(60, 80), fullscreen=True, ultext=None, ultextcolor='white', ultextsize='large', ultextoffset=(16,16)):
    """Create a dropdown selector of hoverpixel visualizations by legend.
    
    Args:
        htmllist: a list of HTML files output from `vipy.visualize.hoverpixel` which have been created with a specific ordering 
        legendlist: a list of strings describing how the hoverpixel was sorted (e.g. Color, Category, Size, Region)
        offset (x,y): A tuple of integers (x=right,y=down) for the absolute position in pixel units of the dropdown menu relative to the upper left of the page.
        fullscreen [bool]: If true, include an expand button next to the selector to put the hoverpixel iframe into fullscreen mode.  Be sure to add allowfullscreen to the iframe.

    Returns:
        An HTML file that loads the hoverpixel HTML in an iframe with an overlaid dropdown menu to select which hoverpixel animation is displayed

    ..note:: The fullscreen display will not work on iPhone since it is unsupported.
    """

    assert len(htmllist) == len(legendlist)
    assert all([ishtml(h) for h in htmllist])
    if any([not isurl(h) for h in htmllist]):
        log.warning('Local HTML files will not load in safari due to security restrictions - use a public https:// URL instead, or use a different browser (e.g. chrome, firefox)')    
    htmllist = [('file://%s' % os.path.abspath(h)) if not isurl(h) else h for h in htmllist]
    
    filename = outfile if outfile is not None else temphtml()
    with open(filename, 'w') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<!--\n    Visym Labs\n    vipy.visualize.hoverpixel_selector (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))
        if fullscreen:
            f.write('<head>\n  <link href="https://use.fontawesome.com/releases/v5.13.0/css/all.css" rel="stylesheet">\n</head>\n')        
        f.write('<html>\n')

        f.write('<body style="background-color:black;">\n')

        html = str(urllib.request.urlopen(htmllist[0]).read())
        assert 'hoverpixelwidth' in html and 'hoverpixelheight' in html  # must be vipy.visualize.hoverpixel() output
        (width, height) = (int(html.split('hoverpixelwidth=')[1].split(' ')[0].replace('"','')), int(html.split('hoverpixelheight=')[1].split(' ')[0].replace('"','')))  # unique attribute search for <img key="val" key2="val2" key3=...>
        f.write('<iframe id="hoverpixelframe" src="%s" style="width:%dpx; height:%dpx; border:0px; visibility:hidden;" onload="this.style.visibility=\'visible\';" allowfullscreen></iframe>\n' % (htmllist[0], width+20, height+20))               
        f.write('<select title="Sort by" id="selector" onchange="seturl(this.value)" style="position: absolute; left:%dpx; top:%dpx; padding: 1px 4px; border-radius:4px; visibility:visible;">\n' % (offset[0], offset[1]))
        for (k,legend) in enumerate(legendlist):
            f.write('  <option value="%s"%s>%s</option>\n' % (legend, ' selected="selected"' if k==0 else '', legend))
        f.write('</select>\n')        
        f.write('<i title="Fullscreen" id="expand" class="fas fa-expand fa-lg" onclick="tofullscreen()" style="color:white; position:absolute; left:163px; top:83px; visibility:%s;" onmouseover="this.style.color=\'black\';" onmouseout="this.style.color=\'white\';" ></i>\n' % ('hidden' if not fullscreen else 'visible'))
        f.write('<i title="Spinner" id="spinner" class="fas fa-spinner fa-spin fa-lg"" style="color:white; position:absolute; left:200px; top:83px; visibility:hidden;"></i>\n')
        if ultext is not None:
            f.write('<div style="position:absolute; top:%dpx; left:%dpx; color:%s; font-size:%s; font-family:Arial, Helvetica, sans-serif;">%s</div>\n' % (ultextoffset[0], ultextoffset[1], ultextcolor, ultextsize, ultext))
        f.write('<script>\n')
        f.write("  var iframe = document.getElementById('hoverpixelframe');\n")
        f.write("  var selector = document.getElementById('selector');\n")
        f.write("  var expand = document.getElementById('expand');\n")
        f.write("  var spinner = document.getElementById('spinner');\n")
        f.write("  function tofullscreen() { if (iframe.requestFullscreen) {iframe.requestFullscreen();} else if (iframe.webkitRequestFullscreen) {iframe.webkitRequestFullscreen();} }\n")
        f.write('  function loading() {\n')
        f.write("    spinner.style.visibility='visible'; iframe.addEventListener('load', function() { spinner.style.visibility='hidden';});\n")
        f.write("  }\n")
        f.write('  function seturl(x) {\n')
        for (url, legend) in zip(htmllist, legendlist):
            f.write('    if (x == "%s") { loading(); iframe.src = "%s"; }\n' % (legend, url))
        f.write('  }\n')
        f.write("  loading();\n")        
        f.write('</script>\n')
        f.write('</body>\n')
        f.write('</html>\n')

    if display:
        log.info('[vipy.visualize.hoverpixel_selector]: Opening "%s" in default browser' % filename)
        webbrowser.open('file://%s' % filename)

    return filename


def mosaic(videos, gridrows=None, gridcols=None):
    """Create a mosaic iterator from an iterable of videos.
    
    A mosaic is a tiling of videos into a grid such that each grid element is one video.  This function returns an iterator that iterates frames in the video mosaic.
    This mosaic generation can also be performed using ffmpeg, but here we use python iterators to zip a set of videos into a spatial mosaic.  

    >>> for im in vipy.visualize.mosaic( (vipy.video.RandomScene(64,64,32), vipy.video.RandomScene(64,64,32)) )
    >>>     im.show()
    
    Args:
        videos [iterable of `vipy.video.Video`]
    
    Returns:
        A generator which yields frames of the mosaic.  All videos are at their native frame rates, and all videos are anisotropically resized to the (height, width) of the first video

    .. note:: This is the streaming version of `vipy.visualize.videomontage` which requires all videos to be loadable
.
    """
    assert (isinstance(videos, list) and all([isinstance(v, vipy.video.Video) for v in videos])) or isinstance(videos, tuple)
    (H,W) = videos[0].shape()
    for frames in zip(*videos):        
        yield vipy.visualize.montage(frames, H, W, gridrows=gridrows, gridcols=gridcols)

        
def videomosaic(videos, gridrows=None, gridcols=None):
    """Return a mosaic video from an iterable of videos.  

    A mosaic will output a video montage such that all videos are exactly the same length, without cycling.  This assumes that the videos are showing the same timestamps like in a security mosaic view.  

    .. note:: This will generate a framewise videomontage, and large videos can result in out of memory condition.

    """
    return vipy.video.Video(frames=list(mosaic(videos, gridrows=gridrows, gridcols=gridcols)), framerate=videos[0].framerate())


def montage(imlist, imgheight=256, imgwidth=256, gridrows=None, gridcols=None, aspectratio=1, crop=False, skip=True, border=1, border_rgb=(128,128,128), do_flush=False, verbose=False):
    """Create a montage image from the of provided list of `vipy.image.Image` objects.

    >>> vipy.visualize.montage([[im.crop() for im in vipy.image.vehicles()]]).show()

    Args:
        imlist: [list, tuple, vipy.dataset.Dataset] iterable of `vipy.image.Image` objects which is used to montage rowwise, or a list of lists such that each element defines `vipy.image.Image` objects on a grid row.
        imgheight: [int] The height of each individual image in the grid, defaults to 256 px
        imgwidth: [int] the width of each individual image in the grid, defaults to 256 px, use centersquare() for isotropic scaling
        gridrows: [int]  The number of images per row, and number of images per column.  This defines the montage shape.
        gridcols: [int]  The number of images per row, and number of images per column.  This defines the montage shape.
        aspectratio: [float].  This is an optional parameter which defines the shape of the montage as (gridcols/gridrows) without specifying the gridrows, gridcols input
        crop: [bool]  If true, the vipy.image.Image objects should call crop(), which will trigger a load
        skip: [bool]  Whether images should be skipped on failure to load(), useful for lazy downloading
        border: [int], (leftright, topbottom)  a border of size in pixels surrounding each image in the grid, may be a tuple specifying top/bottom border and left/right border
        border_rgb [tuple (r,g,b)]:  the border color in a rgb color tuple (r,g,b) in [0,255], uint8
        do_flush: [bool]  flush the loaded images as garbage collection for large montages
        verbose: [bool]  display optional verbose messages

    Returns:
        Return a `vipy.image.Image` montage which is of size (gridrows*(imgheight + 2*border), gridcols*(imgwidth+2*border))
        
    """
    assert isinstance(imlist, (list, tuple, vipy.dataset.Dataset)) and len(imlist)>0 and all([isinstance(im, vipy.image.Image) for im in imlist])
    
    (rows, cols) = (gridrows, gridcols) if isinstance(imlist[0], vipy.image.Image) else (len(imlist), max([len(r) if isinstance(r, list) or isinstance(r, tuple) else 1  for r in imlist]))
    imlist = vipy.util.flatlist(imlist)
    (n,m) = (imgheight, imgwidth)
    (ud, lr) = (border) if isinstance(border, (tuple, list)) and len(border)==2 else (border, border)
    
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
    size = (M * m + ((M + 1) * ud), N * n + ((N + 1) * lr))
    bc = border_rgb
    img_montage = np.array(PIL.Image.new(mode='RGB', size=size, color=bc))
    k = 0
    for j in range(N):
        for i in range(M):
            if k >= n_imgs:
                break
            sliceM, sliceN = i * (m + ud) + ud, j * (n + lr) + lr
            try:
                if crop:
                    if imlist[k].bbox.valid() is False:
                        log.warning('[vipy.visualize.montage] invalid bounding box "%s" ' % str(imlist[k].bbox))
                        if skip is False:
                            log.warning('[vipy.visualize.montage] using original image')
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
                log.warning('[vipy.visualize.montage][%d/%d]: skipping "%s"' % (k+1, len(imlist), str(imlist[k])))
                if skip:
                    log.warning('[vipy.visualize.montage][%d/%d]: "%s"' % (k+1, len(imlist), str(exception)))
                else:
                    raise

            if do_flush:
                imlist[k].flush()  # clear memory
            if verbose and ((k % 100) == 0):
                log.info('[vipy.visualize.montage][%d/%d] processing...' % (k, n_imgs))

            k += 1

    if k == 0:
        log.warning('[vipy.visualize.montage] No images were processed')

    return Image(array=img_montage)


def videomontage(vidlist, imgheight, imgwidth, gridrows=None, gridcols=None, aspectratio=1, crop=False, skip=True, border=1, border_rgb=(128,128,128), do_flush=False, verbose=True, framerate=30.0, max_duration=None):
    """Generate a video montage for the provided videos by creating a image montage for every frame.  

    Args:
        `vipy.visualize.montage`:  See the args   
        framerate: [float] the framerate of the montage video.  All of the input videos are resampled to this common frame rate
        max_duration: [float] If not None, the maximum duration an element in seconds in the montage before it cycles.  If None, then use the duration of the longest video

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
        log.info('[vipy.visualize.videomontage]: Loading %d videos' % len(vidlist))

    vidlist = [v.framerate(framerate) for v in vidlist]  # resample to a common framerate, this must occur prior to load
    vidlist = [v.load() for v in vidlist]  # triggers load, make sure that the vidlist videos have a reasonably small frames
    max_length = max([len(v) for v in vidlist]) if max_duration is None else int(round(max_duration * framerate))   
    
    if verbose:
        log.info('[vipy.visualize.videomontage]: Maximum video length (frames) = %d' % (max_length))

    # FIXME: use stream here:
    # with Video(outfile).stream(write=True) as s:
    #     s.write(montage(...))
    
    montagelist = [montage([v[k % len(v)].mindim(max(imgheight, imgwidth)).centercrop(imgheight, imgwidth) for v in vidlist], imgheight, imgwidth, gridrows, gridcols, aspectratio, crop, skip, border, border_rgb, do_flush, verbose=False)
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
        url = Path(filename).as_uri()
        log.info('[vipy.visualize.urls]: Opening "%s" in default browser' % url)
        webbrowser.open(url)
        
    return filename

    
    
def tohtml(imlist, imdict=None, title='Image Visualization', mindim=1024, outfile=None, display=False, maxwidth=2400):
    """Given a list of vipy.image.Image objects, show the images along with the dictionary contents of imdict (one per image) in a single standalone HTML file
    
    Args:
        imlist: [list `vipy.image.Image`] 
        imdict: [list of dict] An optional list of dictionaries, such that each dictionary is visualized per image
        title: [str] The title of the html file
        imagewidth: [int] The size of the images in the page
        outfile: [str] The path to the output html file
        display: [bool] open the html file in the default system viewer when complete
        maxwidth [int]: the maximum width of the html container in pixels
    
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
    f.write('<div id="container" style="width:%dpx">\n' % maxwidth)
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
            for (j,v) in imdict[k].items():
                f.write('<b>%s</b>: %s<br>\n' % (html.escape(str(j)), html.escape(str(v))))
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
        url = Path(filename).as_uri()
        log.info('[vipy.visualize.tohtml]: Opening "%s" in default browser' % url)
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
        url = Path(filename).as_uri()
        log.info('[vipy.visualize.videolist]: Opening "%s" in default browser' % url)
        webbrowser.open(url)
        
    return filename

    
    
def imagelist(list_of_image_files, outdir, title='Image Visualization', imagewidth=64, maxwidth=2400):
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
    f.write('<div id="container" style="width:%dpx">\n' % maxwidth)
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
