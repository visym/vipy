import urllib
import os
import re
import random
import math
from vipy.util import try_import, isurl, remkdir, temphtml, tempjson, filetail, readjson, isjsonfile, toextension
import tempfile
import numpy as np
from vipy.downloader import common_user_agents
from vipy.globals import print
import warnings
from datetime import datetime
import time
import webbrowser
import html
            

def googlesearch(tag):
    """Return a list of image URLs from google image search associated with the provided tag"""
    raise ValueError('no longer unsupported')
    
    try_import('bs4', 'BeautifulSoup4');  
    from bs4 import BeautifulSoup
    
    url = 'https://www.google.com/search?tbm=isch&q=%s' % tag.replace(' ','+')
    user_agent = random.choice(common_user_agents)
    headers = {'User-Agent':user_agent}
    search_request = urllib.request.Request(url,None,headers)
    search_results = urllib.request.urlopen(search_request)
    search_data = str(search_results.read())

    soup = BeautifulSoup(search_data, features="html.parser")
    links = soup.find_all('a')    
    urls = [tag.get('href',None) for tag in links if tag.get('href',None) is not None]
    urls = [u.replace('/url?q=http','http') if u.startswith('/url?q=http') else u for u in urls]
    urls = [u for u in urls if u.startswith('http') and not u.startswith('https://www.google.com')]
    return urls


def basic_level_categories():
    """Return a list of nouns from wordnet that can be used as an initial list of basic level object categories"""
    try_import('nltk'); import nltk
    nltkdir = remkdir(os.path.join(os.environ['VIPY_CACHE'], 'nltk')) if 'VIPY_CACHE' in os.environ else tempfile.gettempdir()
    os.environ['NLTK_DATA'] = nltkdir
    print('[vipy.annotation.basic_level_categories]: Downloading wordnet to "%s"' % nltkdir)
    nltk.download('wordnet', nltkdir)
    nltk.data.path.append(nltkdir)
    
    from nltk.corpus import wordnet
    nouns = []
    allowed_lexnames = ['noun.animal', 'noun.artifact', 'noun.body', 'noun.food', 'noun.object', 'noun.plant']
    for synset in list(wordnet.all_synsets('n')):
        if synset.lexname() in allowed_lexnames:
            nouns.append(str(synset.lemmas()[0].name()).lower())
    nouns.sort()
    return nouns


def verbs():
    """Return a list of verbs from verbnet that can be used to define a set of activities"""
    try_import('nltk'); import nltk
    nltkdir = remkdir(os.path.join(os.environ['VIPY_CACHE'], 'nltk')) if 'VIPY_CACHE' in os.environ else tempfile.gettempdir()
    os.environ['NLTK_DATA'] = nltkdir
    print('[vipy.annotation.verbs]: Downloading verbnet to "%s"' % nltkdir)    
    nltk.download('verbnet', nltkdir)
    nltk.data.path.append(nltkdir)
    from nltk.corpus import verbnet
    return verbnet.lemmas()


class Selector():
    """HTML+javascript based UI for selecting images from a group.  

    * Given a list of lists of `vipy.image.Image` objects, display each image in a standalone HTML file.  
    * Allow the user to click on desirable images to select them.  
    * When done, the user selects the Download JSON button to export the selected elements.
    * The selected elements are stored in a JSON file which is exported back into this object to return only the selected images

    ```python
    imlist = [[vipy.image.RandomScene(128,128) for k in range(10)] for j in range(5)]
    s = vipy.annotation.Selector(imlist)
    s.html(show=True)  # annotate locally in browser by selecting images to keep, then download the resulting jsonfile
    selected_imlist = s.fromjson('/path/to/downloaded.json')  # returns only those images selected in the UI
    ```
    """
       
    def __init__(self, imlist):
        self._imlist = imlist

    def __iter__(self):
        for im in self._imlist:
            yield im
            
    def html(self, jsonfile=None, htmlfile=None, imdict=None, title='Image Selector', maxwidth=4096, default_opacity=0.5, selected_opacity=1.0, show=False, description="Select the images to keep from each group, download the JSON, then call vipy.annotation.Select.fromjson()", mindim=None):
        """Given a list of tuples of `vipy.image.Image` objects, create a standalone HTML file that will allow the user to select individual images in the group.  Selected images are output to jsonfile"""
        htmlfile = temphtml() if htmlfile is None else htmlfile
        jsonfile = filetail(toextension(htmlfile, '.json')) if jsonfile is None else jsonfile
        imlist = self._imlist
        
        f = open(htmlfile,'w')
        f.write('<!DOCTYPE html>\n')
        f.write('<!--\n    Visym Labs\n    vipy.annotation.Selector (https://visym.github.io/vipy)\n    Generated: %s\n-->\n' % str(datetime.now()))            
        f.write("<head>\n");
        f.write('  <meta charset="UTF-8">\n')
        f.write("  <title>%s</title>\n" % title)
        f.write('  <link href="https://use.fontawesome.com/releases/v5.13.0/css/all.css" rel="stylesheet">\n')
        f.write("</head>\n")        
        f.write('<html>\n')
        f.write('<body>\n')
        f.write('  <script type="text/javascript">\n')
        f.write('    var selected = {};\n');
        f.write('    function set_unselected(x) { x.style.opacity=%s; x.setAttribute("selected","off"); selected[x.id]=false;};\n' % (default_opacity))
        f.write('    function set_selected(x) { x.style.opacity=%s; x.setAttribute("selected","on"); selected[x.id]=true;};\n' % (selected_opacity))
        f.write('    function toggle(x) { if (x.getAttribute("selected") == "off") { set_selected(x); } else { set_unselected(x); }; }\n')
        f.write('    function row_selected(k, n) { for (let i = 0; i < n; i++) { set_selected( document.getElementById(`(${k},${i})`));};}\n')
        f.write('    function row_unselected(k, n) { for (let i = 0; i < n; i++) { set_unselected( document.getElementById(`(${k},${i})`));};}\n')
        f.write('    function toggle_row(x,k,n) { if (x.getAttribute("selected") == "off") { row_selected(k,n); x.setAttribute("selected","on"); } else { row_unselected(k,n); x.setAttribute("selected","off");}; }\n')        
        f.write('    function exportJSON(x) { var data = "text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(selected)); x.setAttribute("href", "data:"+data); x.setAttribute("download", "%s"); }\n' % jsonfile)
        f.write('    function set_transparent(x) { x.style.opacity=0.5; };\n')
        f.write('    function set_opaque(x) { x.style.opacity=1.0; };\n')        
        f.write('  </script>\n')    
        f.write('<div id="container" style="width:%dpx">\n' % maxwidth)
        f.write('<div id="header">\n')
        f.write('<h1 style="margin-bottom:0;">%s</h1><br>\n' % title)
        localtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        f.write('HTML generated on %s<br>\n\n' % localtime)
        f.write('Number of image groups: %d<br>\n' % len(imlist))
        f.write('Number of images: %d<br><br>\n' % len([im for iml in imlist for im in iml]))
        f.write('%s<br><br>\n' % description)
        f.write('<a onclick="exportJSON(this);" class="btn"><button>Download JSON</button></a>\n') 
        f.write('</div>\n')
        f.write('<br>\n')
        f.write('<hr>\n')
        f.write('<div style="float:left;">\n')
    
        # Generate embedded images and html
        for (i, imsrclist) in enumerate(imlist):
            f.write('<p>\n</p>\n')
            if imdict is not None:
                for (k,v) in imdict[i].items():
                    f.write('<b>%s</b>: %s<br>\n' % (html.escape(str(k)), html.escape(str(v))))
                f.write('<p>\n</p>\n')
                
            for (j, im) in enumerate(imsrclist):                    
                im = im if mindim is None else im.clone().mindim(mindim)
                f.write(im.html(id='(%d,%d)' % (i,j), attributes={'loading':'lazy', 'onclick':"toggle(this)", 'selected':'off', 'style':"opacity:%f" % default_opacity}))   # base-64 encoded image with img tag
            f.write('<br><i title="Toggle Select All" onclick="toggle_row(this,%d,%d);" selected="on" onmouseover="set_opaque(this)" onmouseout="set_transparent(this)" class="fas fa-recycle fa-3x" id="row-%d" style="opacity:0.5;margin-right: 10px;"></i>' % (i,len(imsrclist),i))
            
            f.write('<p>\n</p>\n')
            f.write('<br>\n')
            f.write('<hr>\n')
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')
        f.close()

        if show:
            webbrowser.open('file:///%s' % htmlfile)  # chrome only, will not work in safari
        return htmlfile
    
    def fromjson(self, jsonfile):
        """Given the JSON file downloaded from the HTML selector, return only those that were selected"""
        assert isjsonfile(jsonfile)
        d = readjson(jsonfile)
        return [[im for (j,im) in enumerate(iml) if '(%d,%d)'%(i,j) in d and d['(%d,%d)'%(i,j)] == True] for (i,iml) in enumerate(self._imlist)]

    
class Filter(Selector):
    """HTML+javascript based UI for filtering images from a group.  

    * Given a list of lists of `vipy.image.Image` objects, display each image in a standalone HTML file.  
    * Allow the user to click on non-desirable images to remove them.  
    * When done, the user selects the Download JSON button to export the filtered elements.
    * The filtered elements are stored in a JSON file which is exported back into this object to return all but the filtered images

    ```python
    imlist = [[vipy.image.RandomScene(128,128) for k in range(10)] for j in range(5)]
    s = vipy.annotation.Filter(imlist)
    s.html(show=True)  # annotate locally in browser by selecting images to remove and download the resulting jsonfile
    filtered_imlist = s.fromjson('/path/to/downloaded.json')   # returns images with those selected in the UI removed
    ```
    """
    
    def html(self, jsonfile=None, htmlfile=None, imdict=None, title='Image Filter', maxwidth=4096, default_opacity=1.0, selected_opacity=0.5, show=False, description="Select the images to remove from each group, download the JSON, then call vipy.annotation.Filter.fromjson()", mindim=None):
        return super().html(jsonfile=jsonfile, htmlfile=htmlfile, title=title, default_opacity=default_opacity, selected_opacity=selected_opacity, show=show, description=description, mindim=mindim, imdict=imdict, maxwidth=maxwidth)

    def fromjson(self, jsonfile):    
        """Given the JSON file downloader from the HTML selector, return a filter of only those that were *not* selected"""
        assert isjsonfile(jsonfile)
        d = readjson(jsonfile)
        return [[im for (j,im) in enumerate(iml) if '(%d,%d)'%(i,j) not in d or d['(%d,%d)'%(i,j)] == False] for (i,iml) in enumerate(self._imlist)]
            
    
