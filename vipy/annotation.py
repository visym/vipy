import urllib
import os
import re
import random
import math
from vipy.util import try_import, isurl, remkdir
import tempfile
import numpy as np
from vipy.useragent import common_user_agents


def googlesearch(tag):
    """Return a list of image URLs from google image search associated with the provided tag"""
    url = 'https://www.google.com/search?tbm=isch&q=%s' % tag.replace(' ','+')
    user_agent = random.choice(common_user_agents)
    headers = {'User-Agent':user_agent}
    search_request = urllib.request.Request(url,None,headers)
    search_results = urllib.request.urlopen(search_request)
    search_data = str(search_results.read())

    # FIXME: support for gstatic.com URLs
    datalist = search_data.split('http')
    imlist = [re.findall("^http[s]?://.*\.(?:jpg|gif|png)", str('http' + d)) for d in datalist]
    imlist = [im[0] for im in imlist if len(im) > 0]
    imlist_clean = [im for im in imlist if im.find('File:') == -1]
    return [url for url in imlist_clean if isurl(url) and 'gb/images/silhouette' not in url]


def basic_level_categories():
    """Return a list of nouns from wordnet that can be used as an initial list of basic level object categories"""
    try_import('nltk'); import nltk
    nltkdir = remkdir(os.path.join(os.environ['VIPY_CACHE'], 'nltk')) if 'VIPY_CACHE' in os.environ else tempfile.gettempdir()
    os.environ['NLTK_DATA'] = nltkdir
    print('[vipy.annotation.basic_level_categories]: Downloading wordnet to "%s"' % tempfile.gettempdir())
    nltk.download('wordnet', tempfile.gettempdir())

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
    print('[vipy.annotation.verbs]: Downloading verbnet to "%s"' % tempfile.gettempdir())
    nltk.download('verbnet', tempfile.gettempdir())
    from nltk.corpus import verbnet
    return verbnet.lemmas()


def facebookprofilerange(fbid, numid, outdir='./imgs', cleanup=True, hierarchical=False, redownload=False):
    for x in range(fbid, fbid + numid):
        facebookprofile(x, outdir, cleanup, hierarchical, redownload)


def facebookprofile(fbid, outdir='./imgs', cleanup=True, hierarchical=False, redownload=False):
    if hierarchical:
        subdir = remkdir(os.path.join(outdir, str(int(float(fbid) / 1E4))))  # 10000 images per directory
        outfile = os.path.join(subdir, '%d.jpg' % int(fbid))  # outdir/1000/10000001.jpg
    else:
        outfile = os.path.join(outdir, '%d.jpg' % int(fbid))

    url = "http://graph.facebook.com/picture?id=" + str(fbid) + "&width=800"
    if not os.path.exists(outfile) or redownload:
        try:
            print('[facebookprofile.download]: Downloading "%s" to "%s"' % (url, outfile))

            user_agent = np.random.choice(common_user_agents)
            headers = {'User-Agent':user_agent}
            req = urllib.request.Request(url, None, headers)
            imgfile = urllib.request.urlopen(req)
            total_size = int(imgfile.info().getheader('Content-Length').strip())
            downloaded = 0
            CHUNK = 256 * 10240
            with open(outfile, 'wb') as fp:
                while True:
                    chunk = imgfile.read(CHUNK)
                    downloaded += len(chunk)
                    # print math.floor( (downloaded / total_size) * 100 )
                    if not chunk:
                        break
                    fp.write(chunk)

            # urllib.urlretrieve(url, outfile)

            s = os.path.getsize(outfile)
            if cleanup and (s < 11000 or s == 10626 or s == 10491):
                print('[facebookprofile.download]: deleting invalid file "%s"' % outfile)
                os.remove(outfile)

        except (urllib.request.HTTPError):
            print('[fb_image.download]: Skipping "%s"' % (url))
        except (urllib.request.URLError):
            print('[fb_image.download]: Skipping "%s"' % (url))
        except KeyboardInterrupt:
            raise
        except:
            raise

