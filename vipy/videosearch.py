import sys
import os
import ssl
import urllib
import json
import re
import random
from vipy.useragent import common_user_agents, complete_user_agents
from vipy.util import tofilename, remkdir, filepath, filebase, isurl, try_import
import glob
from subprocess import DEVNULL, STDOUT
import subprocess


def isactiveyoutuber(username):
    """Does the youtube user have any uploaded videos?"""
    url = 'http://www.youtube.com/user/%s/videos' % username
    user_agent = random.choice(common_user_agents)
    headers = {'User-Agent':user_agent}
    search_request = urllib.request.Request(url, None, headers)
    try:
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
        search_results = urllib.request.urlopen(search_request, context=gcontext)
    except AttributeError:
        try:
            search_results = urllib.request.urlopen(search_request)
        except:
            return False
    except:
        return False
    s = search_results.read()
    return 'Uploads' in str(s)


def youtubeuser(tag, n_pages=1):
    """return all unique /user/* urls returned for a search for a given query tag"""
    try_import('bs4', 'beautifulsoup4'); from bs4 import BeautifulSoup
    url = 'http://www.youtube.com/results?search_query=%s&page=%d'
    userlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}
        search_request = urllib.request.Request(url % (tag.replace(' ','+'), k + 1), None, headers)
        try:
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
            search_results = urllib.request.urlopen(search_request, context=gcontext)
        except AttributeError:
            search_results = urllib.request.urlopen(search_request)
        links = BeautifulSoup(search_results.read(), features="html.parser").findAll("a")
        for link in links:
            if len(link['href']) > 6 and '/user/' == link['href'][0:6]:
                userlist.append(str('http://www.youtube.com%s' % link['href']))
    return list(set(userlist))


def is_downloadable_url(path):
    """Check to see if youtube-dl can download the path, this requires exeecuting 'youtube-dl $URL -q -j' to see if the returncode is non-zero"""
    ydl_exe = os.path.join(filepath(sys.executable), 'youtube-dl')
    if not os.path.exists(ydl_exe):
        raise ImportError('Optional package "youtube-dl" not installed -  Run "pip install youtube-dl"')
    retcode = subprocess.call([ydl_exe, path, '-q', '-j'], stdout=DEVNULL, stderr=STDOUT) if isurl(path) else -1
    return isurl(path) and retcode == 0


def youtube(tag, n_pages=1, channel=False, video_limit=None, expected_vid_list=None):
    """Return a list of YouTube URLs for the given tag and optional channel"""
    if channel:
        url = 'http://www.youtube.com/user/%s/videos'
        # url = 'http://www.youtube.com/user/%s/videos?sort=dd&view=0&flow=list&live_view=500'
    else:
        url = 'http://www.youtube.com/results?search_query=%s&page=%d'
    vidlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}
        if channel:
            search_request = urllib.request.Request(url % (tag.replace(' ','+')), None, headers)
        else:
            search_request = urllib.request.Request(url % (tag.replace(' ','+'), k + 1), None, headers)
        try:
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
            search_results = urllib.request.urlopen(search_request, context=gcontext)
        except AttributeError:
            search_results = urllib.request.urlopen(search_request)
        except:
            print('[vipy.videosearch.youtube]: URL 404: %s' % (url % (tag.replace(' ','+'))))
            url = 'http://www.youtube.com/channel/%s/videos'
            search_request = urllib.request.Request(url % (tag.replace(' ','+')), None, headers)
            try:
                search_results = urllib.request.urlopen(search_request, context=gcontext)
            except AttributeError:
                search_results = urllib.request.urlopen(search_request)
            except:
                print('[vipy.videosearch.youtube]: URL 404: %s' % (url % (tag.replace(' ','+'))))
                return ([None], [None])
        search_data = str(search_results.read())

        datalist = search_data.split('href="/watch?')

        vidlist.extend(['https://www.youtube.com/watch?%s' % vid.split('"')[0] for vid in datalist if 'DOCTYPE' not in vid.split('"')[0]])
        if expected_vid_list is not None:
            new_list = [v for v in expected_vid_list if v in vidlist]
            if len(new_list) == len(expected_vid_list):
                print('YOUTUBE: PREVIOUS USER CHECK WAS GOOD: %s' % tag)
                return new_list
            else:
                return []
    vidlist = [v for v in set(vidlist) if isurl(v)]   # unique valid URLs
    return(vidlist)


def liveleak(tag, n_pages=1):
    url = 'http://www.liveleak.com/browse?q=%s&page=%d'
    vidlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}

        search_request = urllib.request.Request(url % (tag.replace(' ','+'), k + 1), None, headers)
        search_results = urllib.request.urlopen(search_request)
        search_data = str(search_results.read())

        datalist = search_data.split('href="https://www.liveleak.com/view?')
        vidlist.extend(['https://www.liveleak.com/view?%s' % vid.split('"')[0].split('&')[0] for vid in datalist if 'DOCTYPE' not in vid.split('"')[0]])
    vidlist = list(set(vidlist))  # unique
    return(vidlist)


def download(vidurl, vidfile, skip=False, writeurlfile=True, max_filesize='350m', remove_parts=True, verbose=False):
    """Use youtube-dl to download a video URL to a video file"""

    user_agent = random.choice(complete_user_agents)    
    ydl_exe = os.path.join(filepath(sys.executable), 'youtube-dl')
    if not os.path.exists(ydl_exe):
        raise ImportError('Optional package "youtube-dl" not installed -  Run "pip install youtube-dl"')
    try:
        print('[vipy.videosearch.download]: saving "%s" to "%s"' % (vidurl, vidfile))
        for f in glob.glob("%s*" % vidfile):
            os.remove(f)  # youtube-dl will not overwrite, so we force it
        cmd = '%s %s "%s" -o %s --max-filesize %s --no-check-certificate --user-agent="%s"' % (ydl_exe, '-q' if not verbose else '', vidurl, vidfile, max_filesize, user_agent)  # must be on path            
        if verbose:
            print('[vipy.videosearch.download]: executing \'%s\'' % cmd)
        erno = os.system(cmd)
        if erno != 0:
            raise ValueError('youtube-dl returned %d' % erno)
        if os.path.isfile(vidfile):
            if writeurlfile:
                urlfile = os.path.join(filepath(vidfile), '%s.url' % filebase(vidfile))
                with open(urlfile, 'w') as f:
                    f.write(vidurl + '\n')
        if remove_parts and os.path.isfile(vidfile + '.part'):
            partfile = vidfile + '.part'
            print('[vipy.youtube.download]: removing partial file: %s' % partfile)
            os.remove(partfile)
    except KeyboardInterrupt:
        raise
    except Exception as exception:
        print(exception)
        # http://rg3.github.io/youtube-dl/supportedsites.html
        print('[vipy.videosearch.download]: download failed - skipping')
        return None

    if erno == 256:
        raise KeyboardInterrupt
    elif skip and erno:  # use this if every video on a page fails
        return None
    else:
        return vidfile


def bulkdownload(vidurls, outpattern, skip=True, writeurlfile=True, max_filesize='350m', remove_parts=True):
    """Use youtube-dl to download a list of video URLs to video files using the provided sprintf outpattern=/path/to/out_%d.mp4 where the index is provided by the URL list index"""

    vidfiles = []
    for (k, vidurl) in enumerate(vidurls):
        vidfile = download(vidurl, outpattern % k, skip, writeurlfile, max_filesize, remove_parts)
        if vidfile is not None:
            vidfiles.append(vidfile)
    return vidfiles
