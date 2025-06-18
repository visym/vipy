import sys
import os
import ssl
import urllib
import json
import re
import random
from vipy.globals import log
from vipy.downloader import common_user_agents, complete_user_agents
from vipy.util import tofilename, remkdir, filepath, filebase, isurl, try_import
import glob
from subprocess import DEVNULL, STDOUT
import subprocess
import shutil


youtube_dl_exe = shutil.which('yt-dlp')        
has_youtube_dl = youtube_dl_exe is not None and os.path.exists(youtube_dl_exe)


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
    raise ValueErro('this scraping no longer works')

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
    """Check to see if yt-dlp can download the path, this requires exeecuting 'yt-dlp $URL -q -j' to see if the returncode is non-zero"""
    if not has_youtube_dl:
        raise ImportError('Optional package "yt-dlp" not installed -  Run "pip install yt-dlp"')    
    retcode = subprocess.call([youtube_dl_exe, '-q', '-j'], stdout=DEVNULL, stderr=STDOUT) if isurl(path) else -1
    return isurl(path) and retcode == 0


def youtube(tag, n_pages=1, channel=False, video_limit=None):
    """Return a list of YouTube URLs for the given tag and optional channel"""
    if channel:
        url_template = 'http://www.youtube.com/user/%s/videos'
        # url_template = 'http://www.youtube.com/user/%s/videos?sort=dd&view=0&flow=list&live_view=500'
    else:
        url_template = 'http://www.youtube.com/results?search_query=%s&page=%d'
    vidlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}
        if channel:
            url = url_template % (tag.replace(' ','+'))
            search_request = urllib.request.Request(url, None, headers)
        else:
            url = url_template % (tag.replace(' ','+'), k + 1)
            search_request = urllib.request.Request(url, None, headers)
        try:
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
            search_results = urllib.request.urlopen(search_request, context=gcontext)
        except AttributeError:
            search_results = urllib.request.urlopen(search_request)
        except:
            log.info('[vipy.videosearch.youtube]: URL 404: %s' % (url))
            url_template = 'http://www.youtube.com/channel/%s/videos'
            url = url_template% (tag.replace(' ','+'))
            search_request = urllib.request.Request(url, None, headers)
            try:
                search_results = urllib.request.urlopen(search_request, context=gcontext)
            except AttributeError:
                search_results = urllib.request.urlopen(search_request)
            except:
                log.info('[vipy.videosearch.youtube]: URL 404: %s' % (url))
                return ([None], [None])
        search_data = str(search_results.read())

        datalist = search_data.split('{"webCommandMetadata":{"url":"')

        vidlist.extend(['https://www.youtube.com%s' % vid.split('"')[0] for vid in datalist if 'watch?' in vid])

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


def download(vidurl, vidfile, skip=False, writeurlfile=True, max_filesize='9999m', remove_parts=True, verbose=False):
    """Use yt-dlp to download a video URL to a video file"""

    user_agent = random.choice(complete_user_agents)    
    if not has_youtube_dl:
        raise ImportError('Optional package "yt-dlp" not installed -  Run "pip install yt-dlp"')
    try:
        log.info('saving "%s" to "%s"' % (vidurl, vidfile))
        for f in glob.glob("%s*" % vidfile):
            os.remove(f)  # yt-dlp will not overwrite, so we force it
        cmd = '%s %s "%s" -o "%s" --no-check-certificate --max-filesize="%s" --user-agent="%s"' % (youtube_dl_exe, '-q' if not verbose else '', vidurl, vidfile, max_filesize, user_agent)  # must be on path            
        if verbose:
            log.info('executing \'%s\'' % cmd)
        erno = os.system(cmd)
        if erno != 0:
            raise ValueError('yt-dlp returned %d' % erno)
        if os.path.isfile(vidfile):
            if writeurlfile:
                urlfile = os.path.join(filepath(vidfile), '%s.url' % filebase(vidfile))
                with open(urlfile, 'w') as f:
                    f.write(vidurl + '\n')
        if remove_parts and os.path.isfile(vidfile + '.part'):
            partfile = vidfile + '.part'
            log.info('removing partial file: %s' % partfile)
            os.remove(partfile)
    except KeyboardInterrupt:
        raise
    except Exception as exception:
        log.warning(exception)
        log.warning('download failed')
        return None

    if erno == 2:
        raise KeyboardInterrupt
    elif skip and erno:  # use this if every video on a page fails
        return None
    else:
        return vidfile


def bulkdownload(vidurls, outpattern, skip=True, writeurlfile=True, max_filesize='350m', remove_parts=True):
    """Use yt-dlp to download a list of video URLs to video files using the provided sprintf outpattern=/path/to/out_%d.mp4 where the index is provided by the URL list index"""

    vidfiles = []
    for (k, vidurl) in enumerate(vidurls):
        vidfile = download(vidurl, outpattern % k, skip, writeurlfile, max_filesize, remove_parts)
        if vidfile is not None:
            vidfiles.append(vidfile)
    return vidfiles
