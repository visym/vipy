"""Annotation tools for downloading videos from youtube"""

from strpy.bobo.annotation import common_user_agents
import os
import urllib2, ssl
import json
import re
import random
from strpy.bobo.util import tofilename, remkdir, filepath, filebase, isurl

def dailymotion(tag):
    pass
    
def vine(user_id, outdir=None, video_limit=None):
    # url = 'https://vine.co/u/%s/' % user_id
    url = 'https://api.vineapp.com/timelines/users/%s' % user_id

    user_agent = random.choice(common_user_agents)
    headers = {'User-Agent':user_agent}
    search_request = urllib2.Request(url, None, headers)
    try:
        search_results = urllib2.urlopen(search_request)
    except:
        print '[bobo.videosearch.vine]: Error on URL: %s' % (url)
        return ([None], [None])
    search_data = search_results.read()
    data = json.loads(search_data)
    try:
        # https://github.com/starlock/vino/wiki/API-Reference#response-6
        vidlist = [rec['videoUrl'] for rec in data['data']['records']]
    except KeyError, what:
        print '[bobo.videosearch.vine]: Key error on parse: %s' % (what)
        return ([None], [None])

    vidlist = list(set(vidlist))  # unique
    if outdir is not None:
        remkdir(outdir)
        filelist = download(vidlist, os.path.join(outdir, 'vine_%s_%%04d.mp4' % user_id), video_limit=video_limit)
        return vidlist, filelist
    return(vidlist)

def isactiveyoutuber(username):
    """Does the youtube user have any uploaded videos?"""
    url = 'http://www.youtube.com/user/%s/videos' % username
    user_agent = random.choice(common_user_agents)
    headers = {'User-Agent':user_agent}
    search_request = urllib2.Request(url, None, headers)
    try:
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1) # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
        search_results = urllib2.urlopen(search_request, context=gcontext)
    except AttributeError:
        try:
            search_results = urllib2.urlopen(search_request)
        except:
            return False
    except:
        return False
    s = search_results.read()
    return 'Uploads' in str(s)
        
    
def youtubeuser(tag, n_pages=1):
    """return all unique /user/* urls returned for a search for a given query tag"""
    from BeautifulSoup import BeautifulSoup # venv
    url = 'http://www.youtube.com/results?search_query=%s&page=%d'
    userlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}
        search_request = urllib2.Request(url % (tag.replace(' ','+'), k+1), None, headers)
        try:
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1) # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
            search_results = urllib2.urlopen(search_request, context=gcontext)
        except AttributeError:
            search_results = urllib2.urlopen(search_request)
        links = BeautifulSoup(search_results.read()).findAll("a")
        for link in links:
            if len(link['href']) > 6 and '/user/' == link['href'][0:6]:
                userlist.append(str('http://www.youtube.com%s' % link['href']))
    return list(set(userlist))
    
        
def youtube(tag, n_pages=1, outdir=None, channel=False, video_limit=None, expected_vid_list=None):
    if channel:
        url = 'http://www.youtube.com/user/%s/videos'
        #url = 'http://www.youtube.com/user/%s/videos?sort=dd&view=0&flow=list&live_view=500'
    else:
        url = 'http://www.youtube.com/results?search_query=%s&page=%d'
    vidlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}
        if channel:
            search_request = urllib2.Request(url % (tag.replace(' ','+')), None, headers)
        else:
            search_request = urllib2.Request(url % (tag.replace(' ','+'), k+1), None, headers)
        try:
            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1) # to avoid [SSL: CERTIFICATE_VERIFY_FAILED] exception
            search_results = urllib2.urlopen(search_request, context=gcontext)
        except AttributeError:
            search_results = urllib2.urlopen(search_request)
        except:
            print '[bobo.videosearch.youtube]: URL 404: %s' % (url % (tag.replace(' ','+')))
            url = 'http://www.youtube.com/channel/%s/videos'
            search_request = urllib2.Request(url % (tag.replace(' ','+')), None, headers)
            try:
                search_results = urllib2.urlopen(search_request, context=gcontext)
            except AttributeError:
                search_results = urllib2.urlopen(search_request)
            except:
                print '[bobo.videosearch.youtube]: URL 404: %s' % (url % (tag.replace(' ','+')))
                return ([None], [None])
        search_data = search_results.read()
        
        datalist = search_data.split('href="/watch?')

        vidlist.extend(['https://www.youtube.com/watch?%s' % vid.split('"')[0] for vid in datalist if 'DOCTYPE' not in vid.split('"')[0]])
        if expected_vid_list is not None:
            new_list = [v for v in expected_vid_list if v in vidlist]
            if len(new_list) == len(expected_vid_list):
                print 'YOUTUBE: PREVIOUS USER CHECK WAS GOOD: %s' % tag
                return new_list
            else:
                return []
    vidlist = list(set(vidlist))  # unique
    if outdir is not None:
        filelist = download(vidlist, os.path.join(remkdir(outdir), 'youtube_'+tofilename(tag, hyphen=False)+'_%04d.mp4'), video_limit=video_limit)
        #FIXME: Add this to others
        return vidlist, filelist
    return(vidlist)

def liveleak(tag, n_pages=1, outdir=None):
    url = 'http://www.liveleak.com/browse?q=%s&page=%d'
    vidlist = []
    for k in range(0, n_pages):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}

        search_request = urllib2.Request(url % (tag.replace(' ','+'), k+1), None, headers)
        search_results = urllib2.urlopen(search_request)
        search_data = search_results.read()

        datalist = search_data.split('href="http://www.liveleak.com/view?')
        vidlist.extend(['http://www.liveleak.com/view?%s' % vid.split('"')[0].split('&')[0] for vid in datalist if 'DOCTYPE' not in vid.split('"')[0]])
    vidlist = list(set(vidlist))  # unique

    if outdir is not None:
        download(vidlist, os.path.join(remkdir(outdir), 'liveleak_'+tofilename(tag)+'_%04d.mp4'))
    return(vidlist)


def ustream(tag, n_pages=1, outdir=None):
    url = 'http://www.ustream.tv/search?q=%s'
    vidlist = []
    for k in range(0, 1):
        user_agent = random.choice(common_user_agents)
        headers = {'User-Agent':user_agent}

        search_request = urllib2.Request(url % (tag.replace(' ','+')), None, headers)
        search_results = urllib2.urlopen(search_request)
        search_data = search_results.read()

        datalist = search_data.split('href="/recorded/')
        vidlist.extend(['http://www.ustream.tv/recorded/%s' % vid.split('"')[0] for vid in datalist if 'DOCTYPE' not in vid.split('"')[0]])
    vidlist = list(set(vidlist))  # unique

    print vidlist
    if outdir is not None:
        download(vidlist, os.path.join(remkdir(outdir), 'ustream_'+tofilename(tag)+'_%04d.mp4'))
    return(vidlist)


def download(vidlist, outfile, skip=False, writeurlfiles=True, max_filesize='350m', video_limit=None, remove_parts=True):
    """Use youtube-dl to download videos from url"""
    vidfiles = []
    successes = 0
    for (k,v) in enumerate(vidlist):
        try:
            try:
                vidfile = outfile % k
            except TypeError:
                # Allow caller to explicitly name file
                vidfile = outfile
            print '[bobo.youtube.download]: exporting "%s" to "%s"' % (v, vidfile)
            erno = os.system('youtube-dl "%s" -o %s --max-filesize %s --no-check-certificate' % (v, vidfile, max_filesize))  # must be on path
            print 'youtube-dl returned %d' % erno
            if os.path.isfile(vidfile):
                vidfiles.append(vidfile)
                successes += 1
                if writeurlfiles:
                    urlfile = os.path.join(filepath(vidfile), '%s.url' % filebase(vidfile))
                    with open(urlfile, 'w') as f:
                        f.write(v + '\n')
            elif remove_parts and os.path.isfile(vidfile + '.part'):
                partfile = vidfile + '.part'
                print '[bobo.youtube.download]: removing partial file: %s' % partfile
                os.remove(partfile)
            if video_limit is not None and successes == video_limit:
                break
        except Exception as exception:
            print exception
            # http://rg3.github.io/youtube-dl/supportedsites.html
            print '[bobo.youtube.download]: download failed - skipping'
        else:
            if skip and erno: #use this if every video on a page fails
                break
    return vidfiles
