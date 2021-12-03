import urllib
import os
import re
import random
import math
from vipy.util import try_import, isurl, remkdir
import tempfile
import numpy as np
from vipy.downloader import common_user_agents
from vipy.globals import print
import warnings


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


