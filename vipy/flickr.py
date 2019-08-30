
import flickrapi
import flickrapi.shorturl
import urllib
import os
import urllib2
import re
import random
import math

flickr_api_key = os.environ.get('FLICKR_API_KEY')
#flickr = flickrapi.FlickrAPI(flickrapikey, flickrapisecret)

def search(searchtag):
    flickr = flickrapi.FlickrAPI(flickr_api_key)
    photos = flickr.walk(text=searchtag,per_page='500')  
    return (photos, flickr)

def download(tag='owl'):
    (photos, flickr) = search(tag)
    for img in photos:        
        id = img.get('id')
        info = flickr.photos_getInfo(photo_id=id)    
        url = 'http://farm'+img.get('farm')+'.staticflickr.com/'+img.get('server')+'/'+img.get('id')+'_'+img.get('secret')+'_n.jpg'
        imfile = "/tmp/"+tag.replace(',','_')+"_"+id+".jpg"
        urllib.urlretrieve(url, imfile)
