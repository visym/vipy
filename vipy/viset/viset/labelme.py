raise ValueError('FIXME: update me to the new viset format')


import tables
import os
from os import path
from viset.dataset import CategorizationDetectionViset
import httplib2
from BeautifulSoup import BeautifulSoup, SoupStrainer # EXTERNAL
import urlparse
import string
import viset.util
from viset.util import quietprint
import xml.etree.ElementTree as ET

class LabelMe():
    def scrape(self, baseurl, keeptest):
        http = httplib2.Http()
        (status, response) = http.request(baseurl)
        filelist = []
        for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
            if (hasattr(link, 'href')) and (link.text not in ['Name','Last modified','Size','Description','Parent Directory']):
                dirurl = urlparse.urljoin(baseurl, link['href']) # relative link
                #        print 'Directory: ' + dirurl
                (status, response) = http.request(dirurl)
                for filelink in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
                    if (hasattr(filelink, 'href')) and (filelink.text not in ['Name','Last modified','Size','Description','Parent Directory']):
                        fileurl = urlparse.urljoin(dirurl, filelink['href'])
                        if keeptest(fileurl): 
                            print '[viset.labelme]: ' + fileurl              
                            filelist.append(fileurl) 
        return filelist

    def parse(xmlfile):
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        # http://docs.python.org/2/library/xml.etree.elementtree.html#module-xml.etree.ElementTree
        pass
    
    def export(self, verbose=False):    
        # Create empty database
        db = CategorizationDetectionViset(self._dbname, mode='w', verbose=verbose)

        # Fetch data necessary to initial construction
        print('[viset.labelme]: scraping imagery from ' + self.IMURL)
        imlist = self.scrape(self.IMURL, viset.util.isimg)
        print('[viset.labelme]: scraping annotations from ' + self.ANNOURL)        
        annolist = self.scrape(self.ANNOURL, viset.util.isxml)    

        # Create me (image list only for now)
        imstream = db.image
        for imurl in imlist:
            imstream.write(imurl)

        # Cleanup
        db.close()
        return db.abspath()

class LabelMe3(LabelMe):
    IMURL = 'http://people.csail.mit.edu/brussell/research/LabelMe/Images/'
    ANNOURL = 'http://people.csail.mit.edu/brussell/research/LabelMe/Annotations/'
    SHA1 = None
    verbose = True
    _dbname = 'labelme3'



  ## Many annotations in polygon XML format for each image
  ## store as a finite list of empty columns for max 100 points

  ## Each annotation is scraped from xml document
  ## images are defined by URL from scraping entry

  ## add a DetectionPolygonTable 

  ## http://stackoverflow.com/questions/1080411/retrieve-links-from-web-page-using-python-and-beautiful-soup


  ## matplotlib supports polygons
  ## http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.fill

  ## https://github.com/mpitid/pylabelme
