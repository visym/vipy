"""Helpers to download and extract archives"""

# Adapted for viset from original authors below
# 
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          Nicolas Poilvert <poilvert@rowland.harvard.edu>
# License: BSD 3 clause


#import socket
#socket.setdefaulttimeout(15)  # do not set globally?
import urllib.request, urllib.error, urllib.parse
import base64
from urllib.request import urlopen
from os import path
import hashlib
import sys
from vipy.util import isfile
from . import archive
import os
# FIXME: Remove once bz2 is included in CentOS7 vendor baseline release
try:
    import bz2
except:
    pass


def generate_sha1(filepath):
    sha1 = hashlib.sha1()
    f = open(filepath, 'rb')
    try:
        sha1.update(f.read())
    finally:
        f.close()
    return sha1.hexdigest()

def verify_sha1(filename, sha1):
    data = open(filename, 'rb').read()
    return (sha1 == hashlib.sha1(data).hexdigest())

def verify_md5(filename, md5):
    data = open(filename, 'rb').read()
    if md5 != hashlib.md5(data).hexdigest():
        raise IOError("File '%s': invalid md5 hash! You may want to delete "
                      "this corrupted file..." % filename)


def download(url, output_filename, sha1=None, verbose=True, md5=None, timeout=None, username=None, password=None):
    """Downloads file at `url` and write it in `output_dirname`"""
    if timeout == None:
        timeout = 10

    if username is None and password is None:
        page = urlopen(url, None, timeout=timeout)  # urllib for username:password options
    else:
        request = urllib.request.Request(url)
        base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string)   
        page = urllib.request.urlopen(request, timeout=timeout)        
        
    page_info = page.info()

    output_file = open(output_filename, 'wb+')  # will raise IOError exception on invalid permissions

    # size of the download unit
    block_size = 2 ** 15
    dl_size = 0

    if verbose:
        print("[bobo.cache]: Downloading '%s' to '%s'" % (url, output_filename))
        
    # display  progress only if we know the length
    if 'content-length' in page_info and verbose:
        # file size in Kilobytes
        file_size = int(page_info['content-length']) / 1024.
        while True:
            buffer = page.read(block_size)
            if not buffer:
                break
            dl_size += block_size / 1024
            output_file.write(buffer)
            percent = min(100, 100. * dl_size / file_size)
            status = r"Progress: %20d kilobytes [%4.1f%%]" \
                    % (dl_size, percent)
            status = status + chr(8) * (len(status) + 1) 
            print(status, end=' ') # space instead of newline
            sys.stdout.flush()
        print('')
    else:
        while True:
            buffer = page.read(block_size)
            if not buffer:
                break
            dl_size += block_size / 1024
            output_file.write(buffer)
            #percent = min(100, 100. * dl_size / file_size)
            status = r"Progress: %20d kilobytes" \
                    % (dl_size)
            status = status + chr(8) * (len(status) + 1) 
            print(status, end=' ') # space instead of newline
            sys.stdout.flush()
        print('')
        #output_file.write(page.read())

    output_file.close()

    if sha1 is not None:
        if not verify_sha1(output_filename, sha1):
            raise IOError('invalid sha1')

    if md5 is not None:
        verify_md5(output_filename, md5)


def extract(archive_filename, output_dirname, sha1=None, verbose=True):
    """Extracts `archive_filename` in `output_dirname`.

    Supported archives:
    -------------------
    * Zip formats and equivalents: .zip, .egg, .jar
    * Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2
    * gzip compressed files
    * non-tar .bz2
    """
    if verbose:
        print("[bobo.cache]: Extracting '%s' to '%s'" % (archive_filename, output_dirname))
    if sha1 is not None:
        if verbose:
            print(" SHA-1 verification...")
        verify_sha1(archive_filename, sha1)
    try:
        archive.extract(archive_filename, output_dirname, verbose=verbose)
    except archive.UnrecognizedArchiveFormat:
        base, ext = os.path.splitext(archive_filename)
        # Handle non-tar bz2 archives
        if 'bz2' in globals() and ext == '.bz2':
            # Nothing to do if already extracted
            if os.path.exists(base):
                pass
            else:
                with open(base, 'wb') as outputfile, bz2.BZ2File(archive_filename, 'rb') as bz2_file:
                    data=bz2_file.read()
                    outputfile.write(data)
        else:
            raise



def download_and_extract(url, output_dirname, sha1=None, verbose=True):
    """Downloads and extracts archive in `url` into `output_dirname`.

    Note that `output_dirname` has to exist and won't be created by this
    function.
    """
    archive_basename = path.basename(url)
    archive_filename = path.join(output_dirname, archive_basename)
    download(url, archive_filename, sha1=sha1, verbose=verbose)
    extract(archive_filename, output_dirname, sha1=sha1, verbose=verbose)

def download_extract_cleanup(url, output_dirname, sha1=None, verbose=True):
    download_and_extract(url, output_dirname, sha1, verbose)
    os.remove(path.join(output_dirname, path.basename(url)))

def extract_and_cleanup(archive_filename, output_dirname, sha1=None, verbose=True):
    download_and_extract(url, output_dirname, sha1, verbose)
    os.remove(path.join(output_dirname, path.basename(url)))

def cache_and_extract(url, output_dirname, sha1=None, verbose=True):
    if not isfile(path.join(output_dirname, path.basename(url))):
        download_and_extract(url, output_dirname, sha1, verbose)
