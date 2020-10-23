# Adapted for vipy.downloader from original authors below
#
# Authors: Nicolas Pinto <pinto@rowland.harvard.edu>
#          Nicolas Poilvert <poilvert@rowland.harvard.edu>
# License: BSD 3 clause
#
# Copyright (c) 2010 Gary Wilson Jr. <gary.wilson@gmail.com> and contributers.
# License: MIT license
# http://pypi.python.org/pypi/python-archive/0.1
# http://code.google.com/p/python-archive/

# import socket
# socket.setdefaulttimeout(15)  # do not set globally?
import urllib.request
import urllib.error
import urllib.parse
import base64
from urllib.request import urlopen
from os import path
import hashlib
from vipy.util import isfile, try_import
import os
import tarfile
import zipfile
import sys
try:
    import bz2  # FIXME: Remove once bz2 is included in CentOS7 vendor baseline release?
except:
    pass
from vipy.util import isS3url, filetail
from vipy.globals import print


# FIX <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate
# verify failed (_ssl.c:581)>
# http://stackoverflow.com/questions/27835619/ssl-certificate-verify-failed-error
import ssl
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context


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
    return (md5 == hashlib.md5(data).hexdigest())

def generate_md5(filename):
    return hashlib.md5(open(filename, 'rb').read()).hexdigest()

def scp(url, output_filename, verbose=True):
    """Download using pre-installed SSH keys where hostname is formatted 'scp://hostname.com:/path/to/file.jpg' """        
    try_import('paramiko', 'paramiko scp')
    try_import('scp', 'paramiko scp')    
    import paramiko
    from scp import SCPClient
        
    assert 'scp://' in url, "Invalid URL"
    (hostname, remote_filename) = url.split('scp://')[1].split(':')

    if verbose:
        print("[vipy.downloader]: Downloading '%s' to '%s'" % (url, output_filename))
        
    def progress(filename, size, sent):
        sys.stdout.write("[vipy.downloader]: %s ... %.2f%%   \r" % (filename, float(sent)/float(size)*100) )
    
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname)
    scp = SCPClient(ssh.get_transport(), progress=progress if verbose else None)
    scp.get(remote_filename, output_filename)
    scp.close()
    return output_filename


def s3(url, output_filename, verbose=True):
    """Thin wrapper for boto3"""
    
    # https://aws.amazon.com/blogs/security/how-to-find-update-access-keys-password-mfa-aws-management-console/    
    assert 'VIPY_AWS_ACCESS_KEY_ID' in os.environ and 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ, \
        "AWS access keys not found - You need to create ENVIRONMENT variables ['VIPY_AWS_ACCESS_KEY_ID', 'VIPY_AWS_SECRET_ACCESS_KEY'] with S3 access credentials"   
    try_import('boto3', 'boto3')    
    assert isS3url(url), "Invalid URL - Must be 's3://BUCKETNAME.s3.amazonaws.com/OBJECTNAME.ext'"
    
    import boto3                        
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'],
                      aws_session_token=os.environ['VIPY_AWS_SESSION_TOKEN'] if 'VIPY_AWS_SESSION_TOKEN' in os.environ else None                      
    )
    
    # url format: s3://BUCKETNAME.s3.amazonaws.com/OBJECTNAME.mp4
    bucket_name = urllib.parse.urlparse(url).netloc.split('.')[0]
    object_name = urllib.parse.urlparse(url).path[1:]

    if verbose:
        print('[vipy.downloader.s3]: Downloading "%s" -> "%s"' % (url, output_filename))
    s3.download_file(bucket_name, object_name, output_filename)
    return output_filename


def s3_bucket(bucket_name, object_name, output_filename, verbose=True):
    """Thin wrapper for boto3"""
    # https://aws.amazon.com/blogs/security/how-to-find-update-access-keys-password-mfa-aws-management-console/
    assert 'VIPY_AWS_ACCESS_KEY_ID' in os.environ and 'VIPY_AWS_SECRET_ACCESS_KEY' in os.environ, \
        "AWS access keys not found - You need to create ENVIRONMENT variables ['VIPY_AWS_ACCESS_KEY_ID', 'VIPY_AWS_SECRET_ACCESS_KEY'] with S3 access credentials"   
    try_import('boto3', 'boto3')
    import boto3                        
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'],
                      aws_session_token=os.environ['VIPY_AWS_SESSION_TOKEN'] if 'VIPY_AWS_SESSION_TOKEN' in os.environ else None
    )    
    s3.download_file(bucket_name, object_name, output_filename)
    return output_filename


def download(url, output_filename, sha1=None, verbose=True, md5=None, timeout=None, username=None, password=None):
    """Downloads file at `url` and write it in `output_filename`"""
    if timeout is None:
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
        print("[vipy.downloader]: Downloading '%s' to '%s'" % (url, output_filename))

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
            if verbose:
                print(status, end=' ')  # space instead of newline
                sys.stdout.flush()
        if verbose:
            print('')
    else:
        while True:
            buffer = page.read(block_size)
            if not buffer:
                break
            dl_size += block_size / 1024
            output_file.write(buffer)
            # percent = min(100, 100. * dl_size / file_size)
            status = r"Progress: %20d kilobytes" \
                % (dl_size)
            status = status + chr(8) * (len(status) + 1)
            if verbose:
                print(status, end=' ')  # space instead of newline
                sys.stdout.flush()
        if verbose:
            print('')
        # output_file.write(page.read())

    output_file.close()

    if sha1 is not None:
        if not verify_sha1(output_filename, sha1):
            raise IOError('invalid sha1 for "%s"' % output_filename)

    if md5 is not None:
        if not verify_md5(output_filename, md5):
            raise IOError('invalid md5 for "%s"' % output_filename)


def unpack(archive_filename, output_dirname, sha1=None, verbose=True):
    """Extracts `archive_filename` in `output_dirname`.

    Supported archives:
    -------------------
    * Zip formats and equivalents: .zip, .egg, .jar
    * Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2
    * gzip compressed files
    * non-tar .bz2
    """
    if verbose:
        print("[vipy.downloader.extract]: Extracting '%s' to '%s'" % (archive_filename, output_dirname))
    if sha1 is not None:
        if verbose:
            print(" SHA-1 verification...")
        verify_sha1(archive_filename, sha1)
    try:
        extract(archive_filename, output_dirname, verbose=verbose)
    except UnrecognizedArchiveFormat:
        base, ext = os.path.splitext(archive_filename)
        # Handle non-tar bz2 archives
        if 'bz2' in globals() and ext == '.bz2':
            # Nothing to do if already extracted
            if os.path.exists(base):
                pass
            else:
                with open(base, 'wb') as outputfile, bz2.BZ2File(archive_filename, 'rb') as bz2_file:
                    data = bz2_file.read()
                    outputfile.write(data)
        else:
            raise


def download_and_unpack(url, output_dirname, sha1=None, verbose=True):
    """Downloads and extracts archive in `url` into `output_dirname`.

    Note that `output_dirname` has to exist and won't be created by this
    function.
    """
    archive_basename = path.basename(url)
    archive_filename = path.join(output_dirname, archive_basename)
    download(url, archive_filename, sha1=sha1, verbose=verbose)
    extract(archive_filename, output_dirname, verbose=verbose)


def download_unpack_and_cleanup(url, output_dirname, sha1=None, verbose=True):
    download_and_extract(url, output_dirname, sha1, verbose)
    os.remove(path.join(output_dirname, path.basename(url)))


def unpack_and_cleanup(archive_filename, output_dirname, sha1=None, verbose=True):
    download_and_extract(url, output_dirname, sha1, verbose)
    os.remove(path.join(output_dirname, path.basename(url)))


class ArchiveException(Exception):
    """Base exception class for all archive errors."""


class UnrecognizedArchiveFormat(ArchiveException):
    """Error raised when passed file is not a recognized archive format."""


def extract(archive_filename, output_dirname='./', verbose=True):
    """
    Unpack the tar or zip file at the specified `archive_filename` to the
    directory specified by `output_dirname`.
    """
    Archive(archive_filename).extract(output_dirname)


class Archive(object):
    """
    The external API class that encapsulates an archive implementation.
    """

    def __init__(self, file):
        self._archive = self._archive_cls(file)(file)

    @staticmethod
    def _archive_cls(file):
        cls = None
        if isinstance(file, str):
            filename = file
        else:
            try:
                filename = file.name
            except AttributeError:
                raise UnrecognizedArchiveFormat(
                    "File object not a recognized archive format.")
        base, tail_ext = os.path.splitext(filename.lower())
        cls = extension_map.get(tail_ext)
        if not cls:
            base, ext = os.path.splitext(base)
            cls = extension_map.get(ext)
        if not cls:
            raise UnrecognizedArchiveFormat(
                "Path not a recognized archive format: %s" % filename)
        return cls

    def extract(self, output_dirname=''):
        self._archive.extract(output_dirname)

    def list(self):
        self._archive.list()


class BaseArchive(object):
    """
    Base Archive class.  Implementations should inherit this class.
    """

    def extract(self):
        raise NotImplementedError

    def list(self):
        raise NotImplementedError


class ExtractInterface(object):
    """
    Interface class exposing common extract functionalities for
    standard-library-based Archive classes (e.g. based on modules like tarfile,
    zipfile).
    """

    def extract(self, output_dirname, verbose=True):
        if not verbose:
            self._archive.extractall(output_dirname)
        else:
            members = self.get_members()
            n_members = len(members)
            for mi, member in enumerate(members):
                self._archive.extract(member, path=output_dirname)
                extracted = mi + 1
                status = (r"Progress: %20i files extracted [%4.1f%%]"
                          % (extracted, extracted * 100. / n_members))
                status += chr(8) * (len(status) + 1)
                print(status, end=' ')
                sys.stdout.flush()
            print()


class TarArchive(ExtractInterface, BaseArchive):

    def __init__(self, filename):
        self._archive = tarfile.open(filename)

    def list(self, *args, **kwargs):
        self._archive.list(*args, **kwargs)

    def get_members(self):
        return self._archive.getmembers()


class ZipArchive(ExtractInterface, BaseArchive):

    def __init__(self, filename):
        self._archive = zipfile.ZipFile(filename)

    def list(self, *args, **kwargs):
        self._archive.printdir(*args, **kwargs)

    def get_members(self):
        return self._archive.namelist()


extension_map = {
    '.egg': ZipArchive,
    '.jar': ZipArchive,
    '.tar': TarArchive,
    '.tar.bz2': TarArchive,
    '.tar.gz': TarArchive,
    '.tgz': TarArchive,
    '.tz2': TarArchive,
    '.zip': ZipArchive,
}
