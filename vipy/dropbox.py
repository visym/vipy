import os
from os import path
import urllib.parse as urlparse
from vipy.util import try_import


class Dropbox():

    _access_token = os.environ.get('VIPY_DROPBOX_ACCESS_TOKEN')
    _app_key = os.environ.get('VIPY_DROPBOX_APP_KEY')
    _app_secret = os.environ.get('VIPY_DROPBOX_APP_SECRET_KEY')

    def __init__(self):
        raise ValueError('FIXME: this uses an older version of dropbox')

        try_import('dropbox')
        import dropbox  # optional

        if self._access_token is None:
            self.link()

    def link(self):
        flow = dropboxapi.client.DropboxOAuth2FlowNoRedirect(self._app_key, self._app_secret)

        authorize_url = flow.start()
        print('1. Go to: ' + authorize_url)
        print('2. Click "Allow" (you might have to log in first)')
        print('3. Copy the authorization code.')
        code = raw_input("Enter the authorization code here: ").strip()

        # This will fail if the user enters an invalid authorization code
        access_token, user_id = flow.finish(code)
        self._access_token = access_token
        return access_token

    def put(self, filename, folder=None, verbose=True):
        client = dropbox.client.DropboxClient(self._access_token)
        if verbose:
            print('linked account: ', client.account_info())

        if folder is None:
            dropbox_path = '/' + path.basename(filename)
        else:
            dropbox_path = '/' + str(folder) + '/' + path.basename(filename)

        f = open(filename)
        response = client.put_file(dropbox_path, f)
        if verbose:
            print('uploaded: ', response)

        folder_metadata = client.metadata('/')
        if verbose:
            print('metadata: ', folder_metadata)

        share = client.share(dropbox_path, short_url=False)

        # https://www.dropbox.com/help/201/en
        p = urlparse.urlparse(share['url'])
        public_url = urlparse.urlunsplit(('http','dl.dropboxusercontent.com',p[2],None,None))
        # public_url = public_url + '?dl=1'
        return public_url

    def get(self, filename):
        client = dropbox.client.DropboxClient(self._access_token)
        f, metadata = client.get_file_and_metadata('/' + filename)
        cachefile = filename
        out = open(cachefile, 'w')
        out.write(f.read())
        out.close()
        print(metadata)
        return cachefile
