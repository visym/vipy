MAJOR = 1
MINOR = 16
RELEASE = 1

class Version():
    def __init__(self, major, minor, release=0):
        self.major = int(major)
        self.minor = int(minor)
        self.release = int(release)

    def __repr__(self):
        return self.version()

    def __str__(self):
        return self.version()

    def __eq__(self, other):        
        return str(self) == str(Version.from_string(other))

    def __gt__(self, other):
        return self.int()>Version.from_string(other).int()
    
    def __ge__(self, other):
        return self.int()>=Version.from_string(other).int()        

    def __lt__(self, other):
        return self.int()<Version.from_string(other).int()                
    
    def __le__(self, other):
        return self.int()<=Version.from_string(other).int()                        
        
    @classmethod
    def is_valid(cls, versionstr):
        try:
            Version.from_string(versionstr)
            return True
        except:
            return False
        
    @classmethod
    def from_string(cls, version):
        """version string can be 'PACKAGE-X.Y.Z' or 'X.Y.Z' or 'X.Y'"""
        assert isinstance(version, str), "invalid version string"
        assert version.count('.') in [1,2], "invalid version string"
        assert version.rsplit('-',1)[-1].count('.') in [1,2], "invalid version string"                
        
        return cls(*version.rsplit('-',1)[-1].split('.'))

    def int(self):
        return self.major*10000 + self.minor*100 + self.release
    
    def version(self):
        return '%d.%d.%d' % (self.major, self.minor, self.release)        

    def at_least_version(self, v):
        """Is versionstring='X.Y.Z' at least the current version?"""        
        return self.at_least_major_version(v) and self.at_least_minor_version(v) and self.at_least_release_version(v)

    def at_least_major_version(self, v):
        return self.major >= Version.from_string(v).major
    
    def same_major_version(self, v):
        return self.major == Version.from_string(v).major
    
    def at_least_minor_version(self, v):
        return self.minor  >= Version.from_string(v).minor
    
    def at_least_release_version(self, v):
        return self.release >= Version.from_string(v).release

    
VERSION = Version(MAJOR, MINOR, RELEASE)

