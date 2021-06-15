MAJOR = 1
MINOR = 11
RELEASE = 2
VERSION = '%d.%d.%d' % (MAJOR, MINOR, RELEASE)


def num(versionstring=VERSION):
    """Convert the version string of the form 'X.Y.Z' to an integer 100000*X + 100*Y + Z for version comparison"""
    (major, minor, release) = versionstring.split('.')    
    return 100*100*int(major) + 100*int(minor) + int(release)


def at_least_version(versionstring):    
    """Is versionstring='X.Y.Z' at least the current version?"""
    return num(VERSION) >= num(versionstring)


def is_at_least(versionstring):
    """Synonym for at_least_version"""
    return num(VERSION) >= num(versionstring)    


def is_exactly(versionstring):
    """Is the versionstring = 'X,Y.Z' exactly equal to `vipy.version.VERSION`"""
    return versionstring == VERSION


def at_least_major_version(major):
    """is the major version (e.g. X, for version X.Y.Z) greater than or equal to the major version integer supplied?"""
    return MAJOR >= int(major)


