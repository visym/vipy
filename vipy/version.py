MAJOR = 1
MINOR = 10
RELEASE = 26
VERSION = '%d.%d.%d' % (MAJOR, MINOR, RELEASE)


def num(versionstring=VERSION):
    (major, minor, release) = versionstring.split('.')    
    return 100*100*int(major) + 100*int(minor) + int(release)


def at_least_version(versionstring):
    """Is versionstring='X.Y.Z' at least the current version?"""
    return num(VERSION) >= num(versionstring)


def is_at_least(versionstring):
    """Synonym for at_least_version"""
    return num(VERSION) >= num(versionstring)    


def is_exactly(versionstring):
    return versionstring == VERSION


def at_least_major_version(major):
    return MAJOR >= int(major)


