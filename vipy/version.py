MAJOR = 0
MINOR = 7
RELEASE = 7
VERSION = '%d.%d.%d' % (MAJOR, MINOR, RELEASE)


def at_least_version(versionstring):
    """Is versionstring='X.Y.Z' at least the current version?"""
    (major, minor, release) = versionstring.split('.')
    return at_least_major_version(major) and at_least_minor_version(minor) and at_least_release_version(release)


def at_least_major_version(major):
    return MAJOR >= int(major)


def at_least_minor_version(minor):
    return MINOR >= int(minor)


def at_least_release_version(release):
    return RELEASE >= int(release)
