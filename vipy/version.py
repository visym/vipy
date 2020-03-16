MAJOR = 0
MINOR = 7
RELEASE = 4
VERSION = '%d.%d.%d' % (MAJOR, MINOR, RELEASE)


def at_least_version(versionstring):
    """Is versionstring='X.Y.Z' at least the current version?"""
    (major, minor, release) = versionnum.split('.')
    return at_least_major_version(major) and at_least_minor_version(minor) and at_least_release_version(release)


def at_least_major_version(major):
    return int(major) >= MAJOR


def at_least_minor_version(minor):
    return int(minor) >= MINOR


def at_least_release_version(release):
    return int(release) >= RELEASE

