import os
import sys


def test_import():
    import vipy
    import vipy.activity
    import vipy.annotation
    import vipy.calibration
    import vipy.downloader
    import vipy.flow
    import vipy.geometry    
    import vipy.image
    import vipy.linalg
    import vipy.math
    import vipy.metrics
    import vipy.object
    import vipy.show
    import vipy.util
    import vipy.video
    import vipy.visualize
    import vipy.gui.using_matplotlib
    import vipy.data
    print('[test_import]: PASSED')


if __name__ == "__main__":
    test_import()
