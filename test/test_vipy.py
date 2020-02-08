import test_geometry
import test_image
import test_video

class TestFailed(Exception):
   """Raised when unit test fails"""
   pass

if __name__ == "__main__":
    test_geometry.geometry()
    test_geometry.boundingbox()
    test_geometry.ellipse()
    
    test_image.image()
    test_image.scene()    
    
    test_video.datasets()
    test_video.scene()
    test_video.video()
