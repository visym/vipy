
class TestFailed(Exception):
   """Raised when unit test fails"""
   pass


if __name__ == "__main__":
   import test_import   
   test_import.run()

   import test_geometry   
   test_geometry.geometry()
   test_geometry.boundingbox()
   test_geometry.ellipse()

   import test_image   
   test_image.image()
   test_image.imagedetection()   
   test_image.scene()    

   import test_video
   test_video.datasets()
   test_video.scene()
   test_video.video()
