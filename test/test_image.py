from vipy.image import ImageDetection, Image


im = Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').show(figure=1)

im = ImageDetection(filename='./test/jebyrne.jpg', xmin=100, ymin=100, width=700, height=1000, category='face')
im.show(figure=1)
im.crop().show()


