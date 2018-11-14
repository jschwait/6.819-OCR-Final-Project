import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageReader:
	image = ''

	def __init__(self, image_name):
		self.image = mpimg.imread(image_name)


	def show_image(self):
		plt.imshow(self.image)
		plt.show()






testImage = ImageReader('cupcakesSentence.png')
testImage.show_image()