import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class ImageReader:
	image = ''


	def __init__(self, image_name):
		self.image = mpimg.imread(image_name)


	def show_image(self):
		plt.imshow(self.image)
		plt.show()

	def show_character(self, pic):
		plt.imshow(pic)
		plt.show()

	def get_character_positions_from_single_line(self):
		height, width, color = self.image.shape
		characters = []
		for y in range(width):
			current_column = [0, 0, 0, 0]
			for i in range(height):
				# Average the entire column to see if 
				# there is a character there
				current_column[0] += self.image[i, y][0]
				current_column[1] += self.image[i, y][1]
				current_column[2] += self.image[i, y][2]
				current_column[3] += self.image[i, y][3]
			for i in range(4):
				current_column[i] /= height
			if np.average(current_column) != 1.0:
				characters.append(y)
		letters = [characters[0]]
		for i in range(len(characters)-1):
			if characters[i] != characters[i+1] - 1:
				letters.append(characters[i])
				letters.append(characters[i+1])
		letters.append(characters[-1])
		return letters

	def characters_positions(self, letters_spots):
		character_images_subslices = []
		letters = []
		for i in range(0, len(letters_spots)-2, 2):
			letters.append(letters_spots[i])
			letters.append(letters_spots[i+1])
			if letters_spots[i+1] + 35 < letters_spots[i+2]:
				letters.append(letters_spots[i+1]+ 10)
				letters.append(letters_spots[i+1] + 20)
		letters.append(letters_spots[-2])
		letters.append(letters_spots[-1])
		otherpic = []
		for i in range(0, len(letters), 2):
			character_images_subslices.append(self.image[:, letters[i]:letters[i+1]])
		return character_images_subslices

	def get_words_from_characters(self, characters):
		spaces = []
		for char in characters:
			whitespace = True
			height, width, color = char.shape
			for y in range(width):
				current_column = [0, 0, 0, 0]
				for i in range(height):
					# Average the entire column to see if 
					# there is a character there
					current_column[0] += char[i, y][0]
					current_column[1] += char[i, y][1]
					current_column[2] += char[i, y][2]
					current_column[3] += char[i, y][3]
				for i in range(4):
					current_column[i] /= height
				# if count == 0:
				# 	print(current_column)
					# self.show_character(char)
					# self.show_character(char)
				if np.average(current_column) != 1.0:# and char.all() not in spaces:
					whitespace = False
					break
			else:
				# print(whitespace)
				# print(current_column)
				print(height, width)
				self.show_character(char[1:3, 64:66])
				for i in range(width):
					for y in range(height):
						print(char[2:4, 65:67])

				# if whitespace:
				# 	spaces.append(char)
			break
		return spaces

		# word = []
		# for char in characters:
		# 	print(char)
		# 	print(spaces)
		# 	if spaces[0] != char:
		# 		print(char)
		# 		end = char[:]








testImage = ImageReader('cupcakesSentence.png')
letters_positions = testImage.get_character_positions_from_single_line()
character_images = testImage.characters_positions(letters_positions)
spaces = testImage.get_words_from_characters(character_images)
# for i in spaces:
# 	testImage.show_character(i)