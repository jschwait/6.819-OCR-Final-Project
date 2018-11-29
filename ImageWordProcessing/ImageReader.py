import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


"""
Image object which splits the picture into separate lines, characters,
and words.  The image of the text to be split up MUST BE FULL SCREEN
WHEN TRYING TO BE SPLIT UP
"""
class ImageReader:

	def __init__(self, image_path):
		"""
		Creates ImageReader object to manipulate an image with

		@param image_path: path of image to manipulate
		@type image_path: string
		"""
		self.image = mpimg.imread(image_path)


	def show_image(self, picture_to_display):
		"""
		Displays an image to the use

		@param pic: a read in image
		@type: pic: numpy.array

		@return: none, displays an image

		"""
		plt.imshow(picture_to_display)
		plt.show()

	def get_character_positions_from_single_line(self):
		"""
		Obtains the pixel positions of the start and end of each
		character in the inputted image

		@return list of ints, with each element i, i+1 representing
		the starting and ending pixel of a character in the image
		where i%2 == 0

		"""
		height, width = self.image.shape[:2]
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
			# 1.0 represents an all white image, which is just
			# a space, or background
			if np.average(current_column) < .98:
				characters.append(y)
		letters = [characters[0]]
		for i in range(len(characters)-1):
			# If the pixels are not consecutive, they are
			# different letters
			if characters[i] != characters[i+1] - 1:
				letters.append(characters[i])
				letters.append(characters[i+1])
				#Get the beginning and end pixel of each letter
		letters.append(characters[-1])
		return letters

	def get_lines_from_paragraph(self):
		"""
		Separates lines of a paragraph

		@return: list of numpy.array, each element of which
			represents lines of the inputted paragraph

		"""
		height, width = self.image.shape[:2]
		lines = []
		for y in range(height):
			current_column = [0, 0, 0, 0]
			for i in range(width):
				# Average the entire row to see if 
				# there is are any words there
				current_column[0] += self.image[y, i][0]
				current_column[1] += self.image[y, i][1]
				current_column[2] += self.image[y, i][2]
				current_column[3] += self.image[y, i][3]
			for i in range(4):
				current_column[i] /= width
			# 1.0 represents an all white image, which is just
			# a space, or background
			if np.average(current_column) != 1.0:
				lines.append(y)
		line_indices = [lines[0]]
		for i in range(len(lines)-1):
			# If the pixels are not consecutive, they are
			# different lines
			if lines[i] != lines[i+1] - 1:
				line_indices.append(lines[i])
				line_indices.append(lines[i+1])
				# Get the beginning and end pixel of each line
		line_indices.append(lines[-1])

		lines_subslices = []
		for i in range(0, len(line_indices), 2):
			# Get subslice of image for each line of text
			# +/- 10 is arbitrary for now to get the lines
			# need to fix later to make general
			# TODO
			lines_subslices.append(self.image[line_indices[i]-10:line_indices[i+1]+10, :])

		count = 0
		# Save each new line as its own image so it can be split
		# further into words and characters
		for i in range(len(lines_subslices)):
			mpimg.imsave("image" + str(count) + ".png", lines_subslices[i])
			count += 1

		count = 0
		# Make each line its own ImageReader to be processed further
		for i in range(len(lines_subslices)):
			lines_subslices[i] = ImageReader("image" + str(count) + ".png")
			count += 1

		return lines_subslices

	def characters_positions(self, letters_spots):
		"""
		Get the slice of the image of each character
		on the line

		@param letters_spots: the start and end of each letter 
			in the image
		@type letters_spots: list of ints

		@return: list of numpy.array, each element of which represents
			characters (and spaces) on a line

		"""
		character_images_subslices = []
		letters = []
		dist = 0
		for i in range(0, len(letters_spots)-2, 2):
			dist += letters_spots[i+1] - letters_spots[i]
		dist /= len(letters_spots)
		for i in range(0, len(letters_spots)-2, 2):
			letters.append(letters_spots[i])
			letters.append(letters_spots[i+1])
			# Each two indices in letters_spots is beginning
			# and end of individual letters/characters
			# print(letters_spots[i+1], letters_spots[i+2])
			if letters_spots[i+1] + dist < letters_spots[i+2]:
				# If there is more than the average number of
				# pixels between each letter, then there is 
				# a space between the letters and I must append
				# a space between the letters to my letters array
				letters.append(letters_spots[i+1] + int(dist))
				letters.append(letters_spots[i+1] + int(dist)+1)
				# Add spaces to the letters list to dilineate words
		letters.append(letters_spots[-2])
		letters.append(letters_spots[-1])
		for i in range(0, len(letters), 2):
			# Get subslice of image for each character
			character_images_subslices.append(self.image[:, letters[i]:letters[i+1]])
		return character_images_subslices

	def get_words_from_characters(self, characters):
		"""
		Arrange spliced characters into words

		@param characters: list of spliced images, each element
		represents a single character
		@type characters: numpy.array

		@return list of numpy.array, each element of which represents
			a word on the read in line

		"""
		spaces = []
		for char in range(len(characters)):
			whitespace = True
			# Split words based off whitespacing
			height, width = characters[char].shape[:2]
			for y in range(width):
				current_column = [0, 0, 0, 0]
				for i in range(height):
					# Average the entire column to see if 
					# any of the column is not white to 
					# determine if there is a character there
					current_column[0] += characters[char][i, y][0]
					current_column[1] += characters[char][i, y][1]
					current_column[2] += characters[char][i, y][2]
					current_column[3] += characters[char][i, y][3]
				for i in range(4):
					current_column[i] /= height
				if np.average(current_column) != 1.0:
					# Enter this block if not a whitespace
					whitespace = False
					break
			else:
				# Enter this if character is a whitespace
				# which we will save location of for further use
				spaces.append(char)


		lines_words = []
		single_word = []
		for char in range(len(characters)):
			if char in spaces:
				# If we encounter a space, the
				# characters we were just iterating
				# over is a complete word
				lines_words.append(single_word)
				single_word = []
			else:
				# Otherwise we are iterating over
				# the same word
				single_word.append(characters[char])
		else:
			# Add the last word on the line since there
			# is no whitespace at the end of the line
			lines_words.append(single_word)

		stitched_words = []
		for word in lines_words:
			# Stitch each of the individual words together
			stitched_words.append(self.stitch_characters(word))

		return stitched_words

	def concatonate_images(self, im1, im2):
		"""
		Stitches two images together into one side by side image

		@param im1: the left side of the desired stitched image
		@type im1: numpy.array
		@param im2: the right side of the desired stitched image
		@type im2: numpy.array

		@return a new numpy.array of the two inputted images
			stitched together images side by side

		"""
		height1, width1 = im1.shape[:2]
		height2, width2 = im2.shape[:2]

		# Total width is the combination
		# of the two individual images
		total_width = width1 + width2

		# Create new empty image of correct height and width
		new_image = np.zeros(shape=(height1, total_width, 4))
		# First part of image is image 1
		new_image[:height1, :width1] = im1
		# Second part of image is image 2
		new_image[:height2, width1:width1+width2] = im2

		return new_image	

	def stitch_characters(self, characters):
		"""
		Combines characters together to get the word

		@param characters: list of images of characters
		@type characters: list of numpy.array

		@return a completely stitched together image of
			all the elements of characters side by side
		"""
		output = None
		for i, img in enumerate(characters):
			if i == 0:
				# First element is the base case
				output = img
			else:
				# Otherwise stitch the rest of the characters
				# to the right of the preceding character
				output = self.concatonate_images(output, img)
		return output




# Size 48 font
# testImage = ImageReader('cupcakesSentence.png')
# Size 48 font
# testImage = ImageReader('VeryNiceSentence.png')
# Size 11 font
# testImage = ImageReader('niceDay.png')
# Size 48 font
# testImage = ImageReader('paragraph.png')
# Size 11 font paragraph
testImage = ImageReader('LongParagraph.png')


lines = testImage.get_lines_from_paragraph()

print('1. Lines')
# Shows each line
# for line in lines:
# 	line.show_image(line.image)

letters_positions = []
character_images = []
for line in range(len(lines)):
	letters_positions.append(lines[line].get_character_positions_from_single_line())
	# Get each character on the line
	character_images.append(lines[line].characters_positions(letters_positions[line]))

# Shows each character on each line
# print('2. Characters')
# for line in range(len(lines)):
# 	for char in character_images[line]:
# 		lines[line].show_image(char)


lines_of_words = []
for line in range(len(lines)):
	# Get each word from the parsed paragraph
	lines_of_words.append(lines[line].get_words_from_characters(character_images[line]))

print('3. Words')
# On each line
for line in lines_of_words:
	# Print each word on the line
	for word in line:
		lines[0].show_image(word)
