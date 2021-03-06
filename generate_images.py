# File from former 6.819 project (2017)
# Their project found at this link: https://github.com/maryam-a/ocr-system
# Author: Maryam Archie

import os
import string
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path

# Set seed
np.random.seed(123)

# Constant Values
DATA_ROOT = 'data/'
# Single Char
CHAR_DIR = DATA_ROOT + 'single_char'
CHAR_DATA = DATA_ROOT + 'char_data.txt'
# Single Line
SL_DIR = DATA_ROOT + 'single_line'
SL_DATA = DATA_ROOT + 'sl_data.txt'
# Multiple Line
ML_DIR = DATA_ROOT + 'multiple_lines'
ML_DATA = DATA_ROOT + 'ml_data.txt'
# Demo Data
DEMO_DIR = DATA_ROOT + 'demo'

TEXT_INFO = {
    '/Library/Fonts/Arial.ttf': {'size': 20, 'dir': '/Arial'},
    '/Library/Fonts/Times New Roman.ttf': {'size': 20, 'dir': '/TimesNewRoman'},
    '/Library/Fonts/Calibri.ttf': {'size': 20, 'dir': '/Calibri'}
}

TEXT_FONT = '/Library/Fonts/Calibri.ttf' # Change this
TEXT_SIZE = TEXT_INFO[TEXT_FONT]['size']
CURRENT_CHAR_DIR = CHAR_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_SL_DIR = SL_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_ML_DIR = ML_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_DEMO_DIR = DEMO_DIR + TEXT_INFO[TEXT_FONT]['dir']

IMAGE_FONT = ImageFont.truetype(TEXT_FONT, TEXT_SIZE)
IMAGE_MODE = 'RGB'
BACKGROUND_COLOR = 'white'
TEXT_COLOR = (0, 0, 0)
TEXT_OFFSET = (0, -1)
MIN_CHAR = 10
MAX_CHAR = 64
N_IMAGES = 10000
N_LINES = 10
POSSIBLE_CHARS = list(string.ascii_letters) + list(string.digits) + [' ']

IMAGE_ENCODING = '.png'

def create_text_image(text, filename, size, font=IMAGE_FONT, offset=TEXT_OFFSET,
                      background=BACKGROUND_COLOR, font_color=TEXT_COLOR):
    '''
    Creates an image of text.
    text:       string, The desired text to show in the image
    filename:   string, The name for the image text file
    size:       width, height), The intended size of the image
    font:       FreeTypeFont, The font used to style the text
    offset:     (x, y), The text offset in the image; (0,0) is the top left corner
    background: string, The color of the image itself
    font_color: (r, g, b), The color of the text
    '''
    base_image = Image.new(IMAGE_MODE, size, background)
    drawn_image = ImageDraw.Draw(base_image)
    drawn_image.text(offset, text, font_color, font)
    base_image.save(filename)

def get_ideal_text_image_size(max_chars=MAX_CHAR, num_lines=1, font=IMAGE_FONT,
                              background=BACKGROUND_COLOR):
    '''
    Determines the dimensions of what an image of text should be given the max number of characters
    for a line of text.
    max_chars:      int, The max number of characters for a line of text
    num_lines:   int, The number of lines of text in the image
    font:           FreeTypeFont, The font used to style the text
    background:     string, The color of the image itself
    '''
    # start with image block and expand with text
    base_image = Image.new(IMAGE_MODE, (1, 1), background)
    drawn_image = ImageDraw.Draw(base_image)

    if num_lines != 1:
        # re: text is double spaced
        return drawn_image.textsize('\n'.join(2 * num_lines * ['0' * max_chars]), font=font)
    else:
        return drawn_image.textsize('\n'.join(['0' * max_chars]), font=font)

def create_dir_if_not_present(dir_name):
    '''
    Ensures that the directory is present. Creates it if necessary.
    dir_name:   The name of the directory of interest
    '''
    if not Path(dir_name).is_dir():
        Path(dir_name).mkdir()

def create_single_char_images(num_images=N_IMAGES, save_dir=CURRENT_CHAR_DIR, font=IMAGE_FONT,
                              background=BACKGROUND_COLOR, offset=TEXT_OFFSET,
                              font_color=TEXT_COLOR):
    create_dir_if_not_present(save_dir)
    all_lines = []
    for i in range(num_images):
        random_string = ''.join(np.random.choice(POSSIBLE_CHARS, 1))
        image_name = str(i) + IMAGE_ENCODING
        create_text_image(random_string, save_dir + '/' + image_name, (20, 20), font, offset, background, font_color)
        all_lines.append(image_name + ' ' + random_string) #note space: need to change that in deepocr

    with open(CHAR_DATA, 'w') as output_file:
        output_file.write('\n'.join(all_lines))

def create_single_line_text_images(min_chars=MIN_CHAR, max_chars=MAX_CHAR, num_images=N_IMAGES,
                                  save_dir=CURRENT_SL_DIR, font=IMAGE_FONT, background=BACKGROUND_COLOR,
                                  offset=TEXT_OFFSET, font_color=TEXT_COLOR):
    '''
    Create an image with a single line of random text (lowercase + uppercase ASCII letters,
    punctuation, single whitespace)
    min_chars:  int, The minimum number of characters for the text
    max_chars:  int, The maximum number of characters for the text
    num_images: int, The number of images to create
    save_dir:   string, The directory to save the images to
    font:           FreeTypeFont, The font used to style the text
    background:     string, The color of the image itself
    offset:     (x, y), The text offset in the image; (0,0) is the top left corner
    '''
    create_dir_if_not_present(save_dir)
    size = get_ideal_text_image_size(max_chars, 1, font, background)
    print("This should be (640, 14): " + str(size))

    all_lines = []
    for i in range(num_images):
        random_num = np.random.randint(min_chars, max_chars)
        random_string = ''.join(np.random.choice(POSSIBLE_CHARS, random_num))
        image_name = str(i) + IMAGE_ENCODING
        create_text_image(random_string, save_dir + '/' + image_name, size, font, offset, background, font_color)
        all_lines.append(image_name + ' ' + random_string) #note space: need to change that in deepocr
    
    with open(SL_DATA, 'w') as output_file:
        output_file.write('\n'.join(all_lines))

def create_multiple_line_text_images(num_lines=N_LINES, min_chars=MIN_CHAR, max_chars=MAX_CHAR, num_images=N_IMAGES,
                                    save_dir=CURRENT_ML_DIR, font=IMAGE_FONT, background=BACKGROUND_COLOR,
                                    offset=TEXT_OFFSET, font_color=TEXT_COLOR):
    '''
    Create an image with a single line of random text (lowercase + uppercase ASCII letters,
    punctuation, single whitespace)
    min_chars:  int, The minimum number of characters for the text
    max_chars:  int, The maximum number of characters for the text
    num_images: int, The number of images to create
    save_dir:   string, The directory to save the images to
    font:       FreeTypeFont, The font used to style the text
    background: string, The color of the image itself
    offset:     (x, y), The text offset in the image; (0,0) is the top left corner
    '''
    create_dir_if_not_present(save_dir)
    size = get_ideal_text_image_size(max_chars, num_lines, font, background)

    all_lines = []
    for i in range(num_images):
        temp = []
        rand_num = np.random.randint(1, num_lines+1)

        for _ in range(rand_num):
            random_num = np.random.randint(min_chars, max_chars)
            random_string = ''.join(np.random.choice(POSSIBLE_CHARS, random_num))
            temp.append(random_string)

        image_text = '\n\n'.join(temp)
        file_text = '\\n'.join(temp)
        image_name = str(i) + IMAGE_ENCODING
        create_text_image(image_text, save_dir + '/' + image_name, size, font, offset, background, font_color)
        all_lines.append(image_name + ' ' + file_text) #note space: need to change that in deepocr
    
    with open(ML_DATA, 'w') as output_file:
        output_file.write('\n'.join(all_lines))

def create_demo_data(textfile, max_chars=MAX_CHAR,
                    save_dir=CURRENT_DEMO_DIR, font=IMAGE_FONT, background=BACKGROUND_COLOR,
                    offset=TEXT_OFFSET, font_color=TEXT_COLOR):
    create_dir_if_not_present(save_dir)

    all_lines = []
    with open(textfile) as demo_text:
        for line in demo_text:
            print(line.strip())
            all_lines.append(line.strip())

    size = get_ideal_text_image_size(max_chars, len(all_lines), font, background)
    
    image_text = '\n\n'.join(all_lines)
    image_name = 'demo' + IMAGE_ENCODING
    create_text_image(image_text, save_dir + '/' + image_name, size, font, offset, background, font_color)

if __name__ == "__main__":
    create_dir_if_not_present(DATA_ROOT)
    create_dir_if_not_present(CHAR_DIR)
    create_dir_if_not_present(SL_DIR)
    create_dir_if_not_present(ML_DIR)
    create_single_char_images()
    # create_single_line_text_images()
    # create_multiple_line_text_images(num_images=100)
    # create_demo_data(DATA_ROOT + 'demo.txt')