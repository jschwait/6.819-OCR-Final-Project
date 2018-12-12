import numpy as np
import argparse
import os
import getopt
import string
import sys
from skimage.io import imread
from sklearn.model_selection import ShuffleSplit
from sklearn import model_selection
from TFANN import ANNC
import tensorflow as tf
from scipy.stats import mode as Mode
import matplotlib.pyplot as plt
from image_reader import ImageReader


np.random.seed(123)

# Constant Values
DATA_ROOT = 'data/'
CHAR_DIR = DATA_ROOT + 'single_char/'
SL_DIR = DATA_ROOT + 'single_line/'
ML_DIR = DATA_ROOT + 'multiple_lines/'
DEMO_DIR = DATA_ROOT + 'demo/'
CHAR_FILE = DATA_ROOT + 'char_data.txt'
SL_FILE = DATA_ROOT + 'sl_data.txt'
ML_FILE = DATA_ROOT + 'ml_data.txt'
IMAGE_ENCODING = '.png'
NUM_CHARS = len(string.ascii_letters + string.digits + ' ')

TEXT_INFO = {
    '/Library/Fonts/consola.ttf': {'size': 18, 'dir': 'consola/', 'model': 'model-consolas'},
    '/Library/Fonts/Calibri.ttf': {'size': 20, 'dir': 'Calibri/', 'model': 'model-calibri'}
}

TEXT_FONT = '/Library/Fonts/Calibri.ttf' # Change this
TEXT_SIZE = TEXT_INFO[TEXT_FONT]['size']
CURRENT_CHAR_DIR = CHAR_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_SL_DIR = SL_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_ML_DIR = ML_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_DEMO_DIR = DEMO_DIR + TEXT_INFO[TEXT_FONT]['dir']
CURRENT_MODEL_DIR = TEXT_INFO[TEXT_FONT]['model']
NN_DIR = 'ocrnet'
MODEL_CLASSES = CURRENT_MODEL_DIR + '/_classes.txt'

NUM_CHARS = len(string.ascii_letters + string.digits + ' ')    #Number of possible characters
MAX_CHARS_PER_BLOCK = 64           #Max # characters per block
IMAGE_SIZE_FOR_CNN = [20, 20, 3]       #Image size for CNN
NUM_NETS = 1               #Use N networks with a bagging approach
#Shared placeholders for images and target data
TFIM = tf.placeholder("float", [None] + IMAGE_SIZE_FOR_CNN, name = 'TFIM') 
TFYH = tf.placeholder("float", [None, NUM_CHARS], name = 'TFYH')

# room to improve the architecture - this seems to be the only major change made by the 819
# group we found on github. We should do this as well, look to make it even better, but
# also try to improve other things
# could also improve learning rate, #epochs, optimizer, add dropout layers, etc.    
def MakeNet(nn = 'ocrnet'):
    #Architecture of the neural network
    #The input volume is reduce to the shape of the output in conv layers
    #18 / 2 * 3 * 3 = 1 and 640 / 2 * 5 = 64 output.shape
    network_architecture = [('C', [5, 5,  3, NUM_CHARS // 2], [1, 2, 2, 1]), ('AF', 'relu'),     
          ('C', [4, 4, NUM_CHARS // 2, NUM_CHARS], [1, 6, 2, 1]), ('AF', 'relu'), 
          ('C', [3, 5, NUM_CHARS, NUM_CHARS], [1, 3, 5, 1]), ('AF', 'relu'),
          ('R', [-1, NUM_CHARS])]
    #Create the neural network in TensorFlow
    return ANNC(IMAGE_SIZE_FOR_CNN, network_architecture, batchSize = 10, learnRate = 2e-5, maxIter = 5, name = nn, reg = 1e-5, tol = 1e-2, verbose = True, X = TFIM, Y = TFYH)

def LoadNet():
    CNN = MakeNet('ocrnet')
    if not CNN.RestoreModel(CURRENT_MODEL_DIR + '/', NN_DIR):
        images, gt, image_names = load_data_train()
        # print(images.shape)
        # print(gt.shape)
        # exit()
        shuffled_data = ShuffleSplit(n_splits=1, random_state=123)
        train, test = next(shuffled_data.split(images))

        # train
        CNN.fit(images[train], gt[train])

        # get prediction sequences
        predictions = []
        for image in images:
            predictions.append(CNN.predict(np.expand_dims(image, axis=0)))
        # for i in np.array_split(np.arange(images.shape[0]), 32): # 32 x 32
        #     predictions.append(CNN.predict(images[i]))
        #     print(i)
        #     print(images[i])
        predictions = np.vstack(predictions)
        prediction_string = np.array([''.join(seq) for seq in predictions])

        # Compute accuracy
        train_acc = accuracy(prediction_string[train], gt[train])
        test_acc = accuracy(prediction_string[test], gt[test])
        print('\nTrain accuracy: ' + str(train_acc))
        print('Test accuracy: ' + str(test_acc))

        # Save model for next time
        CNN.SaveModel(os.path.join(CURRENT_MODEL_DIR, NN_DIR))
        with open(MODEL_CLASSES, 'w') as classes_file:
            classes_file.write('\n'.join(CNN._classes))

    else:
        # print('hi')
        with open(MODEL_CLASSES) as classes_file:
            CNN.RestoreClasses(classes_file.read().splitlines())
            # print(type(CNN))
        return CNN

def accuracy(predictions, ground_truth):
    '''
    Calculates the accuracy of the predictions compared to the ground truths
    predictions: np.array, The text predictions for a set of images
    ground_truth: np.array, The ground truth text for a set of images
    '''
    result = sum(sum(i == j for i, j in zip(guess, truth)) / len(guess) 
                            for guess, truth in zip(predictions, ground_truth)) / len(predictions)
    return result

def split_image_into_characters(file):
    testImage = ImageReader(file)
    # testImage.show_image(testImage.image)
    lines = testImage.get_lines_from_paragraph()
    letters_positions = []
    character_images = []
    for line in range(len(lines)):
        letters_positions.append(lines[line].get_character_positions_from_single_line())
        # Get each character on the line
        character_images.append(lines[line].characters_positions(letters_positions[line]))
    return character_images

def load_data_test(test_dir=CURRENT_SL_DIR, test_file=SL_FILE):
    '''
    Loads and prepares the dataset for training and testing.
    train_dir: string, The path to directory with the training images
    train_file: string, The file path to the image and ground truth
    Returns:
    Image Matrix, Ground Truth with Padding Matrix, Original Ground Truth Matrix,
    Image File Name Matrix
    '''
    images, gt, image_names = [], [], []

    with open(test_file) as test_data:
        for line in test_data:
            filename, text = line.strip().split(IMAGE_ENCODING + ' ')
            filename = filename + IMAGE_ENCODING
            # print('text', len(text))
            splits = split_image_into_characters(test_dir + filename)
            # print('Splits', len(splits))
            for line in splits:
                for image in line:
                    # image = image[:20, :20, :3]
                    image = np.resize(image, (20, 20, 3))
                    images.append(image)

            for char in text:
                if char != '\\n':
                    gt.append(char)
            gt.append(text)
            image_names.append(filename)
    imgs = np.stack(images)
    return np.stack(images), np.stack(gt), np.stack(image_names)

def load_data_train(train_dir=CURRENT_CHAR_DIR, train_file=CHAR_FILE):
    images, gt, image_names = [], [], []

    with open(train_file) as train_data:
        for line in train_data:
            filename, text = line.split(IMAGE_ENCODING + ' ')
            filename = filename + IMAGE_ENCODING
            images.append(imread(train_dir + filename))
            text = text[0]
            gt.append(text)
            image_names.append(filename)

    return np.stack(images), np.stack(gt), np.stack(image_names)
    
# CNN = LoadNet()
# YHL = [CNNi.YHL for CNNi in CNNs]    #Prediction placeholders
# TFS = CNNs[-1].GetSes()              #Get tensorflow session
# if __name__ == "__main__":
#     P = argparse.ArgumentParser(description = 'Deep learning based OCR')
#     P.add_argument('-f', action = 'store_true', help = 'Force model training')
#     P.add_argument('Img', metavar = 'I', type = str, nargs = '+', help = 'Image files')
#     PA = P.parse_args()
#     if PA.f:
#         FitModel()
#     for img in PA.Img:
#         I = imread(img)[:, :, :3]    #Read image and discard alpha
#         S = ImageToString(I)
#         print(S)

# this is just to test stuff
if __name__ == "__main__":
    # images, gt_padding, gt, image_names = load_data_ocr()
    # plt.imshow(images[0])
    # plt.show()
    CNN = LoadNet()
    images, gt, image_names = load_data_test()
    predictions = []
    for image in images:
        predictions.append(CNN.predict(np.expand_dims(image, axis=0)))
    predictions = np.vstack(predictions)
    prediction_string = np.array([''.join(seq) for seq in predictions])
    test_acc = accuracy(prediction_string, gt)
    print(test_acc)
