# 6.819-OCR-Final-Project

An Optical Character Recognition system using a Convolutional Neural Network.
More functionality potentially to come.

Alec Kushner, Jared Schwait- Final Project for 6.819 Fall 2018

Getting Started:

Install python- https://www.python.org/downloads/
Install pytorch- https://pytorch.org/

Both can be found on their website

Clone the repo into your own directory.

Once cloned, to break paragraphs and/or lines of text up, use the ImageWordProcessing submodule.

Set the various parameters and change desired specifications in the neural network architecture.

Everything can be done with the generate images file and the integrated_ocr_system file. In generate images, simply go to the main method at the bottom, and decide how many images to generate for each type of file. The character images are used to train the ocr network, while the SL and ML (single line and multi-line) images will be used as test images. If you wish to try this with other fonts, simply ensure you have the fonts installed, add the font and size to the 'TEXT_INFO' dictionary at the top and change the 'TEXT_FONT' variable to that font.

Once you have generated images you can run the main method of integrated_ocr_system. This simply loads the net, loads the data and then makes the predictions on the images with the model. If the net cannot be properly loaded, the net will be created with the network architecture in the MakeNet method and trained on the single character dataset. Ensure that the constants at the top of this file match up with the font you have chosen to use and have generated images for.
