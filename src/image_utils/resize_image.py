from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import cv2

image_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/demo/bowl/'

for image_file in listdir(image_dir):

	image = cv2.imread(image_dir + image_file)
	new_image = cv2.resize(image, (299,299))

	cv2.imwrite(image_dir + image_file, new_image)