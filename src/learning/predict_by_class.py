import numpy as np

# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
# Backend
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping, History

from os import listdir
import os, os.path



from inception_v4_variant_update import *
import sets
import random
import numpy as np
import time
import pickle
import copy
import cv2


num_classes = 6
train_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
validation_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
test_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/demo/test/'
# model_save_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
# model_save_name = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features_update_captured_random_random_selection.h5'
batch_size = 32
img_height = 299
img_width = 299
weights_in_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
weights_in = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features_update_captured_sequential_order_patience_2_cell_phone.h5'
weights_path = weights_in_path + weights_in
captured_images_path = '/home/scatha/research_ws/src/lifelong_object_learning/data/captured_cropped/'

training_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/captured_growing/'
object_classes = ["bowl", "calculator", "cell_phone", "coffee_mug", "notebook", "plate"]
num_instances = 4
num_instance_images = 100
epochs = 50
patience = 0
pickle_filepath = '/home/scatha/research_ws/src/lifelong_object_learning/results/'
pickle_filename = 'random_random_selection_pickle.p'

nb_train_samples = 0
nb_validation_samples = 0

testing_dirs = ['/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-all_by_class']
                # '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-last-instance_by_class',
                # '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-all_by_class',
                # '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-last-instance_by_class']
test_sizes = [3000,
               600,
             18584,
              3812]


# stats
total_metrics = []
training_time = []
total_num_images = 0
instances_seen = []




model = create_model(weights_path=weights_path, num_classes=num_classes, weights='imagenet')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None)

# test_datagens = [ImageDataGenerator() for i in testing_dirs]

for i in range(len(testing_dirs)):
    for object_class in object_classes:

        test_datagen = ImageDataGenerator()
        testing_dir = testing_dirs[i] + "/" + object_class
        img_dir = testing_dir + "/" + object_class
        # test_size = len([name for name in os.listdir(img_dir) if os.path.isfile(name)])
        test_size = 500

        testing_generator = test_datagen.flow_from_directory(
            testing_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        metrics = model.evaluate_generator(
            testing_generator,
            steps=test_size//batch_size)
        print metrics
    print