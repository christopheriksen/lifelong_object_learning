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
model_save_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
model_save_name = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features_update_captured_random_order_random_selection_patience_2_v2.h5'
batch_size = 32
img_height = 299
img_width = 299
weights_in_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
weights_in = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features.h5'
weights_path = weights_in_path + weights_in
captured_images_path = '/home/scatha/research_ws/src/lifelong_object_learning/data/captured_cropped/'

training_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/captured_growing/'
object_classes = ["bowl", "calculator", "cell_phone", "coffee_mug", "notebook", "plate"]
num_instances = 4
num_instance_images = 100
epochs = 50
patience = 2
pickle_filepath = '/home/scatha/research_ws/src/lifelong_object_learning/results/'
# pickle_filename = 'random_order_patience_4_pickle.p'

nb_train_samples = 0
nb_validation_samples = 0

testing_dirs = ['/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-all',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/captured-cropped-last-instance',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-all',
                '/home/scatha/research_ws/src/lifelong_object_learning/data/testing_data/rgbd-dataset-last-instance']
test_sizes = [3000,
               600,
             18584,
              3812]


# stats
total_metrics = []
training_time = []
total_num_images = 0
instances_seen = []
total_train_time = 0.0
train_time = 0.0

# load model
model = create_model(weights_path=weights_path, num_classes=num_classes, weights='imagenet')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None)


# prepare data augmentation configuration       ## FUTURE: more data augmentation?
train_datagen = ImageDataGenerator(
        # rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator() #rescale=1./255)

test_datagens = [ImageDataGenerator() for i in testing_dirs]




# create list of instances to direct data capture over 
instance_list = []
for object_class in object_classes:
    for instance_num in range(num_instances):
        instance = object_class + "_" + str(instance_num+1)
        instance_list.append(instance)

while len(instance_list) != 0:

    # # sequential order
    # instance = instance_list[0]

    # random order
    rand_index = random.randint(0, len(instance_list)-1)
    instance = instance_list[rand_index]


    instance_list.remove(instance)
    print "total num images: " + str(total_num_images)
    print "num instances seen: " + str(len(instances_seen))
    print "instance: " + instance
    instances_seen.append(instance)

    object_class = instance[:-2]
    object_class_index = -1
    for i in range(len(object_classes)):
        if object_class == object_classes[i]:
            object_class_index = i
    instance_img_dir = captured_images_path + object_class + "/" + instance + "/images/"

    # predictions on data
    instance_filenames = listdir(instance_img_dir)
    instance_imgs = []

    for filename in instance_filenames:
        img = load_img(instance_img_dir + filename, target_size=(299, 299))
        x = img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x[0])
        instance_imgs.append(x)
    instance_imgs = np.array(instance_imgs)
    x = np.expand_dims(x, axis=0)
    reduced_instance_imgs = copy.deepcopy(instance_imgs)



    ## while prediction accuracy on instance_imgs is still improving:
    instance_confidences = []
    improving = True
    while improving == True:

        # # find most uncertain image
        predicted_vals = model.predict(reduced_instance_imgs, batch_size=1, verbose=0)
        # min_confidence = float("inf")
        # min_index = None
        # for index in range(len(predicted_vals)):
        #     confidence = predicted_vals[index][object_class_index]
        #     if confidence < min_confidence:
        #         min_confidence = confidence
        #         min_index = index

        # random selection
        rand_index = random.randint(0, len(predicted_vals)-1)
        min_index = rand_index

        instance_filename = instance_filenames[min_index]
        reduced_instance_imgs = np.delete(reduced_instance_imgs, min_index, axis=0)
        instance_filenames.remove(instance_filename)

        ## add least confident image to training set
        new_filename = training_dir + object_class + "/" + instance_filename
        img = cv2.imread(instance_img_dir + instance_filename,cv2.IMREAD_UNCHANGED)
        cv2.imwrite(new_filename,img)
        while cv2.imread(new_filename) is None:
            time.sleep(.01)

        nb_train_samples += 1
        nb_validation_samples += 1

        # batch_size=min(batch_size,nb_train_samples)

        # retrain model
        t0 = time.time()
        train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
                training_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=max(nb_train_samples // batch_size, 1),
            epochs=epochs,
            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'), History()],
            validation_data=validation_generator,
            validation_steps=max(nb_validation_samples // batch_size, 1))

        train_time = time.time() - t0
        training_time.append(train_time)
        total_num_images += 1
        total_train_time += train_time
        

        # model.save_weights(model_save_path + model_save_name)

        ## test model on instance_imgs
        predicted_vals = model.predict(instance_imgs, batch_size=1, verbose=0)
        instance_vals = predicted_vals[:,object_class_index]
        ave_confidence = np.sum(instance_vals)/float(len(instance_imgs))

        if len(instance_confidences) < (patience+1):
            instance_confidences.append(ave_confidence)
        else:
            for instance_confidence in instance_confidences:
                improving = False
                if ave_confidence > instance_confidence:
                    improving = True
            instance_confidences.remove(instance_confidences[0])
            instance_confidences.append(ave_confidence)


        ### also keep track of stats

        # test set performance after each training instance view
        # view_metrics = []
        # for i in range(len(testing_dirs)):
        #     test_datagen = test_datagens[i]
        #     testing_dir = testing_dirs[i]
        #     test_size = test_sizes[i]

        #     testing_generator = test_datagen.flow_from_directory(
        #         testing_dir,
        #         target_size=(img_height, img_width),
        #         batch_size=batch_size,
        #         class_mode='categorical')

        #     metrics = model.evaluate_generator(
        #         testing_generator,
        #         steps=test_size//batch_size)
        #     view_metrics.append(metrics)
        #     print metrics
        # total_metrics.append(view_metrics)


model.save_weights(model_save_path + model_save_name)

# p_list = [total_num_images, training_time, total_metrics]
# pickle.dump(p_list, open( pickle_filepath + pickle_filename, "w"))

print "total num images: " + str(total_num_images)
print "final train time: " + str(train_time)
print "total train time: " + str(total_train_time)
print "ave train time: " + str(total_train_time/total_num_images)

print "instances seen: "
for instance in instances_seen:
    print instance






