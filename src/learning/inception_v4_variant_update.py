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

from os import listdir

#########################################################################################
# Implements the Inception Network v4 (http://arxiv.org/pdf/1602.07261v1.pdf) in Keras. #
#########################################################################################

WEIGHTS_PATH = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def conv2d_bn(x, nb_filter, num_row, num_col,
              padding='same', strides=(1, 1), use_bias=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, (num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_regularizer=regularizers.l2(0.00004),
                      kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
    x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_a(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, strides=(2,2), padding='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_b(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_b(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, strides=(2,2), padding='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_c(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def inception_v4_base(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, strides=(2,2), padding='valid')
    net = conv2d_bn(net, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, strides=(2,2), padding='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, padding='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, padding='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, strides=(2,2), padding='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
    	net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
    	net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
    	net = block_inception_c(net)

    return net


def inception_v4(num_classes, dropout_keep_prob, weights, include_top, weights_path):
    '''
    Creates the inception v4 network
    Args:
    	num_classes: number of classes
    	dropout_keep_prob: float, the fraction to keep before final layer.
    
    Returns: 
    	logits: the logits outputs of the model.
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_data_format() == 'channels_first':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    x = inception_v4_base(inputs)


    # Final pooling and prediction
    if include_top:
        # 1 x 1 x 1536
        x = AveragePooling2D((8,8), padding='valid')(x)
        x = Dropout(dropout_keep_prob)(x)
        x = Flatten()(x)
        print x.shape
        # 1536
        x = Dense(units=1001, activation='softmax')(x)

    model = Model(inputs, x, name='inception_v4')

        # pop off last layer and add num_classes classification layer
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    x = Dense(num_classes, activation='softmax', name='predictions')(model.layers[-1].output)
    x.trainable = True
    model = Model(inputs, x)

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        # if include_top:
        #     weights_path = get_file(
        #         'inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
        #         WEIGHTS_PATH,
        #         cache_subdir='models',
        #         md5_hash='9fe79d77f793fe874470d84ca6ba4a3b')
        # else:
        #     weights_path = get_file(
        #         'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #         WEIGHTS_PATH_NO_TOP,
        #         cache_subdir='models',
        #         md5_hash='9296b46b5971573064d12e4669110969')
        model.load_weights(weights_path, by_name=True)




    return model


def create_model(weights_path, num_classes=1001, dropout_prob=0.2, weights=None, include_top=True):
    return inception_v4(num_classes, dropout_prob, weights, include_top, weights_path)


if __name__ == '__main__':

    num_classes = 6
    train_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
    validation_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/training_data/rgbd-dataset/train'
    test_data_dir = '/home/scatha/research_ws/src/lifelong_object_learning/data/demo/test/'
    model_save_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
    model_save_name = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features_update_rgbd.h5'
    batch_size = 32
    img_height = 299
    img_width = 299
    weights_in_path = '/home/scatha/research_ws/src/lifelong_object_learning/model_weights/'
    weights_in = 'inception_v4_flickr_base_weights_b32_e50_tr100_fixed_imgnet_features.h5'
    weights_path = weights_in_path + weights_in

    model = create_model(weights_path=weights_path, num_classes=num_classes, weights='imagenet')

    # batch_size = 16

    # prepare data augmentation configuration       ## FUTURE: more data augmentation?
    train_datagen = ImageDataGenerator(
            # rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # test_datagen = ImageDataGenerator() #rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    # validation_generator = test_datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(img_height, img_width),
    #         batch_size=batch_size,
    #         class_mode='categorical')


    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None)
    # model.fit(X, Y, batch_size=32, nb_epoch=10, verbose=1, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    # model.evaluate(test_X, test_Y, batch_size=32, verbose=1, sample_weight=None)


    # nb_train_samples = 3000
    # epochs = 50
    # nb_validation_samples = 1000

    # nb_train_samples = 4800         # 800*num_classes
    nb_train_samples = 14782
    epochs = 50
    # nb_validation_samples = 1200    # 200*num_classes

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs)
        # validation_data=validation_generator,
        # validation_steps=nb_validation_samples // batch_size)

    model.save_weights(model_save_path + model_save_name)


    # # predictions on test data
    # test_X = []
    # for filename in listdir(test_data_dir):
    #     img = load_img(test_data_dir + filename, target_size=(299, 299))
    #     x = img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x[0])
    #     test_X.append(x)
    # test_X = np.array(test_X)

    # predicted_vals = model.predict(test_X, batch_size=1, verbose=0)
    # for predicted_val in predicted_vals:
    #     print predicted_val


    # model.fit(X, Y, batch_size=32, nb_epoch=10, verbose=1, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    # # model.evaluate(test_X, test_Y, batch_size=16, verbose=1, sample_weight=None)

    # predicted_vals = model.predict(test_X, batch_size=16, verbose=0)
    # poses = sets.Set()
    
    # misclassifications = 0.0
    # total = 0.0
    # for i in range(len(test_Y)):
    #     truth = 0
    #     if test_Y[i][1] > test_Y[i][0]:
    #         truth = 1

    #     predicted = 0
    #     if predicted_vals[i][1] > predicted_vals[i][0]:
    #         predicted = 1

    #     if truth != predicted:
    #         misclassifications += 1.0
    #         if test_poses[i] not in poses:
    #             poses.add(test_poses[i])

    #     total += 1.0

    
    # poses_file = '/home/morgul/data_gatherer/src/data_collector/data/pose_file.txt'
    # f = open(poses_file, 'w')
    # f.write(str(len(poses)) + "\n")
    # for pose in poses:
    #     f.write(str(pose[0]) + "\t" + str(pose[1]) + "\n")
    # f.close()


    # accuracy = (total - misclassifications)/total
    # print("Num misclassifications: " + str(misclassifications))
    # print("Accuracy: " + str(accuracy))
    # print ("Num locations messed up: " + str(len(poses)))
